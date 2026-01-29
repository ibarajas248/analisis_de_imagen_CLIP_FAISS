import os
import numpy as np
import pandas as pd
import streamlit as st

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import faiss

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Búsqueda visual (CLIP + FAISS)", layout="wide")

DEFAULT_INDEX_PATH = "index.faiss"      # <-- cambia si tu index se llama distinto
DEFAULT_META_PATH  = "metadata.csv"     # <-- cambia si tu df viene de otra ruta
DEFAULT_DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Utils
# ---------------------------
@st.cache_resource
def load_clip(model_name: str, device: str):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    return model, processor

@st.cache_resource
def load_faiss_index(index_path: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No existe el índice FAISS: {index_path}")
    return faiss.read_index(index_path)

@st.cache_data
def load_metadata(meta_path: str):
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No existe metadata: {meta_path}")

    # Soporta CSV o Parquet
    if meta_path.lower().endswith(".parquet"):
        df = pd.read_parquet(meta_path)
    else:
        df = pd.read_csv(meta_path)

    # Requisito: que exista 'id' para mapear contra FAISS IDs
    if "id" not in df.columns:
        raise ValueError("Tu metadata debe tener una columna 'id' (ID que coincide con los IDs del índice).")
    return df

def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)

def text_embedding(text: str, model, processor, device: str) -> np.ndarray:
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        v = model.get_text_features(**inputs)
        v = normalize(v)
    return v.detach().cpu().numpy().astype("float32")

def image_embedding(pil_img: Image.Image, model, processor, device: str) -> np.ndarray:
    pil_img = pil_img.convert("RGB")
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        v = model.get_image_features(**inputs)
        v = normalize(v)
    return v.detach().cpu().numpy().astype("float32")

def search_faiss(q: np.ndarray, index, df: pd.DataFrame, k: int) -> pd.DataFrame:
    D, I = index.search(q, k)  # D: scores, I: ids
    ids = I[0].astype("int64").tolist()
    scores = D[0].tolist()

    df_idx = df.set_index("id")
    rows = []
    for idx, score in zip(ids, scores):
        if idx in df_idx.index:
            row = df_idx.loc[idx].to_dict()
            row["id"] = idx
            row["score"] = float(score)
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("score", ascending=False)

def render_results(df_res: pd.DataFrame, max_cols_preview=None):
    if df_res is None or len(df_res) == 0:
        st.info("No hubo resultados.")
        return

    # Columnas comunes (si existen)
    preferred = [c for c in ["id", "title", "artist", "name", "score", "image_url", "url", "ruta", "path"] if c in df_res.columns]
    if not preferred:
        preferred = df_res.columns.tolist()

    st.dataframe(df_res[preferred], use_container_width=True)

    # Vista de tarjetas con imagen si existe
    img_col = None
    for c in ["image_url", "url", "img_url"]:
        if c in df_res.columns:
            img_col = c
            break

    if img_col:
        st.subheader("Vista visual")
        cols = st.columns(4)
        for i, row in df_res.head(12).iterrows():
            col = cols[int(i) % 4]
            with col:
                st.caption(f"ID {row.get('id')} — score {row.get('score'):.3f}")
                st.image(row.get(img_col), use_container_width=True)
                # Muestra algún título si existe
                title = row.get("title") or row.get("name")
                if title:
                    st.write(title)

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Configuración")

index_path = st.sidebar.text_input("Ruta índice FAISS (.index/.faiss)", value=DEFAULT_INDEX_PATH)
meta_path  = st.sidebar.text_input("Ruta metadata (CSV/Parquet)", value=DEFAULT_META_PATH)

model_name = st.sidebar.selectbox(
    "Modelo CLIP",
    options=["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
    index=0
)

device = st.sidebar.selectbox("Device", options=[DEFAULT_DEVICE, "cpu"], index=0)
k = st.sidebar.slider("Top-K resultados", min_value=3, max_value=50, value=12)

# ---------------------------
# Load resources
# ---------------------------
st.title("🔎 Búsqueda por similitud (texto e imagen) — CLIP + FAISS")

try:
    model, processor = load_clip(model_name, device)
    index = load_faiss_index(index_path)
    df = load_metadata(meta_path)
except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    st.stop()

st.success(f"Recursos listos ✅  |  device={device}  |  index.ntotal={index.ntotal}  |  filas metadata={len(df)}")

# ---------------------------
# Tabs: texto / imagen
# ---------------------------
tab_text, tab_img = st.tabs(["Buscar por texto", "Buscar por imagen"])

with tab_text:
    st.subheader("Consulta por texto")
    query = st.text_input("Escribe una descripción (ej: 'paisaje nocturno con luna', 'retrato barroco', etc.)")
    if st.button("Buscar (texto)", type="primary") and query.strip():
        q = text_embedding(query.strip(), model, processor, device)
        res = search_faiss(q, index, df, k=k)
        render_results(res)

with tab_img:
    st.subheader("Consulta por imagen")
    uploaded = st.file_uploader("Sube una imagen (JPG/PNG/WebP)", type=["jpg", "jpeg", "png", "webp"])

    colA, colB = st.columns([1, 1])

    with colA:
        img_path = st.text_input("O pon ruta local (solo si el servidor tiene acceso a ese archivo)", value="")

    with colB:
        go = st.button("Buscar (imagen)", type="primary")

    pil_img = None
    if uploaded is not None:
        pil_img = Image.open(uploaded)
        st.image(pil_img, caption="Imagen cargada", use_container_width=True)
    elif img_path.strip():
        try:
            pil_img = Image.open(img_path.strip())
            st.image(pil_img, caption="Imagen desde ruta", use_container_width=True)
        except Exception as e:
            st.warning(f"No pude abrir esa ruta: {e}")

    if go:
        if pil_img is None:
            st.warning("Primero sube una imagen o pon una ruta válida.")
        else:
            q = image_embedding(pil_img, model, processor, device)
            res = search_faiss(q, index, df, k=k)
            render_results(res)
