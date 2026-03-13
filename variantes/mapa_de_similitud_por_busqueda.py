# app.py
# Streamlit — Buscador (CLIP texto o CLIP imagen) -> Resultados -> Mapa 2D (PCA) + Detalle en la MISMA FILA
#
# ✅ Layout secuencial (NO columnas) en resultados/cards (secuencial hacia abajo)
# ✅ Mapa 2D + Detalle en una sola fila (columns)
# ✅ FIX CLIP robusto:
#    - Texto: text_model -> pooler_output -> text_projection -> normalize
#    - Imagen: vision_model -> pooler_output -> visual_projection -> normalize
# ✅ Modo Texto→Imagen e Imagen→Imagen (subir imagen)
# ✅ Detalle: IMAGEN primero, luego textos (render_card reordenado)
#
# Requiere:
#   pip install streamlit plotly faiss-cpu mysql-connector-python numpy pandas requests pillow torch transformers

from __future__ import annotations

import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import faiss
import requests
from PIL import Image

from mysql.connector.pooling import MySQLConnectionPool

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="CLIP+FAISS — Buscar → Mapa 2D", layout="wide")
st.title("CLIP + FAISS — Buscar → Resultados → Mapa 2D → Detalle")

st.markdown(
    """
<style>
.card {
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 14px;
  padding: 14px;
  background: rgba(255,255,255,0.03);
}
.small { opacity: 0.75; font-size: 0.9rem; }
.kv { margin: 6px 0; }
.kv b { opacity: 0.85; }
hr.sep { border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0; }
</style>
""",
    unsafe_allow_html=True,
)


# -------------------------
# Secrets (toml)
# -------------------------
INDEX_PATH = st.secrets["app"]["index_path"]
MODEL_NAME = st.secrets["app"].get("model_name", "openai/clip-vit-base-patch32")
DEFAULT_QUERY = st.secrets["app"].get("default_query", "portrait, black and white photo")
TOP_K_DEFAULT = int(st.secrets["app"].get("top_k_default", 50) or 50)

MYSQL_CFG = st.secrets["mysql"]
MYSQL_HOST = MYSQL_CFG["host"]
MYSQL_PORT = int(MYSQL_CFG.get("port", 3306))
MYSQL_USER = MYSQL_CFG["user"]
MYSQL_PASSWORD = MYSQL_CFG["password"]
MYSQL_DB = MYSQL_CFG["database"]


# -------------------------
# Helpers
# -------------------------
def safe_str(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    return str(x).strip()


@st.cache_resource
def load_faiss_index(path: str):
    return faiss.read_index(path)


def base_index(idx):
    return idx.index if hasattr(idx, "index") else idx


@st.cache_resource
def get_mysql_pool() -> MySQLConnectionPool:
    return MySQLConnectionPool(
        pool_name="obras_pool",
        pool_size=5,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        charset="utf8mb4",
        use_unicode=True,
        autocommit=True,
    )


def mysql_fetch_df(sql: str, params: Optional[Tuple] = None) -> pd.DataFrame:
    pool = get_mysql_pool()
    conn = pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params or ())
        rows = cur.fetchall()
        return pd.DataFrame(rows)
    finally:
        try:
            conn.close()
        except Exception:
            pass


@st.cache_data(show_spinner=False)
def fetch_metadata_for_ids(ids: Tuple[int, ...]) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame()

    chunk_size = 900
    dfs = []
    sql_base = """
        SELECT
          id, source, source_url,
          title, artist, date_text, year_start, year_end,
          technique, medium, dimensions,
          description,
          image_url, image_full_url,
          is_public_domain, updated_at
        FROM obras_arte
        WHERE id IN ({})
    """

    ids_list = list(ids)
    for i in range(0, len(ids_list), chunk_size):
        chunk = ids_list[i : i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        sql = sql_base.format(placeholders)
        dfs.append(mysql_fetch_df(sql, tuple(chunk)))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_year_label(row: pd.Series) -> str:
    ys = row.get("year_start")
    ye = row.get("year_end")
    try:
        if pd.notna(ys) and pd.notna(ye):
            ys_i = int(float(ys))
            ye_i = int(float(ye))
            if ys_i != ye_i:
                return f"{ys_i}–{ye_i}"
            return f"{ys_i}"
        if pd.notna(ys):
            return f"{int(float(ys))}"
    except Exception:
        pass
    return safe_str(row.get("date_text"))


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str, max_mb: int = 12) -> Optional[bytes]:
    url = safe_str(url)
    if not url:
        return None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitBot/1.0)"}
    try:
        r = requests.get(url, headers=headers, timeout=18, stream=True)
        r.raise_for_status()
        content = r.content
        if len(content) > max_mb * 1024 * 1024:
            return None
        return content
    except Exception:
        return None


# -------------------------
# Card (Detalle) — IMAGEN primero
# -------------------------
def render_card(row: pd.Series):
    title = safe_str(row.get("title")) or "(Sin título)"
    artist = safe_str(row.get("artist"))
    source = safe_str(row.get("source"))
    ylab = safe_str(row.get("year_label"))
    technique = safe_str(row.get("technique"))
    medium = safe_str(row.get("medium"))
    dims = safe_str(row.get("dimensions"))
    desc = safe_str(row.get("description"))
    source_url = safe_str(row.get("source_url"))
    img_url = safe_str(row.get("image_full_url")) or safe_str(row.get("image_url"))

    st.markdown('<div class="card">', unsafe_allow_html=True)

    # ✅ 1) IMAGEN PRIMERO
    if img_url:
        b = fetch_image_bytes(img_url)
        if b:
            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
                im.thumbnail((1200, 1200))
                st.image(im, caption="Imagen (server-side)", use_container_width=True)
            except Exception:
                st.info("No pude renderizar la imagen, pero la URL existe:")
                st.code(img_url)
        else:
            st.info("No pude descargar la imagen (timeout/bloqueada/muy grande):")
            st.code(img_url)
    else:
        st.warning("Esta obra no tiene image_url / image_full_url.")

    st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

    # ✅ 2) TITULO + ID
    st.markdown(f"### {title}")
    st.markdown(
        f'<div class="small">ID: <b>{int(row["id"])}</b></div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

    # ✅ 3) METADATOS
    if artist:
        st.markdown(f"<div class='kv'><b>Artista:</b> {artist}</div>", unsafe_allow_html=True)
    if ylab:
        st.markdown(f"<div class='kv'><b>Fecha:</b> {ylab}</div>", unsafe_allow_html=True)
    if source:
        st.markdown(f"<div class='kv'><b>Fuente:</b> {source}</div>", unsafe_allow_html=True)
    if technique:
        st.markdown(f"<div class='kv'><b>Técnica:</b> {technique}</div>", unsafe_allow_html=True)
    if medium:
        st.markdown(f"<div class='kv'><b>Medio:</b> {medium}</div>", unsafe_allow_html=True)
    if dims:
        st.markdown(f"<div class='kv'><b>Dimensiones:</b> {dims}</div>", unsafe_allow_html=True)

    # ✅ 4) LINK
    if source_url:
        st.markdown("<hr class='sep'/>", unsafe_allow_html=True)
        st.link_button("Abrir página de la obra", source_url)

    # ✅ 5) DESCRIPCION AL FINAL
    if desc:
        st.markdown("<hr class='sep'/>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='small'>{desc[:700]}{'…' if len(desc)>700 else ''}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------
# CLIP — robusto (texto e imagen)
# -------------------------
@st.cache_resource
def load_clip(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, device


@torch.inference_mode()
def embed_text(query: str) -> np.ndarray:
    model, processor, device = load_clip(MODEL_NAME)

    inputs = processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    text_outputs = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled = text_outputs.pooler_output
    feats = model.text_projection(pooled)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats.detach().cpu().numpy().astype(np.float32)


@torch.inference_mode()
def embed_image(pil_img: Image.Image) -> np.ndarray:
    model, processor, device = load_clip(MODEL_NAME)

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    inputs = processor(images=[pil_img], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    vision_outputs = model.vision_model(pixel_values=pixel_values)
    pooled = vision_outputs.pooler_output
    feats = model.visual_projection(pooled)
    feats = F.normalize(feats, p=2, dim=-1)
    return feats.detach().cpu().numpy().astype(np.float32)


# -------------------------
# Embeddings para PCA
# -------------------------
def reconstruct_by_id(index, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vecs = []
    ids_ok = []
    for _id in ids:
        _id = int(_id)
        if _id < 0:
            continue
        try:
            v = index.reconstruct(_id)
            vecs.append(v)
            ids_ok.append(_id)
        except Exception:
            continue

    if not vecs:
        return np.empty((0, int(index.d)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.vstack(vecs).astype(np.float32, copy=False), np.array(ids_ok, dtype=np.int64)


@st.cache_data(show_spinner=False)
def pca_2d_faiss(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    d = vectors.shape[1]
    pca = faiss.PCAMatrix(d, 2)
    pca.train(vectors)
    coords = pca.apply_py(vectors)
    return coords.astype(np.float32, copy=False)


# -------------------------
# Sidebar
# -------------------------
index = load_faiss_index(INDEX_PATH)
b = base_index(index)

st.sidebar.caption(f"Modelo: {MODEL_NAME}")
st.sidebar.caption(f"FAISS wrapper: {type(index).__name__}")
st.sidebar.caption(f"FAISS base: {type(b).__name__}")
st.sidebar.caption(f"ntotal={int(index.ntotal)} · d={int(index.d)}")

top_k = st.sidebar.slider("Top K resultados", 5, 200, TOP_K_DEFAULT, 5)
color_by_source = st.sidebar.checkbox("Color por source en el mapa", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("Limpiar selección"):
    st.session_state.pop("selected_id", None)


# -------------------------
# Estado
# -------------------------
if "results_df" not in st.session_state:
    st.session_state["results_df"] = pd.DataFrame()
if "results_vecs" not in st.session_state:
    st.session_state["results_vecs"] = None


# -------------------------
# 1) BUSCAR (Texto o Imagen)
# -------------------------
st.header("1) Buscar")

mode = st.radio("Modo de búsqueda", ["Texto → Imagen", "Imagen → Imagen"], horizontal=True)

query = ""
uploaded_img = None
pil_img = None

if mode == "Texto → Imagen":
    query = st.text_input("Texto", value=DEFAULT_QUERY)
else:
    uploaded_img = st.file_uploader(
        "Sube una imagen (jpg/png/webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )
    if uploaded_img is not None:
        try:
            pil_img = Image.open(uploaded_img).convert("RGB")
            st.image(pil_img, caption="Imagen de consulta", use_container_width=True)
        except Exception:
            pil_img = None
            st.error("No pude leer esa imagen. Prueba con otra (jpg/png).")

do_search = st.button("Buscar", type="primary")


def _build_results(ids: np.ndarray, scores: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    meta = fetch_metadata_for_ids(tuple(ids.tolist()))
    if meta.empty:
        df = pd.DataFrame({"id": ids, "score": scores})
    else:
        df = pd.DataFrame({"id": ids, "score": scores}).merge(meta, on="id", how="left")

    df["year_label"] = df.apply(compute_year_label, axis=1)
    df["title_show"] = df.get("title", "").fillna("").astype(str)
    df["artist_show"] = df.get("artist", "").fillna("").astype(str)
    df["source_show"] = df.get("source", "").fillna("").astype(str)

    vecs, ids_ok = reconstruct_by_id(index, df["id"].to_numpy(np.int64))
    if len(ids_ok) > 0:
        df = df[df["id"].isin(ids_ok)].copy()

    return df, vecs


def run_search_text(q: str, k: int):
    q = safe_str(q)
    if not q:
        st.warning("Escribe un texto para buscar.")
        return

    with st.spinner("Embedding CLIP (texto) + búsqueda en FAISS..."):
        qvec = embed_text(q)
        D, I = index.search(qvec, k)

    ids = I[0].astype(np.int64)
    scores = D[0].astype(np.float32)

    mask = ids >= 0
    ids = ids[mask]
    scores = scores[mask]

    df, vecs = _build_results(ids, scores)
    st.session_state["results_df"] = df
    st.session_state["results_vecs"] = vecs


def run_search_image(img: Image.Image, k: int):
    if img is None:
        st.warning("Sube una imagen para buscar.")
        return

    with st.spinner("Embedding CLIP (imagen) + búsqueda en FAISS..."):
        qvec = embed_image(img)
        D, I = index.search(qvec, k)

    ids = I[0].astype(np.int64)
    scores = D[0].astype(np.float32)

    mask = ids >= 0
    ids = ids[mask]
    scores = scores[mask]

    df, vecs = _build_results(ids, scores)
    st.session_state["results_df"] = df
    st.session_state["results_vecs"] = vecs


if do_search:
    if mode == "Texto → Imagen":
        run_search_text(query, int(top_k))
    else:
        run_search_image(pil_img, int(top_k))

df_res: pd.DataFrame = st.session_state["results_df"]
vecs_res = st.session_state["results_vecs"]

if df_res is None or df_res.empty:
    st.info("Haz una búsqueda para ver resultados.")
    st.stop()


# -------------------------
# 2) RESULTADOS
# -------------------------
st.header("2) Resultados")
show_cols = ["score", "title_show", "artist_show", "year_label", "source_show", "id"]
for c in show_cols:
    if c not in df_res.columns:
        df_res[c] = ""
st.dataframe(df_res[show_cols], use_container_width=True, height=420)


# -------------------------
# 3) MAPA 2D + DETALLE (MISMA FILA)
# -------------------------
st.header("3) Mapa 2D + Detalle (misma fila)")

if vecs_res is None or len(vecs_res) < 3:
    st.warning("Muy pocos vectores para PCA/mapa. Sube top_k o cambia la búsqueda.")
    st.stop()

coords = pca_2d_faiss(vecs_res)

df_map = df_res.copy()
df_map = df_map.iloc[: coords.shape[0]].copy()
df_map["x"] = coords[: len(df_map), 0]
df_map["y"] = coords[: len(df_map), 1]

hover_cols = {
    "id": True,
    "score": True,
    "title_show": True,
    "artist_show": True,
    "year_label": True,
    "source_show": True,
}

if color_by_source and "source_show" in df_map.columns:
    fig = px.scatter(
        df_map,
        x="x",
        y="y",
        color="source_show",
        hover_data=hover_cols,
        custom_data=["id"],
        render_mode="webgl",
    )
else:
    fig = px.scatter(
        df_map,
        x="x",
        y="y",
        hover_data=hover_cols,
        custom_data=["id"],
        render_mode="webgl",
    )

fig.update_layout(
    height=720,
    margin=dict(l=10, r=10, t=10, b=10),
    legend_title_text="source" if color_by_source else None,
)

col_map, col_det = st.columns([0.62, 0.38], gap="large")

with col_map:
    st.subheader("Mapa 2D (selecciona un punto)")
    plot_state = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="plot_2d",
    )

selected_id = None
try:
    sel = getattr(plot_state, "selection", None)
    if sel is None and isinstance(plot_state, dict):
        sel = plot_state.get("selection")
    if sel and sel.get("points"):
        cd = sel["points"][0].get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) > 0:
            selected_id = int(cd[0])
except Exception:
    selected_id = None

if selected_id is not None:
    st.session_state["selected_id"] = selected_id

with col_det:
    st.subheader("Detalle")
    if st.button("Limpiar selección", key="clear_sel_inline"):
        st.session_state.pop("selected_id", None)

    sid = st.session_state.get("selected_id")
    if not sid:
        st.info("Selecciona un punto del mapa para ver el detalle aquí.")
    else:
        hit = df_res[df_res["id"] == int(sid)]
        if hit.empty:
            st.warning("No encontré ese ID en los resultados actuales.")
        else:
            render_card(hit.iloc[0])