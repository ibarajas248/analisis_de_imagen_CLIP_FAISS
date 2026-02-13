import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
from transformers import CLIPModel, CLIPProcessor
import mysql.connector
from PIL import Image

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="CLIP + FAISS buscador", layout="wide")

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
.score { font-weight: 600; }
hr.sep { border: none; height: 1px; background: rgba(255,255,255,0.08); margin: 10px 0 0 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# Secrets
# -------------------------
INDEX_PATH = st.secrets["app"]["index_path"]
MODEL_NAME = st.secrets["app"]["model_name"]
DEFAULT_QUERY = st.secrets["app"].get("default_query", "gato")
TOP_K_DEFAULT = int(st.secrets["app"].get("top_k_default", 25))

MYSQL_HOST = st.secrets["mysql"]["host"]
MYSQL_PORT = int(st.secrets["mysql"]["port"])
MYSQL_USER = st.secrets["mysql"]["user"]
MYSQL_PASSWORD = st.secrets["mysql"]["password"]
MYSQL_DATABASE = st.secrets["mysql"]["database"]

# -------------------------
# Load FAISS + CLIP (cache)
# -------------------------
@st.cache_resource
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

@st.cache_resource
def load_clip(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return device, model, processor

def mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        autocommit=True,
        connection_timeout=20,
    )

# -------------------------
# Sources list (para filtro)
# -------------------------
@st.cache_data(ttl=300)
def fetch_sources() -> list[str]:
    conn = mysql_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source FROM obras_arte ORDER BY source")
        rows = cur.fetchall() or []
        return [r[0] for r in rows if r and r[0]]
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

# -------------------------
# Fetch metadata (MySQL) con filtro por source
# -------------------------
@st.cache_data(ttl=30)
def fetch_obras_by_ids(ids: list[int], sources: list[str] | None = None) -> pd.DataFrame:
    ids = [int(x) for x in ids if int(x) != -1]
    if not ids:
        return pd.DataFrame()

    params: list = list(ids)

    ids_placeholders = ",".join(["%s"] * len(ids))
    where = f"id IN ({ids_placeholders})"

    if sources:
        sources = [str(s) for s in sources if str(s).strip()]
        if sources:
            src_placeholders = ",".join(["%s"] * len(sources))
            where += f" AND source IN ({src_placeholders})"
            params.extend(sources)

    sql = f"SELECT * FROM obras_arte WHERE {where}"

    conn = mysql_conn()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        return pd.DataFrame(rows)
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()

# -------------------------
# Embeddings
# -------------------------
def text_embedding(query: str, device, model, processor) -> np.ndarray:
    """Texto -> embedding CLIP normalizado (1, d) float32."""
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    with torch.inference_mode():
        text_out = model.text_model(**inputs)
        pooled = text_out.pooler_output
        v = model.text_projection(pooled)
        v = v / v.norm(dim=-1, keepdim=True)

    return v.detach().cpu().numpy().astype("float32")

def image_embedding_from_pil(img: Image.Image, device, model, processor) -> np.ndarray:
    """Imagen (PIL) -> embedding CLIP normalizado (1, d) float32."""
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))

    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    proj = getattr(model, "visual_projection", None) or getattr(model, "vision_projection", None)

    with torch.inference_mode():
        if proj is not None:
            vision_out = model.vision_model(**inputs)  # requiere pixel_values
            v = proj(vision_out.pooler_output)         # (1, d)
        else:
            # fallback
            v = model.get_image_features(**inputs)

        v = v / v.norm(dim=-1, keepdim=True)

    return v.detach().cpu().numpy().astype("float32")  # (1, d)

# -------------------------
# Search (FAISS)
# -------------------------
def search_text(query: str, k: int, index, device, model, processor):
    q = text_embedding(query, device, model, processor)
    D, I = index.search(q, k)
    return [(int(i), float(s)) for i, s in zip(I[0].tolist(), D[0].tolist()) if int(i) != -1]

def search_by_image_pil(img: Image.Image, k: int, index, device, model, processor):
    q = image_embedding_from_pil(img, device, model, processor)
    D, I = index.search(q, k)
    return [(int(i), float(s)) for i, s in zip(I[0].tolist(), D[0].tolist()) if int(i) != -1]

# -------------------------
# Precision helpers (NO pierde precisión con filtro)
# -------------------------
def is_ip_metric(index) -> bool:
    try:
        return index.metric_type == faiss.METRIC_INNER_PRODUCT
    except Exception:
        return True  # CLIP normalizado típicamente usa IP/cosine

def build_best_score_map(pairs, index) -> dict[int, float]:
    """Mantiene el mejor score por rid (si hay duplicados)."""
    higher_is_better = is_ip_metric(index)
    best: dict[int, float] = {}
    for rid, score in pairs:
        rid = int(rid)
        if rid == -1:
            continue
        score = float(score)
        if rid not in best:
            best[rid] = score
        else:
            if higher_is_better:
                if score > best[rid]:
                    best[rid] = score
            else:
                if score < best[rid]:
                    best[rid] = score
    return best

def search_with_source_filter_adaptive(
    *,
    query_vec: np.ndarray,
    index,
    k_final: int,
    fetch_fn,                 # fetch_obras_by_ids(ids, sources=...)
    sources: list[str] | None,
    max_rounds: int = 8,
    start_fetch: int = 200,
    max_mysql_ids: int | None = None,  # limita query a MySQL (opcional)
) -> tuple[pd.DataFrame, dict]:
    """
    Busca en FAISS y filtra por source sin perder precisión:
    - overfetch adaptativo (dobla hasta completar k_final o agotar ntotal)
    - dedupe ids + conserva el mejor score por id
    """
    ntotal = int(getattr(index, "ntotal", 0) or 0)
    if ntotal <= 0:
        return pd.DataFrame(), {"rounds": 0, "k_fetch": 0, "unique": 0, "kept": 0}

    higher_is_better = is_ip_metric(index)

    k_fetch = min(ntotal, max(start_fetch, k_final))
    best_scores_all: dict[int, float] = {}
    df_best = pd.DataFrame()
    rounds = 0

    for _ in range(max_rounds):
        rounds += 1

        D, I = index.search(query_vec, k_fetch)
        pairs = [(int(i), float(s)) for i, s in zip(I[0].tolist(), D[0].tolist()) if int(i) != -1]

        # mantiene el mejor score por id en esta ronda
        best_round = build_best_score_map(pairs, index)

        # merge global: mejor score por id
        for rid, sc in best_round.items():
            if rid not in best_scores_all:
                best_scores_all[rid] = sc
            else:
                if higher_is_better:
                    if sc > best_scores_all[rid]:
                        best_scores_all[rid] = sc
                else:
                    if sc < best_scores_all[rid]:
                        best_scores_all[rid] = sc

        # ids ordenados por score global
        sorted_ids = sorted(
            best_scores_all.keys(),
            key=lambda rid: best_scores_all[rid],
            reverse=higher_is_better,
        )

        # limitamos lo que pedimos a MySQL para no armar un IN gigante
        limit_mysql = max_mysql_ids if max_mysql_ids is not None else k_fetch
        ask_ids = sorted_ids[: min(len(sorted_ids), limit_mysql)]

        df = fetch_fn(ask_ids, sources=sources)

        if not df.empty:
            df["score"] = df["id"].map(best_scores_all)
            df = df.dropna(subset=["score"]).sort_values("score", ascending=not higher_is_better)
            df_best = df.head(k_final).reset_index(drop=True)

        if len(df_best) >= k_final:
            break

        if k_fetch >= ntotal:
            break

        k_fetch = min(ntotal, k_fetch * 2)

    stats = {
        "rounds": rounds,
        "k_fetch": k_fetch,
        "unique": len(best_scores_all),
        "kept": len(df_best),
    }
    return df_best, stats

# -------------------------
# Render cards
# -------------------------
def render_cards(df_meta: pd.DataFrame, cols_n: int, show_n: int):
    cols = st.columns(cols_n, gap="large")
    n_cards = min(show_n, len(df_meta))

    for i in range(n_cards):
        row = df_meta.iloc[i]
        c = cols[i % cols_n]

        img_url = row.get("image_full_url") or row.get("image_url")
        title = row.get("title") or "(sin título)"
        artist = row.get("artist") or ""
        score = float(row.get("score") or 0.0)

        with c:
            st.markdown('<div class="card">', unsafe_allow_html=True)

            if isinstance(img_url, str) and img_url.strip():
                st.image(img_url, use_container_width=True)
            else:
                st.info("Sin imagen")

            st.markdown(f"**{title}**")
            if artist:
                st.markdown(f"<div class='small'>{artist}</div>", unsafe_allow_html=True)

            st.markdown(
                f"<div class='score'>score: {score:.4f} &nbsp;|&nbsp; id: {int(row['id'])}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

            with st.expander("Detalles"):
                st.json(row.to_dict(), expanded=False)
                src = row.get("source_url")
                if isinstance(src, str) and src.strip():
                    st.link_button("Abrir ficha (source_url)", src)

            st.markdown("</div>", unsafe_allow_html=True)



# -------------------------
# App
# -------------------------
st.title(" Buscador  (CLIP + FAISS )")

index = load_faiss_index(INDEX_PATH)

device, model, processor = load_clip(MODEL_NAME)

st.caption(f"Index ntotal={index.ntotal} dim={index.d} | modelo={MODEL_NAME} | device={device}")

# Detección correcta de IDMap
is_idmap = isinstance(index, faiss.IndexIDMap) or isinstance(index, faiss.IndexIDMap2)
if not is_idmap:
    st.warning("⚠️ Índice no IDMap: FAISS puede devolver posiciones, no obras_arte.id.")

with st.sidebar:
    st.subheader("Configuración")
    k = st.slider("Top K (final)", 5, 200, TOP_K_DEFAULT, 5)

    # ahora este slider es solo el "inicio" del overfetch adaptativo
    overfetch_mult = st.slider("Overfetch inicial (para filtrar)", 1, 30, 10, 1)

    cols_n = st.selectbox("Columnas de cards", [2, 3], index=0)
    show_n = st.slider("Cuántos cards mostrar", 1, 60, min(20, k), 1)

    all_sources = fetch_sources()
    selected_sources = st.multiselect(
        "Filtrar por obras_arte.source (colecciones)",
        options=all_sources,
        default=all_sources,
    )

tab_text, tab_img = st.tabs(["Texto → Imágenes", "Imagen → Imágenes"])

# -------------------------
# TAB 1: Texto -> Imágenes
# -------------------------
with tab_text:
    with st.form("search_form_text"):
        query = st.text_input("Buscar por texto", value=DEFAULT_QUERY)
        submitted = st.form_submit_button("Buscar")

    if submitted:
        query = (query or "").strip()
        if not query:
            st.info("Escribe algo para buscar.")
        else:
            q = text_embedding(query, device, model, processor)

            start_fetch = max(200, k * int(overfetch_mult))
            with st.spinner("Buscando..."):
                df_meta, stats = search_with_source_filter_adaptive(
                    query_vec=q,
                    index=index,
                    k_final=k,
                    fetch_fn=fetch_obras_by_ids,
                    sources=selected_sources,
                    max_rounds=8,
                    start_fetch=start_fetch,
                    max_mysql_ids=None,  # puedes poner p.ej. 3000 si MySQL se pone lento
                )

            if df_meta.empty:
                st.warning("No encontré filas en MySQL para esos IDs con el filtro de source.")
            else:
                st.success(
                    f"Resultados: {len(df_meta)} | rondas={stats['rounds']} | k_fetch_final={stats['k_fetch']} | ids_unicos={stats['unique']}"
                )
                render_cards(df_meta, cols_n=cols_n, show_n=show_n)

                with st.expander("Ver tabla completa"):
                    st.dataframe(df_meta, use_container_width=True)

# -------------------------
# TAB 2: Imagen -> Imágenes (upload)
# -------------------------
with tab_img:
    st.write("Sube una imagen (JPG/PNG/WebP) y buscamos obras visualmente similares.")

    uploaded = st.file_uploader("📤 Subir imagen", type=["png", "jpg", "jpeg", "webp"])

    with st.form("search_form_img"):
        submitted_img = st.form_submit_button("Buscar por imagen")

    if submitted_img:
        if uploaded is None:
            st.warning("Sube una imagen primero.")
        else:
            try:
                img_query = Image.open(uploaded).convert("RGB")
            except Exception as e:
                st.error(f"No pude abrir la imagen: {e}")
                st.stop()

            st.image(img_query, caption="Imagen de consulta", use_container_width=True)

            q = image_embedding_from_pil(img_query, device, model, processor)

            start_fetch = max(200, k * int(overfetch_mult))
            with st.spinner("Buscando por imagen..."):
                df_meta, stats = search_with_source_filter_adaptive(
                    query_vec=q,
                    index=index,
                    k_final=k,
                    fetch_fn=fetch_obras_by_ids,
                    sources=selected_sources,
                    max_rounds=8,
                    start_fetch=start_fetch,
                    max_mysql_ids=None,
                )

            if df_meta.empty:
                st.warning("No encontré filas en MySQL para esos IDs con el filtro de source.")
            else:
                st.success(
                    f"Resultados: {len(df_meta)} | rondas={stats['rounds']} | k_fetch_final={stats['k_fetch']} | ids_unicos={stats['unique']}"
                )
                render_cards(df_meta, cols_n=cols_n, show_n=show_n)

                with st.expander("Ver tabla completa"):
                    st.dataframe(df_meta, use_container_width=True)
