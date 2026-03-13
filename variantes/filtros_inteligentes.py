# app.py — CLIP+FAISS (texto/imagen) + Metadatos (MySQL) + Filtros (chips) + ✅ Mapa 2D (PCA) + Detalle
# ✅ filtros por cualquier campo (detecta tipo automáticamente)
# ✅ UI: seleccionar campo -> operador -> input adecuado -> "Añadir filtro" -> chips removibles
# ✅ Texto→Imágenes / Imagen→Imágenes / Metadatos / ✅ Mapa 2D (PCA) / README
# Requiere Python 3.10+
#
# pip install streamlit plotly faiss-cpu mysql-connector-python numpy pandas pillow torch transformers

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import faiss
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
import mysql.connector
from PIL import Image
import plotly.express as px

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

.chips { display:flex; flex-wrap:wrap; gap:8px; margin: 6px 0 12px 0;}
.chip {
  display:inline-flex; align-items:center; gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.05);
  font-size: 0.88rem;
}

.kv { margin: 6px 0; }
.kv b { opacity: 0.85; }

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
# Schema (columnas y tipos)
# -------------------------
@dataclass
class ColInfo:
    name: str
    data_type: str
    is_nullable: str

def norm_type(dt_: str) -> str:
    dt_ = (dt_ or "").lower()
    if dt_ in {"int", "bigint", "smallint", "mediumint", "tinyint", "decimal", "numeric", "float", "double"}:
        return "number"
    if dt_ in {"date", "datetime", "timestamp"}:
        return "date"
    if dt_ in {"bit", "bool", "boolean"}:
        return "bool"
    return "text"

@st.cache_data(ttl=900)
def get_obras_schema() -> list[ColInfo]:
    conn = mysql_conn()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = %s AND TABLE_NAME = 'obras_arte'
            ORDER BY ORDINAL_POSITION
            """,
            (MYSQL_DATABASE,),
        )
        rows = cur.fetchall() or []
        return [ColInfo(r[0], r[1], r[2]) for r in rows]
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        conn.close()

def schema_map(schema: list[ColInfo]) -> dict[str, ColInfo]:
    return {c.name: c for c in schema}

EXCLUDE_FIELDS_CONTAINS = ("embedding", "vector", "blob", "bytes")
EXCLUDE_EXACT = {"face_embedding"}
DEFAULT_TEXT_SEARCH_FIELDS = ("title", "artist", "description", "source_url")

def ui_fields(schema: list[ColInfo]) -> list[str]:
    out = []
    for c in schema:
        n = c.name.lower()
        if c.name in EXCLUDE_EXACT:
            continue
        if any(x in n for x in EXCLUDE_FIELDS_CONTAINS):
            continue
        out.append(c.name)
    return out

# -------------------------
# Sources list
# -------------------------
@st.cache_data(ttl=300)
def fetch_sources() -> list[str]:
    conn = mysql_conn()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source FROM obras_arte ORDER BY source")
        rows = cur.fetchall() or []
        return [r[0] for r in rows if r and r[0]]
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        conn.close()

# -------------------------
# Distinct values helper
# -------------------------
@st.cache_data(ttl=600)
def fetch_distinct_values(field: str, limit: int = 200) -> list[str]:
    schema = schema_map(get_obras_schema())
    if field not in schema:
        return []
    sql = f"SELECT DISTINCT `{field}` AS v FROM obras_arte WHERE `{field}` IS NOT NULL ORDER BY v LIMIT %s"
    conn = mysql_conn()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(sql, (int(limit),))
        rows = cur.fetchall() or []
        vals = []
        for r in rows:
            v = r[0]
            if v is None:
                continue
            s = str(v).strip()
            if s:
                vals.append(s)
        return vals
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        conn.close()

# -------------------------
# Build WHERE from "chips"
# -------------------------
OPS_BY_TYPE = {
    "text": ["contiene", "igual", "empieza", "termina", "vacío", "no vacío", "en lista"],
    "number": ["=", "!=", ">", ">=", "<", "<=", "entre", "vacío", "no vacío", "en lista"],
    "date": ["=", "!=", ">", ">=", "<", "<=", "entre", "vacío", "no vacío"],
    "bool": ["es", "vacío", "no vacío"],
}

def build_where_from_chips(filters: list[dict], schema: list[ColInfo]) -> tuple[str, list[Any]]:
    sm = schema_map(schema)
    clauses: list[str] = []
    params: list[Any] = []

    for f in filters or []:
        field = str(f.get("field") or "").strip()
        op = str(f.get("op") or "").strip()
        v = f.get("value")

        if not field or field not in sm:
            continue

        t = f.get("type") or norm_type(sm[field].data_type)

        if op == "vacío":
            clauses.append(f"`{field}` IS NULL OR `{field}` = ''")
            continue
        if op == "no vacío":
            clauses.append(f"`{field}` IS NOT NULL AND `{field}` <> ''")
            continue

        if t == "text":
            s = str(v or "")
            if op == "contiene":
                clauses.append(f"`{field}` LIKE %s")
                params.append(f"%{s}%")
            elif op == "igual":
                clauses.append(f"`{field}` = %s")
                params.append(s)
            elif op == "empieza":
                clauses.append(f"`{field}` LIKE %s")
                params.append(f"{s}%")
            elif op == "termina":
                clauses.append(f"`{field}` LIKE %s")
                params.append(f"%{s}")
            elif op == "en lista":
                vals = [x.strip() for x in str(v or "").split(",") if x.strip()]
                if not vals:
                    continue
                placeholders = ",".join(["%s"] * len(vals))
                clauses.append(f"`{field}` IN ({placeholders})")
                params.extend(vals)

        elif t == "number":
            if op == "entre":
                a, b = v if isinstance(v, (list, tuple)) and len(v) == 2 else (None, None)
                if a is None or b is None:
                    continue
                clauses.append(f"`{field}` BETWEEN %s AND %s")
                params.extend([a, b])
            elif op == "en lista":
                vals = [x.strip() for x in str(v or "").split(",") if x.strip()]
                if not vals:
                    continue
                placeholders = ",".join(["%s"] * len(vals))
                clauses.append(f"`{field}` IN ({placeholders})")
                params.extend(vals)
            else:
                clauses.append(f"`{field}` {op} %s")
                params.append(v)

        elif t == "date":
            if op == "entre":
                a, b = v if isinstance(v, (list, tuple)) and len(v) == 2 else (None, None)
                if not a or not b:
                    continue
                clauses.append(f"`{field}` BETWEEN %s AND %s")
                params.extend([a, b])
            else:
                clauses.append(f"`{field}` {op} %s")
                params.append(v)

        elif t == "bool":
            if op == "es":
                clauses.append(f"`{field}` = %s")
                params.append(1 if bool(v) else 0)

    where = " AND ".join([f"({c})" for c in clauses]) if clauses else "1=1"
    return where, params

def chip_label(f: dict) -> str:
    field = f.get("field")
    op = f.get("op")
    val = f.get("value")
    if op in {"vacío", "no vacío"}:
        return f"{field} {op}"
    if op == "entre" and isinstance(val, (list, tuple)) and len(val) == 2:
        return f"{field} entre {val[0]} y {val[1]}"
    if op == "es":
        return f"{field} es {'Sí' if val else 'No'}"
    return f"{field} {op} {val}"

# -------------------------
# Fetch metadata by ids (CLIP results -> MySQL) + apply chips filters
# -------------------------
@st.cache_data(ttl=30)
def fetch_obras_by_ids_filtered(ids: list[int], chip_filters: list[dict]) -> pd.DataFrame:
    ids = [int(x) for x in ids if int(x) != -1]
    if not ids:
        return pd.DataFrame()

    schema = get_obras_schema()
    ids_placeholders = ",".join(["%s"] * len(ids))
    base_where = f"`id` IN ({ids_placeholders})"
    params: list[Any] = list(ids)

    w2, p2 = build_where_from_chips(chip_filters, schema)
    if w2 and w2 != "1=1":
        base_where = f"({base_where}) AND ({w2})"
        params.extend(p2)

    sql = f"SELECT * FROM obras_arte WHERE {base_where}"

    conn = mysql_conn()
    cur = None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        return pd.DataFrame(rows)
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        conn.close()

# -------------------------
# Metadatos search (sin CLIP)
# -------------------------
@st.cache_data(ttl=20)
def search_metadata_mysql(text_q: str | None, chip_filters: list[dict], limit: int = 300) -> pd.DataFrame:
    schema = get_obras_schema()
    cols = {c.name for c in schema}

    where1, params = build_where_from_chips(chip_filters, schema)

    if text_q:
        like = f"%{text_q}%"
        fields = [f for f in DEFAULT_TEXT_SEARCH_FIELDS if f in cols]
        if fields:
            or_clause = " OR ".join([f"`{f}` LIKE %s" for f in fields])
            where1 = f"({where1}) AND ({or_clause})" if where1 != "1=1" else f"({or_clause})"
            params.extend([like] * len(fields))

    select_fields = ["id"]
    for f in ("source", "title", "artist", "image_full_url", "image_url", "source_url", "description", "year", "fecha", "date", "created_at"):
        if f in cols:
            select_fields.append(f)

    sql = f"""
        SELECT {", ".join([f"`{x}`" for x in select_fields])}
        FROM obras_arte
        WHERE {where1}
        ORDER BY id DESC
        LIMIT %s
    """
    params.append(int(limit))

    conn = mysql_conn()
    cur = None
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        rows = cur.fetchall() or []
        return pd.DataFrame(rows)
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass
        conn.close()

# -------------------------
# Embeddings
# -------------------------
def text_embedding(query: str, device, model, processor) -> np.ndarray:
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.inference_mode():
        text_out = model.text_model(**inputs)
        pooled = text_out.pooler_output
        v = model.text_projection(pooled)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype("float32")

def image_embedding_from_pil(img: Image.Image, device, model, processor) -> np.ndarray:
    img = img.convert("RGB")
    img.thumbnail((1024, 1024))
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    proj = getattr(model, "visual_projection", None) or getattr(model, "vision_projection", None)
    with torch.inference_mode():
        if proj is not None:
            vision_out = model.vision_model(**inputs)
            v = proj(vision_out.pooler_output)
        else:
            v = model.get_image_features(**inputs)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.detach().cpu().numpy().astype("float32")

# -------------------------
# FAISS search (adaptativo) + MySQL filtro sin perder precisión
# -------------------------
def is_ip_metric(index) -> bool:
    try:
        return index.metric_type == faiss.METRIC_INNER_PRODUCT
    except Exception:
        return True

def build_best_score_map(pairs, index) -> dict[int, float]:
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

def search_with_filter_adaptive(
    *,
    query_vec: np.ndarray,
    index,
    k_final: int,
    chip_filters: list[dict],
    max_rounds: int = 8,
    start_fetch: int = 200,
    max_mysql_ids: int | None = None,
) -> tuple[pd.DataFrame, dict]:
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

        best_round = build_best_score_map(pairs, index)

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

        sorted_ids = sorted(
            best_scores_all.keys(),
            key=lambda rid: best_scores_all[rid],
            reverse=higher_is_better,
        )

        limit_mysql = max_mysql_ids if max_mysql_ids is not None else k_fetch
        ask_ids = sorted_ids[: min(len(sorted_ids), limit_mysql)]

        df = fetch_obras_by_ids_filtered(ask_ids, chip_filters)

        if not df.empty:
            df["score"] = df["id"].map(best_scores_all)
            df = df.dropna(subset=["score"]).sort_values("score", ascending=not higher_is_better)
            df_best = df.head(k_final).reset_index(drop=True)

        if len(df_best) >= k_final:
            break

        if k_fetch >= ntotal:
            break

        k_fetch = min(ntotal, k_fetch * 2)

    stats = {"rounds": rounds, "k_fetch": k_fetch, "unique": len(best_scores_all), "kept": len(df_best)}
    return df_best, stats

# -------------------------
# Render cards (grid)
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
# Render detalle (para el mapa)
# -------------------------
def render_detail(row: pd.Series):
    title = row.get("title") or "(sin título)"
    artist = row.get("artist") or ""
    source = row.get("source") or ""
    score = float(row.get("score") or 0.0)
    img_url = row.get("image_full_url") or row.get("image_url") or ""
    src_url = row.get("source_url") or ""
    desc = row.get("description") or ""
    year = row.get("year") or row.get("fecha") or row.get("date") or row.get("date_text") or ""

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if isinstance(img_url, str) and img_url.strip():
        st.image(img_url, use_container_width=True)
    else:
        st.info("Sin imagen")

    st.markdown("<hr class='sep'/>", unsafe_allow_html=True)

    st.markdown(f"### {title}")
    st.markdown(f"<div class='small'>ID: <b>{int(row['id'])}</b> · score: <b>{score:.4f}</b></div>", unsafe_allow_html=True)

    if artist:
        st.markdown(f"<div class='kv'><b>Artista:</b> {artist}</div>", unsafe_allow_html=True)
    if year:
        st.markdown(f"<div class='kv'><b>Fecha:</b> {year}</div>", unsafe_allow_html=True)
    if source:
        st.markdown(f"<div class='kv'><b>Fuente:</b> {source}</div>", unsafe_allow_html=True)

    if isinstance(src_url, str) and src_url.strip():
        st.link_button("Abrir ficha (source_url)", src_url)

    if isinstance(desc, str) and desc.strip():
        st.markdown("<hr class='sep'/>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>{desc[:900]}{'…' if len(desc)>900 else ''}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# ✅ Vectores de resultados para PCA (reconstruct)
# -------------------------
def reconstruct_vectors(index, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vecs = []
    ids_ok = []
    for _id in ids.tolist():
        _id = int(_id)
        if _id < 0:
            continue
        try:
            v = index.reconstruct(_id)  # funciona si el índice soporta reconstruct por ID
            vecs.append(v)
            ids_ok.append(_id)
        except Exception:
            continue

    if not vecs:
        return np.empty((0, int(index.d)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.vstack(vecs).astype(np.float32, copy=False), np.array(ids_ok, dtype=np.int64)

def pca_2d_faiss(vectors: np.ndarray) -> np.ndarray:
    vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    d = vectors.shape[1]
    pca = faiss.PCAMatrix(d, 2)
    pca.train(vectors)
    coords = pca.apply_py(vectors)
    return coords.astype(np.float32, copy=False)

# -------------------------
# App init
# -------------------------
st.title("Buscador")

index = load_faiss_index(INDEX_PATH)
device, model, processor = load_clip(MODEL_NAME)

st.caption(f"Index ntotal={index.ntotal} dim={index.d} | modelo={MODEL_NAME} | device={device}")

is_idmap = isinstance(index, faiss.IndexIDMap) or isinstance(index, faiss.IndexIDMap2)
if not is_idmap:
    st.warning("⚠️ Índice no IDMap: FAISS puede devolver posiciones, no obras_arte.id.")

schema = get_obras_schema()
schema_by = schema_map(schema)
fields_for_ui = ui_fields(schema)

# -------------------------
# Session state: chips + last results (para mapa 2D)
# -------------------------
if "chip_filters" not in st.session_state:
    st.session_state.chip_filters = []

if "last_results_df" not in st.session_state:
    st.session_state.last_results_df = pd.DataFrame()

if "selected_id" not in st.session_state:
    st.session_state.selected_id = None

def add_chip(f: dict):
    st.session_state.chip_filters.append(f)

def remove_chip(idx: int):
    st.session_state.chip_filters.pop(idx)

def clear_chips():
    st.session_state.chip_filters = []

# -------------------------
# Sidebar (intuitivo)
# -------------------------
with st.sidebar:
    st.subheader("Configuración")
    k = st.slider("Top K (final)", 5, 200, TOP_K_DEFAULT, 5)
    overfetch_mult = st.slider("Overfetch (para filtrar)", 1, 30, 10, 1)
    cols_n = st.selectbox("Columnas de cards", [2, 3], index=0)
    show_n = st.slider("Cuántos cards mostrar", 1, 60, min(20, k), 1)

    st.markdown("---")
    st.subheader("Filtros ")

    # 1) Source como filtro rápido (multiselect)
    if "source" in schema_by:
        all_sources = fetch_sources()
        sel_sources = st.multiselect("Colecciones (source)", options=all_sources, default=all_sources)
    else:
        sel_sources = None

    # 2) Agregar filtro: elegir campo
    field = st.selectbox(
        "Campo",
        options=fields_for_ui,
        index=(fields_for_ui.index("artist") if "artist" in fields_for_ui else 0),
    )
    col = schema_by[field]
    t = norm_type(col.data_type)

    ops = OPS_BY_TYPE.get(t, OPS_BY_TYPE["text"])
    op = st.selectbox("Condición", options=ops, index=0)

    value: Any = None
    if op in {"vacío", "no vacío"}:
        st.caption("Este filtro no requiere valor.")
    elif t == "bool":
        value = st.checkbox("Valor", value=True)
    elif t == "number":
        if op == "entre":
            a = st.number_input("Desde", value=0.0)
            b = st.number_input("Hasta", value=0.0)
            value = [a, b]
        elif op == "en lista":
            value = st.text_input("Valores separados por coma", value="1,2,3")
        else:
            value = st.number_input("Valor", value=0.0)
    elif t == "date":
        if op == "entre":
            d1 = st.date_input("Desde", value=dt.date(1800, 1, 1))
            d2 = st.date_input("Hasta", value=dt.date.today())
            value = [str(d1), str(d2)]
        else:
            d = st.date_input("Fecha", value=dt.date.today())
            value = str(d)
    else:
        if op == "en lista":
            suggest = st.checkbox("Mostrar sugerencias", value=(field in {"source", "artist"}))
            if suggest:
                vals = fetch_distinct_values(field, limit=200)
                picked = st.multiselect("Selecciona valores", options=vals)
                value = ",".join(picked) if picked else ""
            else:
                value = st.text_input("Valores separados por coma", value="")
        else:
            value = st.text_input("Valor", value="")

    colA, colB = st.columns(2)
    with colA:
        if st.button("➕ Añadir filtro", use_container_width=True):
            add_chip({"field": field, "type": t, "op": op, "value": value})
    with colB:
        if st.button("🧹 Limpiar", use_container_width=True):
            clear_chips()

    st.markdown("**Filtros activos:**")
    if not st.session_state.chip_filters and not (sel_sources and "source" in schema_by):
        st.caption("Ninguno (mostrando todo).")
    else:
        for idx, f in enumerate(st.session_state.chip_filters):
            label = chip_label(f)
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"<div class='chip'>{label}</div>", unsafe_allow_html=True)
            with c2:
                if st.button("❌", key=f"rm_{idx}"):
                    remove_chip(idx)
                    st.rerun()

    effective_filters = list(st.session_state.chip_filters)
    if sel_sources is not None and len(sel_sources) > 0 and "source" in schema_by:
        effective_filters.append({"field": "source", "type": "text", "op": "en lista", "value": ",".join(sel_sources)})

    st.markdown("---")
    st.subheader("Mapa 2D")
    map_color_by_source = st.checkbox("Color por source", value=True)
    if st.button("Limpiar selección del mapa"):
        st.session_state.selected_id = None

# -------------------------
# Tabs
# -------------------------
tab_text, tab_img, tab_meta, tab_map, tab_readme = st.tabs(
    ["Texto → Imágenes", "Imagen → Imágenes", "Metadatos", "Mapa 2D", "README"]
)

# -------------------------
# TAB 1: Texto -> Imágenes
# -------------------------
with tab_text:
    with st.form("search_form_text"):
        query = st.text_input("Buscar por texto (CLIP)", value=DEFAULT_QUERY)
        submitted = st.form_submit_button("Buscar")

    if submitted:
        query = (query or "").strip()
        if not query:
            st.info("Escribe algo para buscar.")
        else:
            qvec = text_embedding(query, device, model, processor)
            start_fetch = max(200, k * int(overfetch_mult))

            with st.spinner("Buscando (FAISS) y aplicando filtros..."):
                df_meta, stats = search_with_filter_adaptive(
                    query_vec=qvec,
                    index=index,
                    k_final=k,
                    chip_filters=effective_filters,
                    max_rounds=8,
                    start_fetch=start_fetch,
                    max_mysql_ids=None,
                )

            if df_meta.empty:
                st.warning("No encontré resultados con esos filtros.")
                st.session_state.last_results_df = pd.DataFrame()
            else:
                st.success(
                    f"Resultados: {len(df_meta)} | rondas={stats['rounds']} | k_fetch_final={stats['k_fetch']} | ids_unicos={stats['unique']}"
                )
                st.session_state.last_results_df = df_meta.copy()
                render_cards(df_meta, cols_n=cols_n, show_n=show_n)
                with st.expander("Ver tabla completa"):
                    st.dataframe(df_meta, use_container_width=True)

# -------------------------
# TAB 2: Imagen -> Imágenes
# -------------------------
with tab_img:
    st.write("Sube una imagen (JPG/PNG/WebP)")
    uploaded = st.file_uploader("Subir imagen", type=["png", "jpg", "jpeg", "webp"])

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
            qvec = image_embedding_from_pil(img_query, device, model, processor)
            start_fetch = max(200, k * int(overfetch_mult))

            with st.spinner("Buscando por imagen (FAISS) y aplicando filtros..."):
                df_meta, stats = search_with_filter_adaptive(
                    query_vec=qvec,
                    index=index,
                    k_final=k,
                    chip_filters=effective_filters,
                    max_rounds=8,
                    start_fetch=start_fetch,
                    max_mysql_ids=None,
                )

            if df_meta.empty:
                st.warning("No encontré resultados con esos filtros.")
                st.session_state.last_results_df = pd.DataFrame()
            else:
                st.success(
                    f"Resultados: {len(df_meta)} | rondas={stats['rounds']} | k_fetch_final={stats['k_fetch']} | ids_unicos={stats['unique']}"
                )
                st.session_state.last_results_df = df_meta.copy()
                render_cards(df_meta, cols_n=cols_n, show_n=show_n)
                with st.expander("Ver tabla completa"):
                    st.dataframe(df_meta, use_container_width=True)

# -------------------------
# TAB 3: Metadatos -> MySQL
# -------------------------
with tab_meta:
    st.write("Buscador clásico por metadatos (sin CLIP/FAISS). Usa filtros del sidebar y texto libre.")

    with st.form("meta_form"):
        text_q = st.text_input("Texto libre (título / artista / descripción)", value="")
        limit = st.slider("Máximo resultados", 10, 5000, 300, 10)
        submitted_meta = st.form_submit_button("Buscar")

    if submitted_meta:
        tq = (text_q or "").strip() or None
        with st.spinner("Buscando en MySQL..."):
            df_m = search_metadata_mysql(tq, effective_filters, limit=int(limit))

        if df_m.empty:
            st.warning("No hay resultados con esos filtros.")
            st.session_state.last_results_df = pd.DataFrame()
        else:
            st.success(f"Resultados: {len(df_m)}")
            df_m = df_m.copy()
            df_m["score"] = 0.0
            st.session_state.last_results_df = df_m.copy()
            render_cards(df_m, cols_n=cols_n, show_n=min(show_n, len(df_m)))
            with st.expander("Ver tabla completa"):
                st.dataframe(df_m, use_container_width=True)

# -------------------------
# ✅ TAB 4: Mapa 2D (PCA) + Detalle (misma fila)
# -------------------------
with tab_map:
    st.header("Mapa 2D (PCA) + Detalle")
    df_last: pd.DataFrame = st.session_state.get("last_results_df", pd.DataFrame())

    if df_last is None or df_last.empty or "id" not in df_last.columns:
        st.info("Primero haz una búsqueda (Texto/Imagen/Metadatos). Aquí se mapearán los resultados actuales.")
        st.stop()

    # ids y vectors
    ids = df_last["id"].astype(int).to_numpy(np.int64)

    with st.spinner("Reconstruyendo vectores desde FAISS (para PCA)..."):
        vecs, ids_ok = reconstruct_vectors(index, ids)

    if vecs is None or vecs.shape[0] < 3:
        st.warning(
            "No pude obtener suficientes vectores para PCA. "
            "Esto suele pasar si tu índice no soporta reconstruct por ID (o si el índice no está alineado con obras_arte.id)."
        )
        st.stop()

    df_map = df_last[df_last["id"].isin(ids_ok)].copy()
    df_map = df_map.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # Alinear vecs al orden de df_map
    id_to_pos = {int(i): p for p, i in enumerate(ids_ok.tolist())}
    order = [id_to_pos[int(i)] for i in df_map["id"].astype(int).tolist() if int(i) in id_to_pos]
    vecs_aligned = vecs[np.array(order, dtype=np.int64)]
    if vecs_aligned.shape[0] < 3:
        st.warning("Muy pocos puntos válidos para PCA.")
        st.stop()

    with st.spinner("Calculando PCA 2D..."):
        coords = pca_2d_faiss(vecs_aligned)

    df_map["x"] = coords[:, 0]
    df_map["y"] = coords[:, 1]

    # columnas mostrables
    df_map["title_show"] = df_map.get("title", "").fillna("").astype(str)
    df_map["artist_show"] = df_map.get("artist", "").fillna("").astype(str)
    df_map["source_show"] = df_map.get("source", "").fillna("").astype(str)

    hover_cols = {
        "id": True,
        "score": True,
        "title_show": True,
        "artist_show": True,
        "source_show": True,
    }

    if map_color_by_source and "source_show" in df_map.columns:
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
        legend_title_text="source" if map_color_by_source else None,
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
            st.session_state.selected_id = selected_id

    with col_det:
        st.subheader("Detalle")
        if st.button("Limpiar selección", key="clear_sel_inline"):
            st.session_state.selected_id = None

        sid = st.session_state.get("selected_id")
        if not sid:
            st.info("Selecciona un punto del mapa para ver el detalle aquí.")
        else:
            hit = df_last[df_last["id"].astype(int) == int(sid)]
            if hit.empty:
                st.warning("No encontré ese ID en los resultados actuales.")
            else:
                render_detail(hit.iloc[0])

# -------------------------
# TAB 5: README
# -------------------------
with tab_readme:
    st.header("El Archivo Visual como Espacio Navegable")
    st.markdown(
        """



En lugar de depender únicamente de los metadatos, el sistema permite buscar imágenes a partir de su proximidad visual dentro de un espacio computacional entrenado (Visual-semántico). Las obras se representan mediante vectores matemáticos que describen su contenido visual. Cuando dos imágenes comparten rasgos similares, ya sea en su composición, temática o estructura, sus representaciones aparecen cercanas dentro de ese espacio.

Este enfoque permite analizar: 

- relaciones iconográficas  
- recurrencias formales  
- patrones compositivos  
- proximidades estilísticas  
- afinidades semánticas (por lenguaje natura)


# Tecnologías utilizadas

### CLIP
CLIP (Contrastive Language–Image Pretraining) es un modelo de aprendizaje profundo que permite representar imágenes y textos dentro de un mismo espacio vectorial. Gracias a este modelo, es posible traducir tanto descripciones escritas como imágenes en vectores numéricos que conservan información semántica.

### FAISS
FAISS es una biblioteca desarrollada por Facebook AI Research para realizar búsquedas de similitud en grandes conjuntos de vectores.


# Modos de exploración

### Texto → Imagen
Introduce una descripción y recupera imágenes visualmente relacionadas.

### Imagen → Imagen
Sube una imagen y encuentra otras con similitud visual.

### Búsqueda por metadatos
Consultas tradicionales por artista, colección o descripción.

### Mapa 2D (PCA)
Proyecta los resultados a 2D para navegar “por cercanía”, y permite seleccionar un punto para ver el detalle al lado.





Iván Barajas Hurtado  
Artista plástico e historiador del arte  
Bogotá, Colombia
"""
    )