"""Streamlit UI — QTO Pipeline."""
import hashlib, logging, os, sys, time, traceback
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
for p in ["/opt/homebrew/bin", "/usr/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# On Streamlit Cloud, secrets come from st.secrets, not .env
for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY"):
    if not os.environ.get(key):
        try:
            os.environ[key] = st.secrets[key]
        except (KeyError, FileNotFoundError):
            pass

from app.agents.document_classifier_agent import classify_documents
from app.extractors.sheet_indexer import build_sheet_index
from app.extractors.table_extractor import extract_tables_from_sheets, tables_to_schedule_rows
from app.extractors.drawing_reader import read_drawings
from app.extractors.vision_quantifier import quantify_items

st.set_page_config(page_title="QTO Pipeline", layout="wide")
st.title("Construction QTO Pipeline")
logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _file_data_hash(file_data):
    """Create a stable hash key from uploaded files (name + size)."""
    h = hashlib.md5()
    for name, data in sorted(file_data, key=lambda x: x[0]):
        h.update(name.encode())
        h.update(str(len(data)).encode())
    return h.hexdigest()


# ── Cached pipeline steps ────────────────────────────────────────────────────
# @st.cache_resource survives Streamlit reruns without re-executing.
# The _file_data arg (underscore prefix) is NOT hashed — we use file_hash instead.

@st.cache_resource(show_spinner="Step 1: Classifying documents...")
def run_step1(file_hash, _file_data):
    logger.info(f"Step 1 START (hash={file_hash})")
    result = classify_documents(_file_data)
    logger.info("Step 1 DONE")
    return result


@st.cache_resource(show_spinner="Step 2a: Sheet index + table extraction...")
def run_step2a(file_hash, _classification):
    logger.info("Step 2a START")
    sheets = build_sheet_index(_classification)
    tables = extract_tables_from_sheets(sheets, _classification)
    rows = tables_to_schedule_rows(tables)
    logger.info(f"Step 2a DONE: {len(sheets)} sheets, {len(tables)} tables")
    return sheets, tables, rows


@st.cache_resource(show_spinner="Step 2d: Reading drawings (Gemini Pro)...")
def run_step2d(file_hash, _classification, _sheets, _tables):
    logger.info("Step 2d START")
    items = read_drawings(_classification, _sheets, _tables)
    logger.info(f"Step 2d DONE: {len(items)} items")
    return items


@st.cache_resource(show_spinner="Step 3: Vision quantification...")
def run_step3(file_hash, _drawing_items, _classification, _sheets):
    logger.info("Step 3 START")
    items = quantify_items(_drawing_items, _classification, _sheets)
    logger.info(f"Step 3 DONE: {len(items)} items")
    return items


# ── Session state ────────────────────────────────────────────────────────────
if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.file_data = None


def on_run():
    files = st.session_state.get("uploader")
    if files:
        st.session_state.started = True
        st.session_state.file_data = [(f.name, f.read()) for f in files]


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="uploader")
    st.button("Run Pipeline", type="primary", on_click=on_run,
              disabled=not st.session_state.get("uploader"))

# ── Pipeline ─────────────────────────────────────────────────────────────────
if st.session_state.started and st.session_state.file_data:
    file_data = st.session_state.file_data
    fhash = _file_data_hash(file_data)

    try:
        t_total = time.time()

        # Step 1
        t0 = time.time()
        classification = run_step1(fhash, file_data)
        t1 = time.time() - t0
        st.success(f"Step 1: Classification ({t1:.0f}s)")

        # Step 2a
        t0 = time.time()
        sheets, tables, rows = run_step2a(fhash, classification)
        t2a = time.time() - t0
        st.success(f"Step 2a: {len(sheets)} sheets, {len(tables)} tables ({t2a:.0f}s)")

        # Step 2d
        t0 = time.time()
        drawing_items = run_step2d(fhash, classification, sheets, tables)
        t2d = time.time() - t0
        st.success(f"Step 2d: {len(drawing_items)} items ({t2d:.0f}s)")

        # Step 3
        t0 = time.time()
        final_items = run_step3(fhash, drawing_items, classification, sheets)
        t3 = time.time() - t0
        st.success(f"Step 3: Vision done ({t3:.0f}s)")

        total = time.time() - t_total
        st.success(f"Pipeline complete! Total: {total:.0f}s")
        st.session_state.started = False

        # ── Results ──────────────────────────────────────────────────────────
        st.divider()
        n_total = len(final_items)
        n_qty = sum(1 for i in final_items if i.qty is not None)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Items", n_total)
        c2.metric("With Quantity", n_qty)
        c3.metric("Manual Review", n_total - n_qty)

        st.dataframe(pd.DataFrame([
            {"Trade": i.trade, "Description": i.item_description,
             "Qty": i.qty if i.qty is not None else "", "Unit": i.unit,
             "Mark": i.schedule_mark or "",
             "Confidence": f"{i.confidence:.0%}" if i.confidence else ""}
            for i in final_items
        ]), use_container_width=True, hide_index=True, height=600)

    except Exception as e:
        st.session_state.started = False
        st.error(f"Pipeline failed: {e}")
        st.code(traceback.format_exc())
