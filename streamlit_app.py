"""Streamlit UI — QTO Pipeline."""
import logging, os, sys, time, traceback
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
for p in ["/opt/homebrew/bin", "/usr/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# On Streamlit Cloud, secrets come from st.secrets, not .env
# Bridge them into os.environ so downstream code can find them
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

# ── Session state ────────────────────────────────────────────────────────────
for _k, _v in [
    ("started", False),
    ("file_data", None),
    ("step1_result", None),
    ("step2a_result", None),
    ("step2d_result", None),
    ("step3_result", None),
    ("pipeline_error", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


def on_run():
    """Callback — runs BEFORE the page reruns (Streamlit guarantee)."""
    files = st.session_state.get("uploader")
    if files:
        st.session_state.started = True
        st.session_state.file_data = [(f.name, f.read()) for f in files]
        # Clear previous results
        st.session_state.step1_result = None
        st.session_state.step2a_result = None
        st.session_state.step2d_result = None
        st.session_state.step3_result = None
        st.session_state.pipeline_error = None


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="uploader")
    st.button("Run Pipeline", type="primary", on_click=on_run,
              disabled=not st.session_state.get("uploader"))

# ── Pipeline ─────────────────────────────────────────────────────────────────
if st.session_state.started and st.session_state.file_data:
    file_data = st.session_state.file_data
    status = st.status("Running pipeline...", expanded=True)

    try:
        # Step 1 — use cached result if available (survives reruns)
        if st.session_state.step1_result is not None:
            classification = st.session_state.step1_result
            status.write("Step 1: Classification (cached)")
        else:
            status.write("Step 1: Classifying documents...")
            t0 = time.time()
            classification = classify_documents(file_data)
            t1 = time.time() - t0
            st.session_state.step1_result = classification
            status.write(f"Step 1: Classification done ({t1:.0f}s)")

        # Step 2a
        if st.session_state.step2a_result is not None:
            sheets, tables, rows = st.session_state.step2a_result
            status.write("Step 2a: Sheet index + tables (cached)")
        else:
            status.write("Step 2a: Sheet index + table extraction...")
            t0 = time.time()
            sheets = build_sheet_index(classification)
            status.write(f"  Built {len(sheets)} sheets, extracting tables...")
            tables = extract_tables_from_sheets(sheets, classification)
            rows = tables_to_schedule_rows(tables)
            t2a = time.time() - t0
            st.session_state.step2a_result = (sheets, tables, rows)
            status.write(f"Step 2a: {len(sheets)} sheets, {len(tables)} tables ({t2a:.0f}s)")

        # Step 2d
        if st.session_state.step2d_result is not None:
            drawing_items = st.session_state.step2d_result
            status.write("Step 2d: Drawing reading (cached)")
        else:
            status.write("Step 2d: Reading drawings (Gemini Pro)...")
            t0 = time.time()
            drawing_items = read_drawings(classification, sheets, tables)
            t2d = time.time() - t0
            st.session_state.step2d_result = drawing_items
            status.write(f"Step 2d: {len(drawing_items)} items ({t2d:.0f}s)")

        # Step 3
        if st.session_state.step3_result is not None:
            final_items = st.session_state.step3_result
            status.write("Step 3: Vision quantification (cached)")
        else:
            status.write("Step 3: Vision quantification...")
            t0 = time.time()
            final_items = quantify_items(drawing_items, classification, sheets)
            t3 = time.time() - t0
            st.session_state.step3_result = final_items
            status.write(f"Step 3: Vision done ({t3:.0f}s)")

        status.update(label="Pipeline complete!", state="complete", expanded=False)
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
        st.session_state.pipeline_error = traceback.format_exc()
        status.update(label="Pipeline failed!", state="error")
        st.error(f"Pipeline failed: {e}")
        st.code(st.session_state.pipeline_error)

# Show error from previous run if any
elif st.session_state.pipeline_error:
    st.error("Last pipeline run failed:")
    st.code(st.session_state.pipeline_error)
