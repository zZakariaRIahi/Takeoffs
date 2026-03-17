"""Streamlit UI — QTO Pipeline.

Uses background threads for long-running steps so Streamlit reruns
cannot interrupt them. Each step runs to completion in its own thread,
and the UI polls for results.
"""
import logging, os, sys, time, traceback, threading
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


# ── Background runner ────────────────────────────────────────────────────────
# Runs the entire pipeline in a thread that Streamlit reruns CANNOT kill.
# Results are stored in a module-level dict (survives reruns, shared across
# script executions within the same server process).

_pipeline_state = {}  # module-level, survives reruns


def _run_pipeline_thread(run_id, file_data):
    """Execute the full pipeline in a background thread."""
    state = _pipeline_state[run_id]
    try:
        # Step 1
        state["status"] = "Step 1: Classifying documents..."
        t0 = time.time()
        classification = classify_documents(file_data)
        state["step1_time"] = time.time() - t0
        state["step1_done"] = True
        logger.info(f"Step 1 done in {state['step1_time']:.0f}s")

        # Step 2a
        state["status"] = "Step 2a: Sheet index + table extraction..."
        t0 = time.time()
        sheets = build_sheet_index(classification)
        tables = extract_tables_from_sheets(sheets, classification)
        rows = tables_to_schedule_rows(tables)
        state["step2a_time"] = time.time() - t0
        state["step2a_done"] = True
        state["n_sheets"] = len(sheets)
        state["n_tables"] = len(tables)
        logger.info(f"Step 2a done in {state['step2a_time']:.0f}s")

        # Step 2d
        state["status"] = "Step 2d: Reading drawings (Gemini Pro)..."
        t0 = time.time()
        drawing_items = read_drawings(classification, sheets, tables)
        state["step2d_time"] = time.time() - t0
        state["step2d_done"] = True
        state["n_drawing_items"] = len(drawing_items)
        logger.info(f"Step 2d done in {state['step2d_time']:.0f}s")

        # Step 3
        state["status"] = "Step 3: Vision quantification..."
        t0 = time.time()
        final_items = quantify_items(drawing_items, classification, sheets)
        state["step3_time"] = time.time() - t0
        state["step3_done"] = True
        logger.info(f"Step 3 done in {state['step3_time']:.0f}s")

        state["final_items"] = final_items
        state["status"] = "complete"

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        state["traceback"] = traceback.format_exc()
        logger.error(f"Pipeline failed: {e}")


# ── Session state ────────────────────────────────────────────────────────────
if "run_id" not in st.session_state:
    st.session_state.run_id = None
    st.session_state.file_data = None


def on_run():
    files = st.session_state.get("uploader")
    if files:
        file_data = [(f.name, f.read()) for f in files]
        run_id = f"run_{id(file_data)}_{time.time()}"

        # Initialize state and start background thread
        _pipeline_state[run_id] = {
            "status": "starting",
            "step1_done": False, "step2a_done": False,
            "step2d_done": False, "step3_done": False,
        }
        thread = threading.Thread(
            target=_run_pipeline_thread,
            args=(run_id, file_data),
            daemon=True,
        )
        thread.start()

        st.session_state.run_id = run_id
        st.session_state.file_data = file_data


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="uploader")
    st.button("Run Pipeline", type="primary", on_click=on_run,
              disabled=not st.session_state.get("uploader"))

# ── Pipeline progress / results ──────────────────────────────────────────────
run_id = st.session_state.run_id
if run_id and run_id in _pipeline_state:
    state = _pipeline_state[run_id]
    status = state["status"]

    if status == "error":
        st.error(f"Pipeline failed: {state['error']}")
        st.code(state["traceback"])
        st.session_state.run_id = None

    elif status == "complete":
        # Show completed step timings
        st.success(f"Step 1: Classification ({state['step1_time']:.0f}s)")
        st.success(f"Step 2a: {state['n_sheets']} sheets, {state['n_tables']} tables ({state['step2a_time']:.0f}s)")
        st.success(f"Step 2d: {state['n_drawing_items']} items ({state['step2d_time']:.0f}s)")
        st.success(f"Step 3: Vision done ({state['step3_time']:.0f}s)")
        total = state["step1_time"] + state["step2a_time"] + state["step2d_time"] + state["step3_time"]
        st.success(f"Pipeline complete! Total: {total:.0f}s")

        # Results
        final_items = state["final_items"]
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

        # Clean up
        st.session_state.run_id = None

    else:
        # Still running — show progress
        if state.get("step1_done"):
            st.success(f"Step 1: Classification ({state['step1_time']:.0f}s)")
        if state.get("step2a_done"):
            st.success(f"Step 2a: {state['n_sheets']} sheets, {state['n_tables']} tables ({state['step2a_time']:.0f}s)")
        if state.get("step2d_done"):
            st.success(f"Step 2d: {state['n_drawing_items']} items ({state['step2d_time']:.0f}s)")

        # Show current step
        st.info(f"Running: {status}")

        # Auto-refresh every 3 seconds to poll for updates
        time.sleep(3)
        st.rerun()
