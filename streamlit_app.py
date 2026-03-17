"""Streamlit UI — QTO Pipeline.

Uses a background thread so Streamlit reruns cannot interrupt the pipeline.
Results are persisted to /tmp so they survive websocket disconnects.
"""
import json, logging, os, sys, time, traceback, threading
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
for p in ["/opt/homebrew/bin", "/usr/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Bridge secrets into os.environ (Streamlit Cloud or Cloud Run secrets)
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

RESULTS_FILE = "/tmp/pipeline_results.json"


# ── Results persistence ──────────────────────────────────────────────────────

def _save_results(state):
    """Save pipeline results to disk so they survive websocket drops."""
    try:
        data = {
            "step1_time": state.get("step1_time", 0),
            "step2a_time": state.get("step2a_time", 0),
            "step2d_time": state.get("step2d_time", 0),
            "step3_time": state.get("step3_time", 0),
            "n_sheets": state.get("n_sheets", 0),
            "n_tables": state.get("n_tables", 0),
            "n_drawing_items": state.get("n_drawing_items", 0),
            "items": [
                {
                    "trade": i.trade,
                    "item_description": i.item_description,
                    "qty": i.qty,
                    "unit": i.unit,
                    "schedule_mark": i.schedule_mark or "",
                    "confidence": i.confidence,
                }
                for i in state.get("final_items", [])
            ],
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(data, f)
        logger.info(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def _load_results():
    """Load previously saved results if they exist."""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


# ── Background runner ────────────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_pipeline_state = {
    "status": "idle",
    "run_id": None,
}


def _run_pipeline_thread(run_id, file_data):
    """Execute the full pipeline in a background thread."""
    state = _pipeline_state
    try:
        # Step 1
        state["step_label"] = "Step 1: Classifying documents..."
        t0 = time.time()
        classification = classify_documents(file_data)
        state["step1_time"] = time.time() - t0
        state["step1_done"] = True
        logger.info(f"Step 1 done in {state['step1_time']:.0f}s")

        # Step 2a
        state["step_label"] = "Step 2a: Sheet index + table extraction..."
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
        state["step_label"] = "Step 2d: Reading drawings (Gemini Pro)..."
        t0 = time.time()
        drawing_items = read_drawings(classification, sheets, tables)
        state["step2d_time"] = time.time() - t0
        state["step2d_done"] = True
        state["n_drawing_items"] = len(drawing_items)
        logger.info(f"Step 2d done in {state['step2d_time']:.0f}s")

        # Step 3
        state["step_label"] = "Step 3: Vision quantification..."
        t0 = time.time()
        final_items = quantify_items(drawing_items, classification, sheets)
        state["step3_time"] = time.time() - t0
        state["step3_done"] = True
        logger.info(f"Step 3 done in {state['step3_time']:.0f}s")

        state["final_items"] = final_items
        state["status"] = "complete"

        # Save to disk so results survive websocket drops
        _save_results(state)

    except Exception as e:
        state["status"] = "error"
        state["error"] = str(e)
        state["traceback"] = traceback.format_exc()
        logger.error(f"Pipeline failed: {e}")
    finally:
        _pipeline_lock.release()


# ── Session state ────────────────────────────────────────────────────────────
if "run_id" not in st.session_state:
    st.session_state.run_id = None


def on_run():
    files = st.session_state.get("uploader")
    if not files:
        return
    if not _pipeline_lock.acquire(blocking=False):
        return

    file_data = [(f.name, f.read()) for f in files]
    run_id = f"run_{time.time()}"

    _pipeline_state.update({
        "status": "running",
        "run_id": run_id,
        "start_time": time.time(),
        "step_label": "Starting...",
        "step1_done": False, "step2a_done": False,
        "step2d_done": False, "step3_done": False,
        "error": None, "traceback": None, "final_items": None,
    })

    # Remove old results file
    if os.path.exists(RESULTS_FILE):
        os.unlink(RESULTS_FILE)

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(run_id, file_data),
        daemon=True,
    )
    thread.start()
    st.session_state.run_id = run_id


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, key="uploader")
    is_running = _pipeline_state["status"] == "running"
    st.button(
        "Running..." if is_running else "Run Pipeline",
        type="primary",
        on_click=on_run,
        disabled=is_running or not st.session_state.get("uploader"),
    )


# ── Helper to display results ────────────────────────────────────────────────

def _show_results_from_saved(data):
    """Display results from saved JSON data."""
    st.success(f"Step 1: Classification ({data['step1_time']:.0f}s)")
    st.success(f"Step 2a: {data['n_sheets']} sheets, {data['n_tables']} tables ({data['step2a_time']:.0f}s)")
    st.success(f"Step 2d: {data['n_drawing_items']} items ({data['step2d_time']:.0f}s)")
    st.success(f"Step 3: Vision done ({data['step3_time']:.0f}s)")
    total = data["step1_time"] + data["step2a_time"] + data["step2d_time"] + data["step3_time"]
    st.success(f"Pipeline complete! Total: {total:.0f}s")

    items = data["items"]
    st.divider()
    n_total = len(items)
    n_qty = sum(1 for i in items if i.get("qty") is not None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Items", n_total)
    c2.metric("With Quantity", n_qty)
    c3.metric("Manual Review", n_total - n_qty)

    st.dataframe(pd.DataFrame([
        {"Trade": i["trade"], "Description": i["item_description"],
         "Qty": i["qty"] if i["qty"] is not None else "", "Unit": i["unit"],
         "Mark": i.get("schedule_mark", ""),
         "Confidence": f"{i['confidence']:.0%}" if i.get("confidence") else ""}
        for i in items
    ]), use_container_width=True, hide_index=True, height=600)


# ── Pipeline progress / results ──────────────────────────────────────────────
status = _pipeline_state["status"]

if status == "error":
    st.error(f"Pipeline failed: {_pipeline_state['error']}")
    st.code(_pipeline_state["traceback"])
    if st.button("Reset"):
        _pipeline_state["status"] = "idle"
        st.rerun()

elif status == "complete":
    # Show from memory (thread just finished)
    _show_results_from_saved({
        "step1_time": _pipeline_state["step1_time"],
        "step2a_time": _pipeline_state["step2a_time"],
        "step2d_time": _pipeline_state["step2d_time"],
        "step3_time": _pipeline_state["step3_time"],
        "n_sheets": _pipeline_state["n_sheets"],
        "n_tables": _pipeline_state["n_tables"],
        "n_drawing_items": _pipeline_state["n_drawing_items"],
        "items": [
            {"trade": i.trade, "item_description": i.item_description,
             "qty": i.qty, "unit": i.unit,
             "schedule_mark": i.schedule_mark or "",
             "confidence": i.confidence}
            for i in _pipeline_state["final_items"]
        ],
    })

elif status == "running":
    done = sum([
        _pipeline_state.get("step1_done", False),
        _pipeline_state.get("step2a_done", False),
        _pipeline_state.get("step2d_done", False),
        _pipeline_state.get("step3_done", False),
    ])
    elapsed = int(time.time() - _pipeline_state.get("start_time", time.time()))

    st.warning(f"Pipeline running... ({elapsed}s elapsed)")
    st.progress(done / 4, text=_pipeline_state.get("step_label", "Starting..."))

    if _pipeline_state.get("step1_done"):
        st.success(f"Step 1: Classification ({_pipeline_state['step1_time']:.0f}s)")
    if _pipeline_state.get("step2a_done"):
        st.success(f"Step 2a: {_pipeline_state['n_sheets']} sheets, {_pipeline_state['n_tables']} tables ({_pipeline_state['step2a_time']:.0f}s)")
    if _pipeline_state.get("step2d_done"):
        st.success(f"Step 2d: {_pipeline_state['n_drawing_items']} items ({_pipeline_state['step2d_time']:.0f}s)")

    time.sleep(3)
    st.rerun()

else:
    # Idle — check if there are saved results from a previous run
    saved = _load_results()
    if saved and saved.get("items"):
        st.info("Showing results from last pipeline run:")
        _show_results_from_saved(saved)
