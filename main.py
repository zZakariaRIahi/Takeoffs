"""FastAPI backend — QTO Pipeline."""
import json, logging, os, sys, time, traceback, threading
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(__file__))
for p in ["/opt/homebrew/bin", "/usr/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app.agents.document_classifier_agent import classify_documents
from app.extractors.sheet_indexer import build_sheet_index
from app.extractors.table_extractor import extract_tables_from_sheets, tables_to_schedule_rows
from app.extractors.drawing_reader import read_drawings
from app.extractors.vision_quantifier import quantify_items

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_FILE = "/tmp/pipeline_results.json"

app = FastAPI(title="QTO Pipeline")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Pipeline state ───────────────────────────────────────────────────────────

_lock = threading.Lock()
_state = {
    "status": "idle",  # idle | running | complete | error
    "step_label": "",
    "start_time": 0,
    "step1_done": False, "step2a_done": False,
    "step2d_done": False, "step3_done": False,
    "step1_time": 0, "step2a_time": 0, "step2d_time": 0, "step3_time": 0,
    "n_sheets": 0, "n_tables": 0, "n_drawing_items": 0,
    "error": None, "traceback": None,
    "items": None,
}


def _save_results():
    """Persist results to disk."""
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump({
                "step1_time": _state["step1_time"],
                "step2a_time": _state["step2a_time"],
                "step2d_time": _state["step2d_time"],
                "step3_time": _state["step3_time"],
                "n_sheets": _state["n_sheets"],
                "n_tables": _state["n_tables"],
                "n_drawing_items": _state["n_drawing_items"],
                "items": _state["items"],
            }, f)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def _load_saved():
    """Load results from disk if they exist."""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _run_pipeline(file_data: List[Tuple[str, bytes]]):
    """Run the full 4-step pipeline in a background thread."""
    try:
        # Step 1
        _state["step_label"] = "Step 1: Classifying documents..."
        t0 = time.time()
        classification = classify_documents(file_data)
        _state["step1_time"] = time.time() - t0
        _state["step1_done"] = True
        logger.info(f"Step 1 done in {_state['step1_time']:.0f}s")

        # Step 2a
        _state["step_label"] = "Step 2a: Sheet index + table extraction..."
        t0 = time.time()
        sheets = build_sheet_index(classification)
        tables = extract_tables_from_sheets(sheets, classification)
        rows = tables_to_schedule_rows(tables)
        _state["step2a_time"] = time.time() - t0
        _state["step2a_done"] = True
        _state["n_sheets"] = len(sheets)
        _state["n_tables"] = len(tables)
        logger.info(f"Step 2a done in {_state['step2a_time']:.0f}s")

        # Step 2d
        _state["step_label"] = "Step 2d: Reading drawings (Gemini Pro)..."
        t0 = time.time()
        drawing_items = read_drawings(classification, sheets, tables)
        _state["step2d_time"] = time.time() - t0
        _state["step2d_done"] = True
        _state["n_drawing_items"] = len(drawing_items)
        logger.info(f"Step 2d done in {_state['step2d_time']:.0f}s")

        # Step 3
        _state["step_label"] = "Step 3: Vision quantification..."
        t0 = time.time()
        final_items = quantify_items(drawing_items, classification, sheets)
        _state["step3_time"] = time.time() - t0
        _state["step3_done"] = True
        logger.info(f"Step 3 done in {_state['step3_time']:.0f}s")

        # Serialize items
        _state["items"] = [
            {
                "trade": i.trade,
                "description": i.item_description,
                "qty": i.qty,
                "unit": i.unit,
                "mark": i.schedule_mark or "",
                "confidence": round(i.confidence, 2) if i.confidence else 0,
            }
            for i in final_items
        ]
        _state["status"] = "complete"
        _save_results()
        logger.info("Pipeline complete — results saved")

    except Exception as e:
        _state["status"] = "error"
        _state["error"] = str(e)
        _state["traceback"] = traceback.format_exc()
        logger.error(f"Pipeline failed: {e}")
    finally:
        _lock.release()


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not _lock.acquire(blocking=False):
        return JSONResponse({"error": "Pipeline already running"}, status_code=409)

    file_data = [(f.filename, await f.read()) for f in files]

    # Reset state
    _state.update({
        "status": "running",
        "start_time": time.time(),
        "step_label": "Starting...",
        "step1_done": False, "step2a_done": False,
        "step2d_done": False, "step3_done": False,
        "step1_time": 0, "step2a_time": 0, "step2d_time": 0, "step3_time": 0,
        "n_sheets": 0, "n_tables": 0, "n_drawing_items": 0,
        "error": None, "traceback": None, "items": None,
    })

    if os.path.exists(RESULTS_FILE):
        os.unlink(RESULTS_FILE)

    thread = threading.Thread(target=_run_pipeline, args=(file_data,), daemon=True)
    thread.start()

    return {"status": "started"}


@app.get("/status")
async def status():
    elapsed = int(time.time() - _state["start_time"]) if _state["start_time"] else 0
    resp = {
        "status": _state["status"],
        "elapsed": elapsed,
        "step_label": _state["step_label"],
        "step1_done": _state["step1_done"],
        "step2a_done": _state["step2a_done"],
        "step2d_done": _state["step2d_done"],
        "step3_done": _state["step3_done"],
        "step1_time": _state["step1_time"],
        "step2a_time": _state["step2a_time"],
        "step2d_time": _state["step2d_time"],
        "step3_time": _state["step3_time"],
        "n_sheets": _state["n_sheets"],
        "n_tables": _state["n_tables"],
        "n_drawing_items": _state["n_drawing_items"],
    }
    if _state["status"] == "error":
        resp["error"] = _state["error"]
        resp["traceback"] = _state["traceback"]
    return resp


@app.get("/results")
async def results():
    # Try memory first, then disk
    items = _state.get("items")
    if items:
        return {
            "step1_time": _state["step1_time"],
            "step2a_time": _state["step2a_time"],
            "step2d_time": _state["step2d_time"],
            "step3_time": _state["step3_time"],
            "n_sheets": _state["n_sheets"],
            "n_tables": _state["n_tables"],
            "n_drawing_items": _state["n_drawing_items"],
            "items": items,
        }
    saved = _load_saved()
    if saved:
        return saved
    return JSONResponse({"error": "No results available"}, status_code=404)


@app.post("/reset")
async def reset():
    if _state["status"] == "running":
        return JSONResponse({"error": "Cannot reset while running"}, status_code=409)
    _state["status"] = "idle"
    _state["items"] = None
    return {"status": "idle"}
