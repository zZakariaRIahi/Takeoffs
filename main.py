"""FastAPI backend — QTO Pipeline with Job Queue."""
import json, logging, os, sys, time, traceback, threading, shutil, uuid, io, queue
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage as gcs
import google.auth
from google.auth.transport import requests as auth_requests
import requests as http_requests
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
for p in ["/opt/homebrew/bin", "/usr/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from app.agents.document_classifier_agent import classify_documents
from app.extractors.context_extractor import extract_context
from app.extractors.trade_extractor import extract_by_trade
from app.extractors.specs_extractor import (
    extract_from_specs_only,
    extract_from_specs_with_drawings,
    merge_specs_into_packages,
)
from app.core.document_classification import DocumentCategory

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logger = logging.getLogger(__name__)

GCS_BUCKET = "takeoffs-uploads-670952019485"
RESULTS_DIR = "/tmp/pipeline_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="QTO Pipeline")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Job system ───────────────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}  # job_id → job state
_job_queue: queue.Queue = queue.Queue()
_worker_running = False
MAX_CONCURRENT = 3  # max pipelines running simultaneously
_running_count = 0
_running_lock = threading.Lock()


def _new_job_state(job_id: str, session_id: str, filenames: List[str]) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "session_id": session_id,
        "filenames": filenames,
        "status": "queued",  # queued, running, complete, error
        "step_label": "Waiting in queue...",
        "start_time": 0,
        "queue_time": time.time(),
        "step1_done": False, "step2a_done": False,
        "step2d_done": False, "step3_done": False,
        "step1_time": 0, "step2a_time": 0, "step2d_time": 0, "step3_time": 0,
        "n_sheets": 0, "n_tables": 0, "n_drawing_items": 0,
        "error": None, "traceback": None, "items": None,
        "queue_position": 0,
    }


GCS_RESULTS_PREFIX = "results/"  # results stored in same GCS bucket under results/


def _save_job_results(job_id: str, job: Dict):
    """Persist job results to GCS for durability across deploys."""
    result_data = {
        "job_id": job_id,
        "filenames": job.get("filenames", []),
        "step1_time": job["step1_time"],
        "step2a_time": job["step2a_time"],
        "step2d_time": job["step2d_time"],
        "step3_time": job["step3_time"],
        "n_sheets": job["n_sheets"],
        "n_tables": job["n_tables"],
        "n_drawing_items": job["n_drawing_items"],
        "items": job["items"],
        "completed_at": time.time(),
    }
    # Save to GCS
    try:
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_RESULTS_PREFIX}{job_id}.json")
        blob.upload_from_string(json.dumps(result_data), content_type="application/json")
        logger.info(f"Saved job {job_id} results to GCS")
    except Exception as e:
        logger.error(f"Failed to save job results to GCS: {e}")
    # Also save locally as cache
    try:
        path = os.path.join(RESULTS_DIR, f"{job_id}.json")
        with open(path, "w") as f:
            json.dump(result_data, f)
    except Exception:
        pass


def _load_job_results(job_id: str):
    """Load job results — check local cache first, then GCS."""
    # Local cache
    try:
        path = os.path.join(RESULTS_DIR, f"{job_id}.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    # GCS
    try:
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(f"{GCS_RESULTS_PREFIX}{job_id}.json")
        if blob.exists():
            data = json.loads(blob.download_as_text())
            # Cache locally
            try:
                path = os.path.join(RESULTS_DIR, f"{job_id}.json")
                with open(path, "w") as f:
                    json.dump(data, f)
            except Exception:
                pass
            return data
    except Exception:
        pass
    return None


def _list_all_jobs_from_gcs():
    """List all completed jobs from GCS."""
    jobs = []
    try:
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=GCS_RESULTS_PREFIX)
        for blob in blobs:
            if blob.name.endswith(".json"):
                job_id = blob.name.replace(GCS_RESULTS_PREFIX, "").replace(".json", "")
                jobs.append({
                    "job_id": job_id,
                    "created": blob.time_created.isoformat() if blob.time_created else "",
                    "size": blob.size,
                })
    except Exception as e:
        logger.error(f"Failed to list jobs from GCS: {e}")
    return jobs


def _cleanup_gcs(session_id: str):
    """Delete uploaded files from GCS after pipeline completes."""
    try:
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blobs = list(bucket.list_blobs(prefix=f"{session_id}/"))
        for blob in blobs:
            blob.delete()
        logger.info(f"Cleaned up {len(blobs)} files from GCS session {session_id}")
    except Exception as e:
        logger.error(f"GCS cleanup failed: {e}")


# ── File parsing ─────────────────────────────────────────────────────────────

def _parse_tabular_files(file_data: List[Tuple[str, bytes]]):
    """Parse CSV/XLSX/DOCX files into pre-extracted tables."""
    from app.core.estimate_models import ExtractedTable

    pdf_files = []
    extra_tables = []

    for fname, content in file_data:
        ext = os.path.splitext(fname)[1].lower()

        if ext == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(content))
                if not df.empty:
                    extra_tables.append(ExtractedTable(
                        page_number=-1, filter_type="uploaded_file",
                        table_title=os.path.splitext(fname)[0],
                        schedule_type="uploaded",
                        headers=list(df.columns),
                        rows=[dict(row) for _, row in df.iterrows()],
                    ))
                    logger.info(f"Parsed CSV {fname}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to parse CSV {fname}: {e}")

        elif ext in (".xlsx", ".xls"):
            try:
                xls = pd.ExcelFile(io.BytesIO(content))
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name)
                    if not df.empty:
                        title = f"{os.path.splitext(fname)[0]} — {sheet_name}" if len(xls.sheet_names) > 1 else os.path.splitext(fname)[0]
                        extra_tables.append(ExtractedTable(
                            page_number=-1, filter_type="uploaded_file",
                            table_title=title, schedule_type="uploaded",
                            headers=list(df.columns),
                            rows=[{k: (str(v) if pd.notna(v) else "") for k, v in row.items()} for _, row in df.iterrows()],
                        ))
                        logger.info(f"Parsed Excel {fname}/{sheet_name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Failed to parse Excel {fname}: {e}")

        elif ext in (".docx", ".doc"):
            try:
                from docx import Document
                doc = Document(io.BytesIO(content))
                for i, tbl in enumerate(doc.tables):
                    rows_data = []
                    headers = [cell.text.strip() for cell in tbl.rows[0].cells]
                    for row in tbl.rows[1:]:
                        row_dict = {}
                        for j, cell in enumerate(row.cells):
                            key = headers[j] if j < len(headers) else f"Col{j}"
                            row_dict[key] = cell.text.strip()
                        rows_data.append(row_dict)
                    if headers and rows_data:
                        extra_tables.append(ExtractedTable(
                            page_number=-1, filter_type="uploaded_file",
                            table_title=f"{os.path.splitext(fname)[0]} — Table {i+1}",
                            schedule_type="uploaded", headers=headers, rows=rows_data,
                        ))
                        logger.info(f"Parsed DOCX {fname}/Table {i+1}: {len(rows_data)} rows")
            except Exception as e:
                logger.error(f"Failed to parse DOCX {fname}: {e}")

        elif ext == ".pdf":
            pdf_files.append((fname, content))
        else:
            logger.warning(f"Unsupported file type: {fname}")

    return pdf_files, extra_tables


# ── Pipeline runner ──────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, file_data: List[Tuple[str, bytes]]):
    """Run the full pipeline for one job."""
    job = _jobs[job_id]
    job["status"] = "running"
    job["start_time"] = time.time()
    job["step_label"] = "Starting..."

    try:
        # Separate PDFs from tabular files
        pdf_files, extra_tables = _parse_tabular_files(file_data)
        if extra_tables:
            logger.info(f"[{job_id}] Pre-parsed {len(extra_tables)} tables from uploaded files")

        # Step 1: Classify
        job["step_label"] = "Step 1: Classifying documents..."
        t0 = time.time()
        classification = classify_documents(pdf_files) if pdf_files else type('C', (), {'files': [], 'raw_pdf_bytes': {}})()
        job["step1_time"] = time.time() - t0
        job["step1_done"] = True
        logger.info(f"[{job_id}] Step 1 done in {job['step1_time']:.0f}s")

        # Detect what we have
        has_drawings = any(
            DocumentCategory.CONSTRUCTION_DRAWINGS in (cf.categories or set())
            for cf in getattr(classification, 'files', [])
        )
        has_specs = any(
            DocumentCategory.PROJECT_SPECIFICATIONS in (cf.categories or set())
            for cf in getattr(classification, 'files', [])
        )
        logger.info(f"[{job_id}] Has drawings: {has_drawings}, Has specs: {has_specs}")

        # ── CASE 1: Specs only (no drawings) ──
        if has_specs and not has_drawings:
            job["step_label"] = "Step 2: Extracting takeoff from specifications..."
            t0 = time.time()
            final_items = extract_from_specs_only(classification)
            job["step2a_time"] = time.time() - t0
            job["step2a_done"] = True
            job["step2d_done"] = True
            job["step3_done"] = True
            job["n_drawing_items"] = len(final_items)
            logger.info(f"[{job_id}] Specs-only: {len(final_items)} items in {job['step2a_time']:.0f}s")

        # ── CASE 2/3: Has drawings → try Step 2, then validate ──
        else:
            # Step 2: Extract from drawings
            job["step_label"] = "Step 2: Extracting schedules, keynotes & context..."
            t0 = time.time()
            if pdf_files:
                sheets, tables, rows, packages = extract_context(classification)
            else:
                sheets, tables, rows, packages = [], [], [], {}
            tables = tables + extra_tables
            job["step2a_time"] = time.time() - t0
            job["step2a_done"] = True
            job["step2d_done"] = True
            job["n_sheets"] = len(sheets)
            job["n_tables"] = len(tables)
            logger.info(f"[{job_id}] Step 2 done in {job['step2a_time']:.0f}s")

            # Validate: did Step 2 find a real drawing set?
            if len(sheets) < 3:
                logger.warning(
                    f"[{job_id}] Drawing set invalid — only {len(sheets)} sheets found "
                    f"(need >= 3 for a real drawing set)"
                )
                if has_specs:
                    # Fall back to specs extraction
                    logger.info(f"[{job_id}] Falling back to specs extraction")
                    job["step_label"] = "Step 2: Falling back to specs extraction..."
                    t0 = time.time()
                    final_items = extract_from_specs_only(classification)
                    job["step2a_time"] += time.time() - t0
                    job["step3_done"] = True
                    job["n_drawing_items"] = len(final_items)
                    logger.info(f"[{job_id}] Specs fallback: {len(final_items)} items")
                else:
                    # No specs either — use whatever Step 2 found
                    logger.info(f"[{job_id}] No specs available — proceeding with limited drawings")
                    job["step_label"] = "Step 3: Reading plans & counting quantities..."
                    t0 = time.time()
                    if pdf_files and packages:
                        final_items = extract_by_trade(classification, sheets, tables, packages)
                    else:
                        final_items = []
                    job["step3_time"] = time.time() - t0
                    job["step3_done"] = True
                    job["n_drawing_items"] = len(final_items)
                    logger.info(f"[{job_id}] Step 3 done in {job['step3_time']:.0f}s")
            else:
                # Real drawing set — proceed to Step 3
                job["step_label"] = "Step 3: Reading plans & counting quantities..."
                t0 = time.time()
                if pdf_files and packages:
                    final_items = extract_by_trade(classification, sheets, tables, packages)
                else:
                    final_items = []
                job["step3_time"] = time.time() - t0
                job["step3_done"] = True
                job["n_drawing_items"] = len(final_items)
                logger.info(f"[{job_id}] Step 3 done in {job['step3_time']:.0f}s")

        # Serialize
        job["items"] = [
            {
                "trade": i.trade,
                "description": i.item_description,
                "qty": i.qty,
                "unit": i.unit,
                "mark": i.schedule_mark or "",
                "confidence": round(i.confidence, 2) if i.confidence else 0,
                "source": i.source or "",
                "method": i.extraction_method or "",
                "review": i.review_reason or "",
            }
            for i in final_items
        ]
        job["status"] = "complete"
        _save_job_results(job_id, job)
        logger.info(f"[{job_id}] Pipeline complete — {len(final_items)} items saved")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        logger.error(f"[{job_id}] Pipeline failed: {e}")
    finally:
        # Clean up GCS
        session_id = job.get("session_id")
        if session_id:
            _cleanup_gcs(session_id)


# ── Queue worker ─────────────────────────────────────────────────────────────

def _run_job_thread(job_id: str, file_data: List[Tuple[str, bytes]]):
    """Run a single job in its own thread, then pick up next queued job."""
    global _running_count
    try:
        _run_pipeline(job_id, file_data)
    finally:
        with _running_lock:
            _running_count -= 1
        # Check if there's a queued job to pick up
        _dispatch_next()


def _dispatch_next():
    """Start the next queued job if we have capacity."""
    global _running_count
    with _running_lock:
        if _running_count >= MAX_CONCURRENT:
            return
        try:
            job_id, file_data = _job_queue.get_nowait()
        except queue.Empty:
            return
        _running_count += 1

    logger.info(f"Dispatching job {job_id} ({_running_count}/{MAX_CONCURRENT} running)")
    thread = threading.Thread(target=_run_job_thread, args=(job_id, file_data), daemon=True)
    thread.start()
    _update_queue_positions()


def _update_queue_positions():
    """Update queue position for all queued jobs."""
    pos = 1
    for jid, job in _jobs.items():
        if job["status"] == "queued":
            job["queue_position"] = pos
            pos += 1


def _start_job_or_queue(job_id: str, file_data: List[Tuple[str, bytes]]):
    """Start job immediately if capacity available, otherwise queue it."""
    global _running_count
    with _running_lock:
        if _running_count < MAX_CONCURRENT:
            _running_count += 1
            logger.info(f"Starting job {job_id} immediately ({_running_count}/{MAX_CONCURRENT} running)")
            thread = threading.Thread(target=_run_job_thread, args=(job_id, file_data), daemon=True)
            thread.start()
            return
    # Queue it
    _job_queue.put((job_id, file_data))
    _update_queue_positions()
    logger.info(f"Job {job_id} queued (queue size: {_job_queue.qsize()})")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()


@app.post("/get-upload-urls")
async def get_upload_urls(filenames: List[str]):
    """Generate signed URLs for direct browser → GCS uploads."""
    session_id = uuid.uuid4().hex[:12]

    credentials, project = google.auth.default()
    credentials.refresh(auth_requests.Request())

    storage_client = gcs.Client(credentials=credentials)
    bucket = storage_client.bucket(GCS_BUCKET)

    CONTENT_TYPES = {
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls": "application/vnd.ms-excel",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }

    urls = []
    for fname in filenames:
        ext = os.path.splitext(fname)[1].lower()
        content_type = CONTENT_TYPES.get(ext, "application/octet-stream")
        blob = bucket.blob(f"{session_id}/{fname}")
        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=30),
            method="PUT",
            content_type=content_type,
            service_account_email=credentials.service_account_email,
            access_token=credentials.token,
        )
        urls.append({"filename": fname, "url": signed_url, "content_type": content_type})

    logger.info(f"Generated {len(urls)} upload URLs for session {session_id}")
    return {"session_id": session_id, "urls": urls}


@app.post("/start/{session_id}")
async def start(session_id: str):
    """Queue a pipeline job for files uploaded to GCS."""
    # Download files from GCS
    try:
        client = gcs.Client()
        bucket = client.bucket(GCS_BUCKET)
        blobs = list(bucket.list_blobs(prefix=f"{session_id}/"))

        if not blobs:
            return JSONResponse({"error": "No files found in upload session"}, status_code=400)

        file_data = []
        filenames = []
        for blob in blobs:
            fname = blob.name.split("/", 1)[1]
            content = blob.download_as_bytes()
            file_data.append((fname, content))
            filenames.append(fname)
            logger.info(f"Downloaded {fname} from GCS ({len(content)/1024/1024:.1f} MB)")

    except Exception as e:
        logger.error(f"Failed to read files from GCS: {e}")
        return JSONResponse({"error": f"Failed to read uploaded files: {e}"}, status_code=500)

    # Create job
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = _new_job_state(job_id, session_id, filenames)

    # Start immediately or queue
    _start_job_or_queue(job_id, file_data)

    status = _jobs[job_id]["status"]
    queue_pos = _jobs[job_id].get("queue_position", 0)
    return {"status": status, "job_id": job_id, "queue_position": queue_pos}


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        # Not in memory — check GCS for a saved result
        saved = _load_job_results(job_id)
        if saved:
            return {
                "job_id": job_id,
                "status": "complete",
                "elapsed": 0,
                "queue_elapsed": 0,
                "queue_position": 0,
                "step_label": "Complete",
                "step1_done": True, "step2a_done": True,
                "step2d_done": True, "step3_done": True,
                "step1_time": saved.get("step1_time", 0),
                "step2a_time": saved.get("step2a_time", 0),
                "step2d_time": saved.get("step2d_time", 0),
                "step3_time": saved.get("step3_time", 0),
                "n_sheets": saved.get("n_sheets", 0),
                "n_tables": saved.get("n_tables", 0),
                "n_drawing_items": saved.get("n_drawing_items", 0),
                "filenames": saved.get("filenames", []),
            }
        return JSONResponse({"error": "Job not found"}, status_code=404)

    elapsed = int(time.time() - job["start_time"]) if job["start_time"] else 0
    queue_elapsed = int(time.time() - job["queue_time"]) if job["queue_time"] else 0

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "elapsed": elapsed,
        "queue_elapsed": queue_elapsed,
        "queue_position": job.get("queue_position", 0),
        "step_label": job["step_label"],
        "step1_done": job["step1_done"],
        "step2a_done": job["step2a_done"],
        "step2d_done": job["step2d_done"],
        "step3_done": job["step3_done"],
        "step1_time": job["step1_time"],
        "step2a_time": job["step2a_time"],
        "step2d_time": job["step2d_time"],
        "step3_time": job["step3_time"],
        "n_sheets": job["n_sheets"],
        "n_tables": job["n_tables"],
        "n_drawing_items": job["n_drawing_items"],
        "filenames": job.get("filenames", []),
    }
    if job["status"] == "error":
        resp["error"] = job["error"]
        resp["traceback"] = job["traceback"]
    return resp


# Legacy /status endpoint — returns the most recent job
@app.get("/status")
async def status_legacy():
    if not _jobs:
        return {
            "status": "idle", "elapsed": 0, "step_label": "",
            "step1_done": False, "step2a_done": False,
            "step2d_done": False, "step3_done": False,
            "step1_time": 0, "step2a_time": 0, "step2d_time": 0, "step3_time": 0,
            "n_sheets": 0, "n_tables": 0, "n_drawing_items": 0,
        }
    # Find most recent job
    latest = max(_jobs.values(), key=lambda j: j["queue_time"])
    return await job_status(latest["job_id"])


@app.get("/results/{job_id}")
async def job_results(job_id: str):
    # Check in-memory first
    job = _jobs.get(job_id)
    if job and job.get("items"):
        return {
            "job_id": job_id,
            "filenames": job.get("filenames", []),
            "step1_time": job["step1_time"],
            "step2a_time": job["step2a_time"],
            "step2d_time": job["step2d_time"],
            "step3_time": job["step3_time"],
            "n_sheets": job["n_sheets"],
            "n_tables": job["n_tables"],
            "n_drawing_items": job["n_drawing_items"],
            "items": job["items"],
        }
    # Check disk
    saved = _load_job_results(job_id)
    if saved:
        return saved
    return JSONResponse({"error": "No results available"}, status_code=404)


# Legacy /results — returns most recent completed job
@app.get("/results")
async def results_legacy():
    # Find most recent completed job
    completed = [j for j in _jobs.values() if j["status"] == "complete" and j.get("items")]
    if completed:
        latest = max(completed, key=lambda j: j["queue_time"])
        return await job_results(latest["job_id"])
    # Check disk for any saved results
    for fname in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(RESULTS_DIR, fname)) as f:
                    return json.load(f)
            except Exception:
                continue
    return JSONResponse({"error": "No results available"}, status_code=404)


@app.get("/jobs")
async def list_jobs():
    """List all jobs — in-memory active + GCS history."""
    jobs_list = []
    seen_ids = set()

    # In-memory jobs (active/recent)
    for jid, job in sorted(_jobs.items(), key=lambda x: x[1]["queue_time"], reverse=True):
        elapsed = int(time.time() - job["start_time"]) if job["start_time"] else 0
        jobs_list.append({
            "job_id": jid,
            "status": job["status"],
            "filenames": job.get("filenames", []),
            "elapsed": elapsed,
            "queue_position": job.get("queue_position", 0),
            "n_items": len(job["items"]) if job.get("items") else 0,
            "queued_at": job.get("queue_time", 0),
        })
        seen_ids.add(jid)

    # GCS history (completed jobs from previous deploys)
    for gcs_job in _list_all_jobs_from_gcs():
        jid = gcs_job["job_id"]
        if jid not in seen_ids:
            jobs_list.append({
                "job_id": jid,
                "status": "complete",
                "filenames": [],
                "elapsed": 0,
                "queue_position": 0,
                "n_items": 0,
                "created": gcs_job.get("created", ""),
            })

    return {"jobs": jobs_list, "queue_size": _job_queue.qsize()}


# ── Pricing ──────────────────────────────────────────────────────────────────

PRICING_API = "https://cost-rag-pricing-670952019485.us-central1.run.app/price"

_pricing_state = {
    "status": "idle",
    "start_time": 0,
    "total_items": 0,
    "result": None,
    "error": None,
}


def _run_pricing(items: list, project_id: str):
    """Run pricing in background thread."""
    try:
        pricing_items = [
            {
                "trade": it.get("trade", ""),
                "description": it.get("description", ""),
                "qty": it.get("qty") or 1,
                "unit": it.get("unit", ""),
                "mark": it.get("mark", ""),
                "source": it.get("source", ""),
            }
            for it in items
        ]

        logger.info(f"Sending {len(pricing_items)} items to pricing agent")

        resp = http_requests.post(
            PRICING_API,
            json={"project_id": project_id, "items": pricing_items},
            timeout=1200,
        )
        if not resp.ok:
            logger.error(f"Pricing agent returned {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        pricing_result = resp.json()
        _pricing_state["result"] = pricing_result
        _pricing_state["status"] = "complete"
        logger.info(f"Pricing complete: {pricing_result.get('summary', {})}")

    except Exception as e:
        _pricing_state["status"] = "error"
        _pricing_state["error"] = str(e)
        logger.error(f"Pricing failed: {e}")


@app.post("/submit-pricing")
async def submit_pricing(payload: dict):
    """Start pricing in background thread."""
    items = payload.get("items", [])
    project_id = payload.get("project_id", uuid.uuid4().hex[:12])

    if not items:
        return JSONResponse({"error": "No items provided"}, status_code=400)

    if _pricing_state["status"] == "running":
        return JSONResponse({"error": "Pricing already running"}, status_code=409)

    _pricing_state.update({
        "status": "running",
        "start_time": time.time(),
        "total_items": len(items),
        "result": None,
        "error": None,
    })

    thread = threading.Thread(target=_run_pricing, args=(items, project_id), daemon=True)
    thread.start()

    return {"status": "started", "total_items": len(items)}


@app.get("/pricing-status")
async def pricing_status():
    elapsed = int(time.time() - _pricing_state["start_time"]) if _pricing_state["start_time"] else 0
    resp = {
        "status": _pricing_state["status"],
        "elapsed": elapsed,
        "total_items": _pricing_state["total_items"],
    }
    if _pricing_state["status"] == "complete":
        resp["result"] = _pricing_state["result"]
    elif _pricing_state["status"] == "error":
        resp["error"] = _pricing_state["error"]
    return resp


# ── Reset ────────────────────────────────────────────────────────────────────

@app.post("/force-reset")
async def force_reset():
    """Force reset — clears current job state."""
    # Reset pricing
    _pricing_state.update({"status": "idle", "start_time": 0, "total_items": 0, "result": None, "error": None})
    logger.info("Force reset")
    return {"status": "idle"}
