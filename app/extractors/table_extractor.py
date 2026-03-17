"""Step 2a — Sheet Index + Table Extraction.

3-phase approach:
  Phase 1: img2table (no OCR, <1s/page) detects table bboxes at 150 DPI
  Phase 2: Send crops at 150 DPI to Gemini Flash (5 per call, parallel)
           to filter out false positives (plans, notes, title blocks)
  Phase 3: Re-render confirmed tables at 300 DPI, send to Gemini Flash
           for accurate content extraction

No Tesseract needed. img2table uses line detection only.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
from img2table.document import Image as Img2TableImage
from google import genai
from google.genai import types as genai_types

from app.core.document_classification import (
    ClassifiedFile,
    DocumentCategory,
    DocumentClassificationResult,
    PageInfo,
)
from app.core.estimate_models import (
    EstimateItem,
    ExtractedScheduleRow,
    ExtractedTable,
    SheetInfo,
)

logger = logging.getLogger(__name__)

DETECT_DPI = 150
CROP_DPI = 300
FILTER_BATCH = 5
EXTRACT_BATCH = 5
WORKERS = 4

# Minimum crop size to skip noise
MIN_CROP_W = 80
MIN_CROP_H = 40


# ════════════════════════════════════════════════════════════════════════════════
# Client
# ════════════════════════════════════════════════════════════════════════════════

def _get_genai_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        try:
            from app.config.settings import settings
            api_key = settings.GOOGLE_API_KEY
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════════
# Phase 1: img2table bbox detection (no OCR)
# ════════════════════════════════════════════════════════════════════════════════

def _detect_bboxes_on_page(
    pdf_bytes: bytes, local_page_idx: int, global_page: int,
) -> List[dict]:
    """Detect table bounding boxes using img2table line detection (no OCR)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = DETECT_DPI / 72.0
    pix = doc[local_page_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    page_w, page_h = pix.width, pix.height
    page_area = page_w * page_h

    tmp_path = os.path.join(tempfile.gettempdir(), f"det_{global_page}_{id(doc)}.png")
    with open(tmp_path, "wb") as f:
        f.write(pix.tobytes("png"))
    doc.close()

    try:
        img_doc = Img2TableImage(src=tmp_path)
        detected = img_doc.extract_tables(ocr=None, borderless_tables=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    results = []
    for table in detected:
        bbox = table.bbox
        w = bbox.x2 - bbox.x1
        h = bbox.y2 - bbox.y1

        # Skip tiny (noise)
        if w < MIN_CROP_W or h < MIN_CROP_H:
            continue
        # Skip full-page (drawing border)
        if w * h > page_area * 0.90:
            continue
        # Skip title block (small region bottom-right)
        if (w < 600 and h < 300
                and bbox.x2 > page_w * 0.85
                and bbox.y2 > page_h * 0.85):
            continue

        results.append({
            "global_page": global_page,
            "bbox": (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            "page_w": page_w,
            "page_h": page_h,
        })

    return results


# ════════════════════════════════════════════════════════════════════════════════
# Phase 2: Gemini Flash filtering (150 DPI crops)
# ════════════════════════════════════════════════════════════════════════════════

FILTER_PROMPT = """\
You are filtering cropped regions from construction drawing pages.

For EACH crop image below, classify it as ONE of:
- "sheet_index" — a drawing index / sheet list
- "table" — a schedule or data table useful for construction estimation
  (door schedule, finish schedule, panel schedule, fixture schedule, etc.)
- "not_table" — a plan view, section, detail, note block, title block,
  legend, or anything that is NOT a structured data table

IMPORTANT: Do NOT discard any real table or schedule. When in doubt, classify as "table".
Only mark as "not_table" if it clearly is NOT a data table.

Return JSON array:
[
  {"crop_id": 0, "type": "table"},
  {"crop_id": 1, "type": "not_table"},
  {"crop_id": 2, "type": "sheet_index"}
]

Return ONLY valid JSON, no markdown fences."""


def _crop_from_page(pdf_bytes: bytes, local_page_idx: int, bbox: tuple, dpi: int) -> bytes:
    """Render a page at given DPI and crop a bbox region."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    pix = doc[local_page_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()

    # Scale bbox from DETECT_DPI to target DPI
    scale = dpi / DETECT_DPI
    pad = 15
    x1 = max(0, int(bbox[0] * scale) - pad)
    y1 = max(0, int(bbox[1] * scale) - pad)
    x2 = min(img.width, int(bbox[2] * scale) + pad)
    y2 = min(img.height, int(bbox[3] * scale) + pad)

    cropped = img.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _filter_batch(client, batch: List[Tuple[int, bytes]]) -> List[Tuple[int, str]]:
    """Send a batch of crop images to Gemini Flash for filtering.

    Args:
        batch: list of (crop_index, jpeg_bytes)
    Returns:
        list of (crop_index, type) where type is "table"|"sheet_index"|"not_table"
    """
    contents = [FILTER_PROMPT]
    for crop_idx, img_bytes in batch:
        contents.append(f"--- crop_id={crop_idx} ---")
        contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=2048),
    )

    raw = (response.text or "[]").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        results = json.loads(raw)
        if not isinstance(results, list):
            results = [results]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                results = json.loads(match.group())
            except json.JSONDecodeError:
                return [(idx, "table") for idx, _ in batch]  # Keep all on parse failure
        else:
            return [(idx, "table") for idx, _ in batch]

    out = []
    for r in results:
        if isinstance(r, dict):
            out.append((r.get("crop_id", -1), r.get("type", "table")))
    return out


# ════════════════════════════════════════════════════════════════════════════════
# Phase 3: Gemini Flash extraction (300 DPI crops)
# ════════════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """\
You are reading construction schedules/tables from high-resolution crops.

Read EVERY cell exactly as written. Do NOT interpret, correct, or guess values.
Common construction abbreviations you MUST preserve exactly:
  HM = Hollow Metal, SC = Sealed Concrete, RB = Rubber Base,
  ACT = Acoustical Ceiling Tile, GWB = Gypsum Wall Board,
  OHD = Overhead Door, EXP = Exposed, FBO = Furnished By Owner,
  (P) = Existing, (E) = Existing, (N) = New, NIC = Not In Contract

For EACH table crop, return a JSON object:
{
  "table_title": "Door Schedule",
  "schedule_type": "door",
  "headers": ["MARK", "TYPE", "SIZE", "MATERIAL", "FRAME"],
  "rows": [
    {"MARK": "D1", "TYPE": "A", "SIZE": "3'-0\\" x 7'-0\\"", "MATERIAL": "HM", "FRAME": "HM"}
  ]
}

If multiple crops are shown, return a JSON array of table objects.

schedule_type: one of sheet_index, door, window, finish, fixture, equipment,
  panel, lighting, hardware, ventilation, mechanical, plumbing_fixture, other

Rules:
- Copy EXACTLY what is written — do NOT substitute or guess
- Extract EVERY row — do NOT skip any
- Empty cells = ""
- If you cannot read a cell clearly, use "?" — do NOT hallucinate
- Numbers as strings (preserve formatting)
- Merged cells: repeat value in each row
- If a crop has no readable table, return {"table_title": "", "rows": []}

Return ONLY valid JSON, no markdown fences."""


def _extract_batch(client, batch: List[Tuple[int, bytes]]) -> List[Tuple[int, dict]]:
    """Send table crops at 300 DPI to Gemini Flash for extraction."""
    contents = [EXTRACT_PROMPT]
    for crop_idx, img_bytes in batch:
        contents.append(f"--- Table crop {crop_idx} ---")
        contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=32768),
    )

    raw = (response.text or "{}").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                return []
        else:
            return []

    # Normalize to list
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    # Map each parsed table to its crop_idx
    results = []
    for i, item in enumerate(parsed):
        if isinstance(item, dict):
            idx = batch[min(i, len(batch) - 1)][0] if i < len(batch) else batch[-1][0]
            results.append((idx, item))
    return results


# ════════════════════════════════════════════════════════════════════════════════
# JSON helper
# ════════════════════════════════════════════════════════════════════════════════

def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ════════════════════════════════════════════════════════════════════════════════
# Sheet ID / discipline helpers
# ════════════════════════════════════════════════════════════════════════════════

_PREFIX_MAP: Dict[str, str] = {
    "AD": "Architectural Demolition", "A": "Architectural", "S": "Structural",
    "D": "Demolition", "M": "Mechanical", "E": "Electrical", "P": "Plumbing",
    "FP": "Fire Protection", "L": "Landscape", "C": "Civil", "G": "General",
    "T": "Title/Index", "I": "Interior",
}

_SHEET_ID_PATTERNS = [
    re.compile(r'\b((?:AD|FP)[.-]?\d{1,3}[.-]?\d{0,2}[A-Z]?)\b', re.IGNORECASE),
    re.compile(r'\b([A-Z][.-]?\d{1,3}[.-]?\d{0,2}[A-Z]?)\b'),
    re.compile(r'(?:SHEET|DWG)\s*[:#]?\s*([A-Z]{1,2}[.-]?\d+[.-]?\d*)', re.IGNORECASE),
]

def _detect_sheet_id(text: str, page_number: int) -> str:
    for pat in _SHEET_ID_PATTERNS:
        matches = pat.findall(text)
        if matches:
            matches.sort(key=len, reverse=True)
            return matches[0].upper()
    return f"PAGE-{page_number}"

def _detect_discipline(sheet_id: str, text: str) -> str:
    sid = sheet_id.upper()
    if sid.startswith("PAGE-"):
        return "Unknown"
    for prefix in sorted(_PREFIX_MAP, key=len, reverse=True):
        if sid.startswith(prefix):
            return _PREFIX_MAP[prefix]
    return "Unknown"


# ════════════════════════════════════════════════════════════════════════════════
# Page matching
# ════════════════════════════════════════════════════════════════════════════════

def _match_pages_to_index(
    all_pages, sheet_index_map, index_page_idxs,
):
    n_pages = len(all_pages)
    index_ids = list(sheet_index_map.keys())
    sid_to_pages = {sid: [] for sid in index_ids}
    page_to_matched = {}

    for pg_idx, (_cf, page) in enumerate(all_pages):
        if pg_idx in index_page_idxs:
            continue
        text_upper = page.extracted_text.upper()
        for sid in index_ids:
            if re.search(r'\b' + re.escape(sid) + r'\b', text_upper):
                page_to_matched.setdefault(pg_idx, []).append(sid)
                sid_to_pages[sid].append(pg_idx)

    assigned_pages = {}
    assigned_ids = {}

    for pg_idx in sorted(index_page_idxs):
        if index_ids and index_ids[0] not in assigned_ids and pg_idx not in assigned_pages:
            assigned_pages[pg_idx] = index_ids[0]
            assigned_ids[index_ids[0]] = pg_idx

    for sid in index_ids:
        if sid in assigned_ids:
            continue
        candidates = sid_to_pages[sid]
        if len(candidates) == 1 and candidates[0] not in assigned_pages:
            assigned_pages[candidates[0]] = sid
            assigned_ids[sid] = candidates[0]

    for sid in index_ids:
        if sid in assigned_ids:
            continue
        candidates = [p for p in sid_to_pages[sid] if p not in assigned_pages]
        if candidates:
            best = min(candidates, key=lambda p: len(page_to_matched.get(p, [])))
            assigned_pages[best] = sid
            assigned_ids[sid] = best

    unassigned_pages = sorted(i for i in range(n_pages) if i not in assigned_pages)
    unassigned_ids = [sid for sid in index_ids if sid not in assigned_ids]
    for pg, sid in zip(unassigned_pages, unassigned_ids):
        assigned_pages[pg] = sid
        assigned_ids[sid] = pg

    result = {}
    for pg_idx, sid in assigned_pages.items():
        title, discipline = sheet_index_map[sid]
        result[pg_idx] = (sid, title, discipline)
    logger.info(f"  Page matching: {len(result)}/{len(index_ids)} index entries mapped")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

_KEEP_CATEGORIES = {
    DocumentCategory.PROJECT_SPECIFICATIONS,
    DocumentCategory.CONSTRUCTION_DRAWINGS,
}


def extract_sheets_and_tables(
    classification: DocumentClassificationResult,
) -> Tuple[List[SheetInfo], List[ExtractedTable], List[ExtractedScheduleRow]]:
    """Combined Step 2a: img2table detection → Gemini filter → Gemini extraction.

    Phase 1: img2table (no OCR, <1s/page) detects table bboxes at 150 DPI
    Phase 2: Crops at 150 DPI sent to Gemini Flash 5-at-a-time to filter noise
    Phase 3: Confirmed tables re-cropped at 300 DPI, sent to Gemini Flash for extraction
    """
    client = _get_genai_client()

    kept_files = [f for f in classification.files if f.categories & _KEEP_CATEGORIES]
    drawing_files = [f for f in kept_files if DocumentCategory.CONSTRUCTION_DRAWINGS in f.categories]

    if not drawing_files:
        logger.warning("No drawing files found")
        return [], [], []

    all_pages: List[Tuple[ClassifiedFile, PageInfo]] = []
    for cf in drawing_files:
        for page in cf.pages:
            all_pages.append((cf, page))

    pdf_bytes_map = classification.raw_pdf_bytes
    logger.info(f"Step 2a: {len(all_pages)} drawing pages from {len(drawing_files)} files")

    # ── Phase 1: img2table bbox detection ────────────────────────────────
    logger.info(f"  Phase 1: img2table detection at {DETECT_DPI} DPI...")

    all_detections = []  # list of {global_page, bbox, page_w, page_h, local_page, filename}

    def _detect_one(cf, page):
        pdf_bytes = pdf_bytes_map.get(cf.filename)
        if not pdf_bytes:
            return []
        return _detect_bboxes_on_page(pdf_bytes, page.page_number, page.global_page_number)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_detect_one, cf, page): (cf, page)
            for cf, page in all_pages
        }
        for future in as_completed(futures):
            cf, page = futures[future]
            try:
                dets = future.result()
                for d in dets:
                    d["local_page"] = page.page_number
                    d["filename"] = cf.filename
                all_detections.extend(dets)
            except Exception as e:
                logger.error(f"  Detection failed page {page.global_page_number}: {e}")

    logger.info(f"  Phase 1 done: {len(all_detections)} candidate regions on {len(set(d['global_page'] for d in all_detections))} pages")

    if not all_detections:
        logger.warning("  No table regions detected")
        return _build_sheets_only(all_pages, {}, drawing_files), [], []

    # ── Phase 2: Gemini Flash filtering ──────────────────────────────────
    logger.info(f"  Phase 2: Filtering {len(all_detections)} crops with Gemini Flash...")

    # Render 150 DPI crops for filtering
    filter_crops = []  # (crop_idx, jpeg_bytes)
    for i, det in enumerate(all_detections):
        pdf_bytes = pdf_bytes_map.get(det["filename"])
        if not pdf_bytes:
            continue
        crop = _crop_from_page(pdf_bytes, det["local_page"], det["bbox"], DETECT_DPI)
        filter_crops.append((i, crop))

    # Batch and filter in parallel
    filter_batches = [filter_crops[i:i + FILTER_BATCH] for i in range(0, len(filter_crops), FILTER_BATCH)]
    crop_types = {}  # crop_idx → type

    def _run_filter(batch_idx, batch):
        results = _filter_batch(client, batch)
        return results

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_run_filter, i, b): i for i, b in enumerate(filter_batches)}
        for future in as_completed(futures):
            try:
                results = future.result()
                for crop_idx, crop_type in results:
                    crop_types[crop_idx] = crop_type
            except Exception as e:
                batch_idx = futures[future]
                logger.error(f"  Filter batch {batch_idx} failed: {e}")
                # Keep all on failure
                for idx, _ in filter_batches[batch_idx]:
                    crop_types[idx] = "table"

    # Keep only tables and sheet_index
    kept = [(i, all_detections[i], crop_types.get(i, "table"))
            for i in range(len(all_detections))
            if crop_types.get(i, "table") in ("table", "sheet_index")]
    discarded = len(all_detections) - len(kept)
    logger.info(f"  Phase 2 done: {len(kept)} tables kept, {discarded} filtered out")

    # ── Phase 3: Gemini Flash extraction at 300 DPI ──────────────────────
    logger.info(f"  Phase 3: Extracting {len(kept)} tables at {CROP_DPI} DPI...")

    # Render 300 DPI crops
    extract_crops = []  # (crop_idx, jpeg_bytes, global_page, crop_type)
    for crop_idx, det, crop_type in kept:
        pdf_bytes = pdf_bytes_map.get(det["filename"])
        if not pdf_bytes:
            continue
        crop = _crop_from_page(pdf_bytes, det["local_page"], det["bbox"], CROP_DPI)
        extract_crops.append((crop_idx, crop, det["global_page"], crop_type))

    # Batch and extract in parallel
    extract_batches = [extract_crops[i:i + EXTRACT_BATCH] for i in range(0, len(extract_crops), EXTRACT_BATCH)]

    all_sheet_index_rows = []
    all_raw_tables = []  # (global_page, crop_type, parsed_dict)

    def _run_extract(batch_idx, batch):
        batch_for_api = [(idx, crop_bytes) for idx, crop_bytes, _, _ in batch]
        results = _extract_batch(client, batch_for_api)
        return [(batch[min(i, len(batch)-1)][2], batch[min(i, len(batch)-1)][3], parsed)
                for i, (_, parsed) in enumerate(results)]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_run_extract, i, b): i for i, b in enumerate(extract_batches)}
        for future in as_completed(futures):
            try:
                results = future.result()
                for gpn, ctype, parsed in results:
                    if ctype == "sheet_index":
                        for row in parsed.get("rows", []):
                            if isinstance(row, dict):
                                all_sheet_index_rows.append(row)
                        logger.info(f"    Page {gpn}: sheet index → {len(parsed.get('rows', []))} entries")
                    else:
                        all_raw_tables.append((gpn, ctype, parsed))
                        logger.info(f"    Page {gpn}: {parsed.get('table_title', 'table')} → {len(parsed.get('rows', []))} rows")
            except Exception as e:
                logger.error(f"  Extract batch failed: {e}")

    logger.info(f"  Phase 3 done: {len(all_sheet_index_rows)} index entries, {len(all_raw_tables)} tables")

    # ── Build sheet index map ────────────────────────────────────────────
    sheet_index_map: Dict[str, Tuple[str, str]] = {}
    for entry in all_sheet_index_rows:
        sid = ""
        title = ""
        discipline = "General"
        for key, val in entry.items():
            key_upper = key.upper()
            val_str = str(val).strip()
            if any(k in key_upper for k in ("SHEET", "NUMBER", "NO", "ID", "MARK")):
                sid = val_str.upper()
            elif any(k in key_upper for k in ("TITLE", "NAME", "DESCRIPTION")):
                title = val_str
            elif any(k in key_upper for k in ("DISCIPLINE", "SECTION", "GROUP")):
                discipline = val_str.title()
        if not sid and len(entry) >= 2:
            values = list(entry.values())
            sid = str(values[0]).strip().upper()
            title = str(values[1]).strip() if len(values) > 1 else ""
        if sid and re.match(r'^[A-Z]', sid):
            if not discipline or discipline == "General":
                discipline = _detect_discipline(sid, "")
            sheet_index_map[sid] = (title, discipline)

    if sheet_index_map:
        logger.info(f"  Sheet index: {len(sheet_index_map)} sheets mapped")

    # ── Build SheetInfo + ExtractedTable ─────────────────────────────────
    return _build_final_output(
        all_pages, sheet_index_map, all_raw_tables, drawing_files, pdf_bytes_map,
    )


def _build_sheets_only(all_pages, sheet_index_map, drawing_files):
    """Build SheetInfo objects without tables."""
    sheets = []
    for i, (cf, page) in enumerate(all_pages):
        sheet_id = _detect_sheet_id(page.extracted_text, page.global_page_number)
        discipline = _detect_discipline(sheet_id, page.extracted_text)
        sheets.append(SheetInfo(
            sheet_id=sheet_id, title="", discipline=discipline,
            global_page_number=page.global_page_number,
            source_file=cf.filename, extracted_text=page.extracted_text,
            image_bytes=page.image_bytes, tables=[],
        ))
    return sheets


def _build_final_output(all_pages, sheet_index_map, all_raw_tables, drawing_files, pdf_bytes_map):
    """Build SheetInfo objects, ExtractedTables, and schedule rows."""
    # Match pages to index
    page_assignments = {}
    if sheet_index_map:
        page_assignments = _match_pages_to_index(all_pages, sheet_index_map, {0})

    # Build SheetInfo
    sheets = []
    for i, (cf, page) in enumerate(all_pages):
        if i in page_assignments:
            sheet_id, title, discipline = page_assignments[i]
        else:
            sheet_id = _detect_sheet_id(page.extracted_text, page.global_page_number)
            title = ""
            discipline = _detect_discipline(sheet_id, page.extracted_text)
        sheets.append(SheetInfo(
            sheet_id=sheet_id, title=title, discipline=discipline,
            global_page_number=page.global_page_number,
            source_file=cf.filename, extracted_text=page.extracted_text,
            image_bytes=page.image_bytes, tables=[],
        ))

    logger.info(f"  Built {len(sheets)} SheetInfo objects")
    disciplines = {}
    for s in sheets:
        disciplines[s.discipline] = disciplines.get(s.discipline, 0) + 1
    for disc, count in sorted(disciplines.items()):
        logger.info(f"    {disc}: {count}")

    # Build ExtractedTable
    pg_to_sheet = {s.global_page_number: s for s in sheets}
    all_tables = []

    for gpn, ctype, parsed in all_raw_tables:
        table_list = [parsed] if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
        for t_data in table_list:
            if not isinstance(t_data, dict):
                continue
            headers = t_data.get("headers", [])
            rows = t_data.get("rows", [])
            if not rows:
                continue
            clean_rows = [r for r in rows if isinstance(r, dict) and any(str(v).strip() for v in r.values())]
            if not clean_rows:
                continue
            sheet = pg_to_sheet.get(gpn)
            sheet_id = sheet.sheet_id if sheet else f"PAGE-{gpn}"
            stype = t_data.get("schedule_type", ctype or "other")
            table = ExtractedTable(
                page_number=gpn, sheet_id=sheet_id, schedule_type=stype,
                headers=headers, rows=clean_rows, confidence=0.90,
            )
            all_tables.append(table)
            if sheet:
                sheet.tables.append(table)

    logger.info(f"  Extracted {len(all_tables)} tables")
    type_counts = {}
    for t in all_tables:
        type_counts[t.schedule_type] = type_counts.get(t.schedule_type, 0) + 1
    for stype, count in sorted(type_counts.items()):
        logger.info(f"    {stype}: {count}")

    rows = tables_to_schedule_rows(all_tables)
    return sheets, all_tables, rows


# ════════════════════════════════════════════════════════════════════════════════
# Schedule row conversion
# ════════════════════════════════════════════════════════════════════════════════

_MARK_HEADERS = {
    "MARK", "NO", "NO.", "NUMBER", "DOOR NO", "DOOR NO.",
    "ROOM", "ROOM NO", "ROOM NO.", "ROOM NUMBER", "SPACE",
    "TYPE", "TAG", "SYMBOL", "SET", "HW SET", "HARDWARE SET",
    "CKT", "CIRCUIT", "FIXTURE",
}

def _find_mark_column(headers):
    for i, h in enumerate(headers):
        if h.upper().strip() in _MARK_HEADERS:
            return i
    return 0

def tables_to_schedule_rows(tables):
    rows = []
    for table in tables:
        if table.schedule_type in ("unknown",):
            continue
        mark_col = _find_mark_column(table.headers)
        for row_data in table.rows:
            mark = ""
            if mark_col is not None and mark_col < len(table.headers):
                mark_header = table.headers[mark_col]
                mark = row_data.get(mark_header, "").strip()
            rows.append(ExtractedScheduleRow(
                schedule_type=table.schedule_type, row_data=row_data,
                mark=mark, page_number=table.page_number, sheet_id=table.sheet_id,
            ))
    logger.info(f"  Converted {len(rows)} schedule rows from {len(tables)} tables")
    return rows


# ════════════════════════════════════════════════════════════════════════════════
# Schedule rows → EstimateItems
# ════════════════════════════════════════════════════════════════════════════════

_SCHEDULE_TYPE_TO_TRADE = {
    "door": "Doors and Windows", "window": "Doors and Windows",
    "finish": "Painting", "fixture": "Plumbing", "plumbing_fixture": "Plumbing",
    "equipment": "General Requirements", "panel": "Electrical",
    "lighting": "Electrical", "hardware": "Doors and Windows",
    "ventilation": "HVAC and Sheet Metals", "mechanical": "HVAC and Sheet Metals",
}

def schedule_rows_to_estimate_items(schedule_rows):
    items = []
    for row in schedule_rows:
        trade = _SCHEDULE_TYPE_TO_TRADE.get(row.schedule_type, "General Requirements")
        desc_parts = []
        if row.mark:
            desc_parts.append(f"{row.schedule_type.title()} {row.mark}")
        for key in ("TYPE", "DESCRIPTION", "SIZE", "MATERIAL", "FINISH"):
            val = row.row_data.get(key, "").strip()
            if not val:
                val = row.row_data.get(key.lower(), "").strip()
            if val:
                desc_parts.append(val)
        description = " — ".join(desc_parts) if desc_parts else f"{row.schedule_type.title()} item"
        material_parts = []
        for key in ("MATERIAL", "FRAME", "FRAME MATERIAL", "GLAZING", "GLASS",
                     "MANUFACTURER", "MFR", "MODEL", "CATALOG"):
            val = row.row_data.get(key, "").strip()
            if val:
                material_parts.append(f"{key}: {val}")
        notes_parts = []
        for key in ("REMARKS", "NOTES", "COMMENTS"):
            val = row.row_data.get(key, "").strip()
            if val:
                notes_parts.append(val)
        item = EstimateItem(
            trade=trade, item_description=description, qty=1.0, unit="EA",
            extraction_method="schedule_parse", confidence=0.90,
            source_page=row.page_number, sheet_id=row.sheet_id,
            material_spec="; ".join(material_parts), schedule_mark=row.mark,
            notes="; ".join(notes_parts), source=f"schedule:{row.schedule_type}",
        )
        items.append(item)
    logger.info(f"  Created {len(items)} estimate items from schedule rows")
    return items
