"""Step 2a — Sheet Index + Table Extraction (Pure Gemini Flash).

2-phase approach (no img2table, no OpenCV):
  Phase 1: Send all page images to Gemini Flash (5 per call, parallel)
           to scan for tables/schedules on each page
  Phase 2: Re-render pages with tables at 300 DPI, send to Gemini Flash
           for accurate content extraction
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import fitz  # PyMuPDF
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

SCAN_BATCH = 5        # pages per Flash call in Phase 1
EXTRACT_DPI = 300     # DPI for Phase 2 extraction
WORKERS = 10          # parallel Gemini calls


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
# Phase 1: Scan pages for tables with Gemini Flash
# ════════════════════════════════════════════════════════════════════════════════

SCAN_PROMPT = """\
You are analyzing construction drawing pages to find tables and schedules.

For EACH page image below, identify ALL tables, schedules, and sheet indices.
Construction drawings often have MIXED content — a single page may contain
a floor plan AND a door schedule AND a finish schedule side by side.

What counts as a table/schedule:
- Door schedules, window schedules, finish schedules, fixture schedules
- Panel schedules, lighting schedules, equipment schedules
- Plumbing fixture schedules, mechanical equipment schedules
- Hardware groups/sets, glass type tables
- Sheet index / drawing index (list of sheet numbers and titles)
- Code review data tables, inspection schedules
- Abbreviation tables/legends with structured columns
- ANY structured data with rows and columns (even small 2-3 row tables)
- Quantity summary tables, bid item tables

What is NOT a table:
- Floor plans, ceiling plans, site plans (even if they have grid lines)
- Elevation drawings, section drawings, detail drawings
- Note blocks (paragraphs of text without column structure)
- Title blocks (the standard border info at bottom-right)
- Symbol legends that are just a list of symbols with labels (no columns)
- Dimension strings on drawings

IMPORTANT: Do NOT miss any table. When in doubt, include it.
Be especially careful with pages that mix plans and schedules — look at
ALL areas of the page, including margins, sidebars, and corners.

For each page, return:
- page_index: the index number shown above the image
- has_tables: true/false
- tables: array of objects with "title" and "type" for each table found
  type: one of "sheet_index", "door", "window", "finish", "fixture",
        "equipment", "panel", "lighting", "hardware", "ventilation",
        "mechanical", "plumbing_fixture", "inspection", "abbreviation",
        "code_data", "quantity", "other"

Return JSON array:
[
  {
    "page_index": 0,
    "has_tables": true,
    "tables": [
      {"title": "SHEET INDEX", "type": "sheet_index"},
      {"title": "DOOR SCHEDULE", "type": "door"}
    ]
  },
  {
    "page_index": 1,
    "has_tables": false,
    "tables": []
  }
]

Return ONLY valid JSON, no markdown fences."""


def _scan_batch(client, batch: List[Tuple[int, bytes]]) -> List[dict]:
    """Send a batch of page images to Flash to find tables."""
    contents = [SCAN_PROMPT]
    for page_idx, img_bytes in batch:
        contents.append(f"--- page_index={page_idx} ---")
        contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=4096,
            http_options=genai_types.HttpOptions(timeout=180_000),
        ),
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
                return [{"page_index": idx, "has_tables": True, "tables": []} for idx, _ in batch]
        else:
            return [{"page_index": idx, "has_tables": True, "tables": []} for idx, _ in batch]

    return results


# ════════════════════════════════════════════════════════════════════════════════
# Phase 2: Extract tables at high DPI with Gemini Flash
# ════════════════════════════════════════════════════════════════════════════════

EXTRACT_PROMPT = """\
You are reading construction drawing pages to extract ALL tables and schedules.

This page has been identified as containing tables. Extract EVERY table you see.
A single page may have MULTIPLE tables (e.g., door schedule + finish schedule +
abbreviations table all on one page). Extract ALL of them.

For EACH table found, return a JSON object with:
- table_title: the title/header of the table (e.g., "DOOR SCHEDULE")
- schedule_type: one of sheet_index, door, window, finish, fixture, equipment,
  panel, lighting, hardware, ventilation, mechanical, plumbing_fixture,
  inspection, abbreviation, code_data, quantity, other
- headers: array of column header names
- rows: array of objects mapping header → cell value

Rules for reading cell values:
- Copy EXACTLY what is written — do NOT interpret, correct, or guess
- Common construction abbreviations you MUST preserve exactly:
  HM = Hollow Metal, SC = Sealed Concrete, RB = Rubber Base,
  ACT = Acoustical Ceiling Tile, GWB = Gypsum Wall Board,
  OHD = Overhead Door, EXP = Exposed, FBO = Furnished By Owner,
  (P) = Existing, (E) = Existing, (N) = New, NIC = Not In Contract,
  MTL PNL = Metal Panel, PT = Paint, LVT = Luxury Vinyl Tile,
  WD = Wood, GL = Glass, GYP = Gypsum Wall Board, STL = Steel
- Extract EVERY row — do NOT skip any
- Empty cells = ""
- If you cannot read a cell clearly, use "?" — do NOT hallucinate
- Numbers as strings (preserve formatting like 3'-0", 1-3/4")
- Merged cells: repeat value in each row
- If a table has subheaders or grouped columns, flatten them
  (e.g., "DOORS | MTL" becomes "DOORS MTL")

For SHEET INDEX tables specifically:
- headers should be ["Sheet Number", "Description"]
- Extract every sheet number and its description/title

If multiple tables exist on this page, return a JSON array of table objects.
If only one table, still return it inside an array.

Return ONLY valid JSON, no markdown fences."""


def _extract_page(client, page_idx: int, img_bytes: bytes) -> List[dict]:
    """Send a high-DPI page image to Flash for table extraction."""
    contents = [
        EXTRACT_PROMPT,
        f"--- Page {page_idx} ---",
        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(
            temperature=0, max_output_tokens=32768,
            http_options=genai_types.HttpOptions(timeout=180_000),
        ),
    )

    raw = (response.text or "[]").strip()
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

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    return [t for t in parsed if isinstance(t, dict)]


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
    """Step 2a: Pure Gemini Flash table scan + extraction.

    Phase 1: Send all page images to Flash (5 per call, parallel) to find tables
    Phase 2: Re-render pages with tables at 300 DPI, extract with Flash
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

    # ── Phase 1: Scan all pages for tables ────────────────────────────────
    logger.info(f"  Phase 1: Scanning {len(all_pages)} pages with Gemini Flash...")

    # Use existing page images from classification (already rendered)
    page_images = []  # (local_idx_in_all_pages, global_page_number, image_bytes)
    for i, (cf, page) in enumerate(all_pages):
        if page.image_bytes:
            page_images.append((i, page.global_page_number, page.image_bytes))
        else:
            # Render if not available
            pdf_bytes = pdf_bytes_map.get(cf.filename)
            if pdf_bytes:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                zoom = 150 / 72.0
                pix = doc[page.page_number].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                page_images.append((i, page.global_page_number, pix.tobytes("png")))
                doc.close()

    # Build batches of (global_page_number, image_bytes)
    scan_items = [(gpn, img) for _, gpn, img in page_images]
    batches = [scan_items[i:i + SCAN_BATCH] for i in range(0, len(scan_items), SCAN_BATCH)]

    pages_with_tables: Dict[int, List[dict]] = {}  # global_page → table info list

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_scan_batch, client, b): bi for bi, b in enumerate(batches)}
        for future in as_completed(futures):
            bi = futures[future]
            try:
                results = future.result()
                for r in results:
                    if isinstance(r, dict) and r.get("has_tables"):
                        pidx = r.get("page_index", -1)
                        tables = r.get("tables", [])
                        pages_with_tables[pidx] = tables
                        titles = [t.get("title", "?") for t in tables]
                        logger.info(f"    Page {pidx}: {', '.join(titles)}")
            except Exception as e:
                logger.error(f"  Scan batch {bi} failed: {e}")
                # On failure, mark all pages in batch as having tables
                for gpn, _ in batches[bi]:
                    pages_with_tables[gpn] = []

    logger.info(f"  Phase 1 done: {len(pages_with_tables)} pages have tables")

    if not pages_with_tables:
        logger.warning("  No tables found on any page")
        return _build_sheets_only(all_pages, {}, drawing_files), [], []

    # ── Phase 2: Extract tables at 300 DPI ────────────────────────────────
    logger.info(f"  Phase 2: Extracting tables from {len(pages_with_tables)} pages at {EXTRACT_DPI} DPI...")

    # Build global_page → (filename, local_page_idx) map
    gpn_to_file = {}
    for cf, page in all_pages:
        gpn_to_file[page.global_page_number] = (cf.filename, page.page_number)

    # Render table pages at high DPI
    hires_images: Dict[int, bytes] = {}
    for gpn in pages_with_tables:
        file_info = gpn_to_file.get(gpn)
        if not file_info:
            continue
        filename, local_idx = file_info
        pdf_bytes = pdf_bytes_map.get(filename)
        if not pdf_bytes:
            continue
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        zoom = EXTRACT_DPI / 72.0
        pix = doc[local_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        hires_images[gpn] = pix.tobytes("png")
        doc.close()

    logger.info(f"    Rendered {len(hires_images)} pages at {EXTRACT_DPI} DPI")

    # Extract in parallel (one page per call for full context)
    all_sheet_index_rows = []
    all_raw_tables = []  # (global_page, schedule_type, parsed_dict)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_extract_page, client, gpn, img): gpn
            for gpn, img in hires_images.items()
        }
        for future in as_completed(futures):
            gpn = futures[future]
            try:
                tables = future.result()
                for t in tables:
                    stype = t.get("schedule_type", "other")
                    title = t.get("table_title", "")
                    n_rows = len(t.get("rows", []))
                    logger.info(f"    Page {gpn}: {title!r} ({stype}) — {n_rows} rows")

                    if stype == "sheet_index":
                        for row in t.get("rows", []):
                            if isinstance(row, dict):
                                all_sheet_index_rows.append(row)
                    else:
                        all_raw_tables.append((gpn, stype, t))
            except Exception as e:
                logger.error(f"    Page {gpn} extraction failed: {e}")

    logger.info(f"  Phase 2 done: {len(all_sheet_index_rows)} index entries, {len(all_raw_tables)} tables")

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
            title = t_data.get("table_title", "")
            table = ExtractedTable(
                page_number=gpn, sheet_id=sheet_id, schedule_type=stype,
                headers=headers, rows=clean_rows, confidence=0.90,
                table_title=title,
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

_QTY_HEADERS = {"QTY", "QUANTITY", "COUNT", "NO.", "NO", "TOTAL", "AMOUNT"}
_UNIT_HEADERS = {"UNIT", "UNITS", "UOM", "U/M"}


def _extract_qty_from_row(row_data: dict) -> tuple:
    """Extract qty and unit from schedule row if QTY/UNIT columns exist.

    Returns (qty, unit) — defaults to (1.0, "EA") if no qty column found.
    """
    qty = 1.0
    unit = "EA"

    for key, val in row_data.items():
        key_upper = key.upper().strip()
        val_str = str(val).strip()
        if key_upper in _QTY_HEADERS and val_str:
            try:
                qty = float(re.sub(r"[^\d.]", "", val_str))
            except (ValueError, TypeError):
                pass
        if key_upper in _UNIT_HEADERS and val_str:
            unit = val_str.upper()

    return qty, unit


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

        qty, unit = _extract_qty_from_row(row.row_data)

        item = EstimateItem(
            trade=trade, item_description=description, qty=qty, unit=unit,
            extraction_method="schedule_parse", confidence=0.90,
            source_page=row.page_number, sheet_id=row.sheet_id,
            material_spec="; ".join(material_parts), schedule_mark=row.mark,
            notes="; ".join(notes_parts), source=f"schedule:{row.schedule_type}",
        )
        items.append(item)
    logger.info(f"  Created {len(items)} estimate items from schedule rows")
    return items
