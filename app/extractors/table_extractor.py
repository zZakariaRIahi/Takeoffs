"""Step 2a — Sheet Index + Table Extraction (Gemini Flash vision).

Sends drawing page images in batches of 5 to Gemini Flash.
One prompt handles both sheet index detection AND schedule extraction.
No Tesseract or img2table — Gemini does everything visually.

4 parallel workers, 5 pages per batch → 40 pages in ~2 batches of calls.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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
import os

logger = logging.getLogger(__name__)

BATCH_SIZE = 5
MAX_WORKERS = 4

# ════════════════════════════════════════════════════════════════════════════════
# Gemini client
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
# Combined prompt — sheet index + table extraction
# ════════════════════════════════════════════════════════════════════════════════

BATCH_PROMPT = """You are analyzing construction drawing pages to extract structured data.

For EACH page image below, do the following:

1. **SHEET INDEX / DRAWING INDEX**: If a page contains a sheet index (list of all drawings
   in the set, usually on the cover/title sheet), extract it as sheet_index entries.

2. **SCHEDULES & TABLES**: If a page contains any schedule or table useful for construction
   estimation (door schedule, window schedule, finish schedule, fixture schedule, equipment
   schedule, panel schedule, lighting schedule, hardware schedule, plumbing fixture schedule,
   mechanical schedule, ventilation schedule, or ANY other data table), extract ALL rows.

3. **SKIP**: If a page has NO tables or schedules (just plans, elevations, sections, details,
   notes, or general drawings), skip it entirely — do not output anything for that page.

Return a JSON object with TWO arrays:

{
  "sheet_index": [
    {"sheet_id": "T101", "title": "Cover Sheet", "discipline": "General"},
    {"sheet_id": "A101", "title": "Floor Plan", "discipline": "Architectural"}
  ],
  "tables": [
    {
      "page_global": 5,
      "table_title": "Door Schedule",
      "schedule_type": "door",
      "headers": ["MARK", "TYPE", "SIZE", "MATERIAL"],
      "rows": [
        {"MARK": "101", "TYPE": "A", "SIZE": "3'-0\" x 7'-0\"", "MATERIAL": "Wood"},
        {"MARK": "102", "TYPE": "B", "SIZE": "6'-0\" x 7'-0\"", "MATERIAL": "Aluminum"}
      ]
    }
  ]
}

RULES:
- sheet_index: only if the page has an actual drawing index/sheet list. Copy EXACT sheet IDs and titles.
  The discipline should be the section header (GENERAL, ARCHITECTURAL, STRUCTURAL, ELECTRICAL, etc.)
- schedule_type: one of: door, window, finish, fixture, equipment, panel, lighting, hardware,
  ventilation, mechanical, plumbing_fixture, other
- Extract EVERY row from EVERY table — do NOT skip any rows
- Use exact header names from the table
- Empty cells: use ""
- Numbers: keep as strings (preserve original formatting)
- Merged cells spanning multiple rows: repeat the value in each row
- page_global: use the global page number shown in the label for each page
- If a page has NO tables/schedules, do not include it in the output at all
- If NO pages in this batch have tables, return: {"sheet_index": [], "tables": []}

Return ONLY valid JSON, no markdown fences."""


# ════════════════════════════════════════════════════════════════════════════════
# Batch processing
# ════════════════════════════════════════════════════════════════════════════════

def _process_batch(
    client,
    batch: List[Tuple[int, bytes]],
) -> dict:
    """Send a batch of page images to Gemini Flash and extract tables.

    Args:
        client: Gemini client
        batch: list of (global_page_number, jpeg_bytes)

    Returns:
        Parsed JSON dict with sheet_index and tables
    """
    contents = [BATCH_PROMPT]
    for global_pg, img_bytes in batch:
        contents.append(f"--- Page (global_page_number={global_pg}) ---")
        contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=32768,
        ),
    )

    raw = (response.text or "{}").strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse batch response: {raw[:300]}")
        return {"sheet_index": [], "tables": []}


# ════════════════════════════════════════════════════════════════════════════════
# Discipline / sheet ID helpers (from sheet_indexer)
# ════════════════════════════════════════════════════════════════════════════════

_PREFIX_MAP: Dict[str, str] = {
    "AD": "Architectural Demolition",
    "A":  "Architectural",
    "S":  "Structural",
    "D":  "Demolition",
    "M":  "Mechanical",
    "E":  "Electrical",
    "P":  "Plumbing",
    "FP": "Fire Protection",
    "L":  "Landscape",
    "C":  "Civil",
    "G":  "General",
    "T":  "Title/Index",
    "I":  "Interior",
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
# Page matching (from sheet_indexer)
# ════════════════════════════════════════════════════════════════════════════════

def _match_pages_to_index(
    all_pages: List[Tuple[ClassifiedFile, PageInfo]],
    sheet_index_map: Dict[str, Tuple[str, str]],
    index_page_idxs: set,
) -> Dict[int, Tuple[str, str, str]]:
    """Match PDF pages to sheet index entries."""
    n_pages = len(all_pages)
    index_ids = list(sheet_index_map.keys())

    page_to_matched_ids: Dict[int, List[str]] = {}
    sid_to_pages: Dict[str, List[int]] = {sid: [] for sid in index_ids}

    for pg_idx, (_cf, page) in enumerate(all_pages):
        if pg_idx in index_page_idxs:
            continue
        text_upper = page.extracted_text.upper()
        for sid in index_ids:
            pattern = r'\b' + re.escape(sid) + r'\b'
            if re.search(pattern, text_upper):
                page_to_matched_ids.setdefault(pg_idx, []).append(sid)
                sid_to_pages[sid].append(pg_idx)

    assigned_pages: Dict[int, str] = {}
    assigned_ids: Dict[str, int] = {}

    for pg_idx in sorted(index_page_idxs):
        if index_ids and index_ids[0] not in assigned_ids and pg_idx not in assigned_pages:
            assigned_pages[pg_idx] = index_ids[0]
            assigned_ids[index_ids[0]] = pg_idx

    for sid in index_ids:
        if sid in assigned_ids:
            continue
        candidates = sid_to_pages[sid]
        if len(candidates) == 1:
            pg = candidates[0]
            if pg not in assigned_pages:
                assigned_pages[pg] = sid
                assigned_ids[sid] = pg

    for sid in index_ids:
        if sid in assigned_ids:
            continue
        candidates = [p for p in sid_to_pages[sid] if p not in assigned_pages]
        if candidates:
            best = min(candidates, key=lambda p: len(page_to_matched_ids.get(p, [])))
            assigned_pages[best] = sid
            assigned_ids[sid] = best

    unassigned_pages = sorted(i for i in range(n_pages) if i not in assigned_pages)
    unassigned_ids = [sid for sid in index_ids if sid not in assigned_ids]
    if unassigned_pages and unassigned_ids:
        for pg, sid in zip(unassigned_pages, unassigned_ids):
            assigned_pages[pg] = sid
            assigned_ids[sid] = pg

    result: Dict[int, Tuple[str, str, str]] = {}
    for pg_idx, sid in assigned_pages.items():
        title, discipline = sheet_index_map[sid]
        result[pg_idx] = (sid, title, discipline)

    logger.info(f"Page matching: {len(result)}/{len(index_ids)} index entries mapped")
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point — replaces build_sheet_index + extract_tables_from_sheets
# ════════════════════════════════════════════════════════════════════════════════

_KEEP_CATEGORIES = {
    DocumentCategory.PROJECT_SPECIFICATIONS,
    DocumentCategory.CONSTRUCTION_DRAWINGS,
}


def extract_sheets_and_tables(
    classification: DocumentClassificationResult,
) -> Tuple[List[SheetInfo], List[ExtractedTable], List[ExtractedScheduleRow]]:
    """Combined Step 2a: sheet index + table extraction using Gemini Flash.

    Sends drawing page images in batches of 5 to Gemini Flash.
    4 parallel workers. One prompt handles both sheet index and tables.

    Returns: (sheets, tables, schedule_rows)
    """
    client = _get_genai_client()

    # Filter to drawing + spec files
    kept_files = [f for f in classification.files if f.categories & _KEEP_CATEGORIES]
    drawing_files = [f for f in kept_files if DocumentCategory.CONSTRUCTION_DRAWINGS in f.categories]

    if not drawing_files:
        logger.warning("No drawing files found")
        return [], [], []

    logger.info(f"Step 2a: {len(drawing_files)} drawing files, sending pages in batches of {BATCH_SIZE}")

    # Collect all drawing pages with images
    all_pages: List[Tuple[ClassifiedFile, PageInfo]] = []
    batch_items: List[Tuple[int, bytes]] = []  # (global_page_number, jpeg_bytes)

    for cf in drawing_files:
        for page in cf.pages:
            all_pages.append((cf, page))
            if page.image_bytes:
                batch_items.append((page.global_page_number, page.image_bytes))

    logger.info(f"  {len(batch_items)} pages with images to process")

    # Split into batches of BATCH_SIZE
    batches = [batch_items[i:i + BATCH_SIZE] for i in range(0, len(batch_items), BATCH_SIZE)]
    logger.info(f"  {len(batches)} batches, {MAX_WORKERS} parallel workers")

    # Process batches in parallel
    all_sheet_index = []
    all_raw_tables = []

    def _run_batch(batch_idx, batch):
        logger.info(f"  Batch {batch_idx + 1}/{len(batches)}: {len(batch)} pages")
        result = _process_batch(client, batch)
        logger.info(
            f"  Batch {batch_idx + 1} done: "
            f"{len(result.get('sheet_index', []))} index entries, "
            f"{len(result.get('tables', []))} tables"
        )
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_run_batch, i, batch): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                result = future.result()
                all_sheet_index.extend(result.get("sheet_index", []))
                all_raw_tables.extend(result.get("tables", []))
            except Exception as e:
                logger.error(f"  Batch {batch_idx + 1} failed: {e}")

    logger.info(f"  All batches done: {len(all_sheet_index)} index entries, {len(all_raw_tables)} raw tables")

    # ── Build sheet index map ────────────────────────────────────────────
    sheet_index_map: Dict[str, Tuple[str, str]] = {}
    for entry in all_sheet_index:
        sid = str(entry.get("sheet_id", "")).strip().upper()
        title = str(entry.get("title", "")).strip()
        discipline = str(entry.get("discipline", "General")).strip().title()
        if sid and re.match(r'^[A-Z]', sid):
            sheet_index_map[sid] = (title, discipline)

    if sheet_index_map:
        logger.info(f"  Sheet index: {len(sheet_index_map)} sheets mapped")
    else:
        logger.warning("  No sheet index found — using detected sheet IDs")

    # ── Match pages to sheet index ───────────────────────────────────────
    page_assignments: Dict[int, Tuple[str, str, str]] = {}
    if sheet_index_map:
        index_page_idxs = {0}  # Assume page 0 is the title sheet
        page_assignments = _match_pages_to_index(all_pages, sheet_index_map, index_page_idxs)

    # ── Build SheetInfo objects ──────────────────────────────────────────
    sheets: List[SheetInfo] = []
    for i, (cf, page) in enumerate(all_pages):
        if i in page_assignments:
            sheet_id, title, discipline = page_assignments[i]
        else:
            sheet_id = _detect_sheet_id(page.extracted_text, page.global_page_number)
            title = ""
            discipline = _detect_discipline(sheet_id, page.extracted_text)

        sheets.append(SheetInfo(
            sheet_id=sheet_id,
            title=title,
            discipline=discipline,
            global_page_number=page.global_page_number,
            source_file=cf.filename,
            extracted_text=page.extracted_text,
            image_bytes=page.image_bytes,
            tables=[],
        ))

    logger.info(f"  Built {len(sheets)} SheetInfo objects")

    # Log discipline breakdown
    disciplines: Dict[str, int] = {}
    for s in sheets:
        disciplines[s.discipline] = disciplines.get(s.discipline, 0) + 1
    for disc, count in sorted(disciplines.items()):
        logger.info(f"    {disc}: {count} sheets")

    # ── Build ExtractedTable objects ─────────────────────────────────────
    # Map global_page_number → sheet for attaching tables
    pg_to_sheet: Dict[int, SheetInfo] = {s.global_page_number: s for s in sheets}

    all_tables: List[ExtractedTable] = []
    for raw_t in all_raw_tables:
        headers = raw_t.get("headers", [])
        rows = raw_t.get("rows", [])
        if not rows:
            continue

        # Filter empty rows
        clean_rows = [r for r in rows if isinstance(r, dict) and any(str(v).strip() for v in r.values())]
        if not clean_rows:
            continue

        pg_num = raw_t.get("page_global", -1)
        sheet = pg_to_sheet.get(pg_num)
        sheet_id = sheet.sheet_id if sheet else f"PAGE-{pg_num}"

        table = ExtractedTable(
            page_number=pg_num,
            sheet_id=sheet_id,
            schedule_type=raw_t.get("schedule_type", "other"),
            headers=headers,
            rows=clean_rows,
            confidence=0.90,
        )
        all_tables.append(table)
        if sheet:
            sheet.tables.append(table)

    logger.info(f"  Extracted {len(all_tables)} tables total")
    type_counts: Dict[str, int] = {}
    for t in all_tables:
        type_counts[t.schedule_type] = type_counts.get(t.schedule_type, 0) + 1
    for stype, count in sorted(type_counts.items()):
        logger.info(f"    {stype}: {count}")

    # ── Convert to schedule rows ─────────────────────────────────────────
    rows = tables_to_schedule_rows(all_tables)

    return sheets, all_tables, rows


# ════════════════════════════════════════════════════════════════════════════════
# Schedule row conversion (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

_MARK_HEADERS = {
    "MARK", "NO", "NO.", "NUMBER", "DOOR NO", "DOOR NO.",
    "ROOM", "ROOM NO", "ROOM NO.", "ROOM NUMBER", "SPACE",
    "TYPE", "TAG", "SYMBOL", "SET", "HW SET", "HARDWARE SET",
    "CKT", "CIRCUIT", "FIXTURE",
}


def _find_mark_column(headers: List[str]) -> Optional[int]:
    for i, h in enumerate(headers):
        if h.upper().strip() in _MARK_HEADERS:
            return i
    return 0


def tables_to_schedule_rows(tables: List[ExtractedTable]) -> List[ExtractedScheduleRow]:
    """Flatten tables into individual rows for downstream linking."""
    rows: List[ExtractedScheduleRow] = []
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
                schedule_type=table.schedule_type,
                row_data=row_data,
                mark=mark,
                page_number=table.page_number,
                sheet_id=table.sheet_id,
            ))
    logger.info(f"  Converted {len(rows)} schedule rows from {len(tables)} tables")
    return rows


# ════════════════════════════════════════════════════════════════════════════════
# Schedule rows → EstimateItems (unchanged)
# ════════════════════════════════════════════════════════════════════════════════

_SCHEDULE_TYPE_TO_TRADE: Dict[str, str] = {
    "door": "Doors and Windows",
    "window": "Doors and Windows",
    "finish": "Painting",
    "fixture": "Plumbing",
    "plumbing_fixture": "Plumbing",
    "equipment": "General Requirements",
    "panel": "Electrical",
    "lighting": "Electrical",
    "hardware": "Doors and Windows",
    "ventilation": "HVAC and Sheet Metals",
    "mechanical": "HVAC and Sheet Metals",
}


def schedule_rows_to_estimate_items(
    schedule_rows: List[ExtractedScheduleRow],
) -> List[EstimateItem]:
    items: List[EstimateItem] = []
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
            trade=trade,
            item_description=description,
            qty=1.0,
            unit="EA",
            extraction_method="schedule_parse",
            confidence=0.90,
            source_page=row.page_number,
            sheet_id=row.sheet_id,
            material_spec="; ".join(material_parts),
            schedule_mark=row.mark,
            notes="; ".join(notes_parts),
            source=f"schedule:{row.schedule_type}",
        )
        items.append(item)
    logger.info(f"  Created {len(items)} estimate items from schedule rows")
    return items
