"""Table Extractor — img2table bbox detection + Gemini Flash LLM extraction.

Detects table regions on drawing pages using img2table (150 DPI),
crops at 300 DPI for quality, sends each crop to Gemini 2.5 Flash
for structured JSON extraction.

Replaces PyMuPDF find_tables() which only works on native PDF tables
and misses CAD-rendered schedules.
"""
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR
from google.genai import types as genai_types

from app.core.document_classification import DocumentClassificationResult
from app.core.estimate_models import (
    EstimateItem,
    ExtractedScheduleRow,
    ExtractedTable,
    SheetInfo,
)
from google import genai

def _get_genai_client():
    """Get initialized Gemini client (same pattern as sheet_indexer.py)."""
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        try:
            from app.config.settings import settings
            api_key = settings.GOOGLE_API_KEY
        except Exception:
            try:
                from app.config.settings import settings
                api_key = settings.GOOGLE_API_KEY
            except Exception:
                pass
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════

RENDER_DPI = 150   # For img2table bbox detection
CROP_DPI = 300     # For high-quality LLM crops
SCALE = CROP_DPI / RENDER_DPI  # 2.0

# Minimum crop dimensions at CROP_DPI to skip noise
MIN_CROP_W = 100
MIN_CROP_H = 50

EXTRACTION_PROMPT = """You are extracting a construction schedule/table from a drawing page.

Look at this cropped table image and extract ALL data as structured JSON.

Return a JSON object with:
{
  "table_title": "Name of the schedule (e.g. 'Door Schedule', 'Finish Schedule')",
  "schedule_type": "one of: door, window, finish, fixture, equipment, panel, lighting, hardware, ventilation, mechanical, plumbing_fixture, other",
  "headers": ["col1", "col2", ...],
  "rows": [
    {"col1": "val1", "col2": "val2", ...},
    ...
  ]
}

Rules:
- Extract EVERY row, not just examples
- Use exact header names from the table
- If a cell is empty, use ""
- If the image contains multiple sub-tables, return a JSON array of table objects
- For merged cells spanning multiple rows, repeat the value in each row
- Numbers should be strings (preserve original formatting)

Return ONLY valid JSON, no markdown fences."""


# ════════════════════════════════════════════════════════════════════════════════
# Image helpers
# ════════════════════════════════════════════════════════════════════════════════

def _crop_table_region(hd_img: Image.Image, bbox, pad: int = 10) -> Image.Image:
    """Crop a table region from high-DPI image using img2table bbox."""
    x1 = max(0, int(bbox.x1 * SCALE) - pad)
    y1 = max(0, int(bbox.y1 * SCALE) - pad)
    x2 = min(hd_img.width, int(bbox.x2 * SCALE) + pad)
    y2 = min(hd_img.height, int(bbox.y2 * SCALE) + pad)
    return hd_img.crop((x1, y1, x2, y2))


def _image_to_jpeg_bytes(img: Image.Image, max_dim: int = 4000, quality: int = 90) -> bytes:
    """Convert PIL Image to JPEG bytes, downscaling if too large."""
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════════
# LLM extraction
# ════════════════════════════════════════════════════════════════════════════════

def _extract_table_with_llm(client, img_bytes: bytes) -> str:
    """Send table crop to Gemini Flash and get structured JSON back."""
    contents = [
        EXTRACTION_PROMPT,
        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=16384,
        ),
    )
    return response.text or ""


def _parse_llm_response(raw: str):
    """Parse LLM JSON response, handling markdown fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return {"error": str(e), "raw": raw[:500]}


def _llm_result_to_tables(
    parsed,
    page_number: int,
    sheet_id: str,
) -> List[ExtractedTable]:
    """Convert parsed LLM response into ExtractedTable objects."""
    tables: List[ExtractedTable] = []

    # Normalize to list of table dicts
    if isinstance(parsed, dict) and "error" not in parsed:
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return tables

    for item in items:
        if not isinstance(item, dict):
            continue
        headers = item.get("headers", [])
        rows = item.get("rows", [])
        schedule_type = item.get("schedule_type", "other")

        # Skip empty tables
        if not rows:
            continue

        # Filter out empty rows (all values blank)
        clean_rows = []
        for row in rows:
            if isinstance(row, dict) and any(
                str(v).strip() for v in row.values()
            ):
                clean_rows.append(row)

        if not clean_rows:
            continue

        tables.append(ExtractedTable(
            page_number=page_number,
            sheet_id=sheet_id,
            schedule_type=schedule_type,
            headers=headers,
            rows=clean_rows,
            confidence=0.90,
        ))

    return tables


# ════════════════════════════════════════════════════════════════════════════════
# Main extraction
# ════════════════════════════════════════════════════════════════════════════════

def extract_tables_from_sheets(
    sheets: List[SheetInfo],
    classification: DocumentClassificationResult,
) -> List[ExtractedTable]:
    """Extract tables from all drawing pages using img2table + Gemini Flash.

    For each page:
      1. Render at 150 DPI → img2table detects table bounding boxes
      2. Render at 300 DPI → crop each bbox region
      3. Send crop to Gemini Flash → structured JSON extraction
      4. Parse response → ExtractedTable objects

    Args:
        sheets: SheetInfo objects from sheet_indexer
        classification: Document classification (for raw PDF bytes)

    Returns:
        List of ExtractedTable objects found across all pages.
        Also populates each SheetInfo.tables in-place.
    """
    all_tables: List[ExtractedTable] = []
    client = _get_genai_client()
    ocr = TesseractOCR(lang="eng")

    # Group sheets by source file to open each PDF only once
    sheets_by_file: Dict[str, List[SheetInfo]] = {}
    for sheet in sheets:
        sheets_by_file.setdefault(sheet.source_file, []).append(sheet)

    # Build list of (sheet, page_idx, pdf_bytes) for parallel processing
    work_items = []
    for filename, file_sheets in sheets_by_file.items():
        pdf_bytes = classification.raw_pdf_bytes.get(filename)
        if not pdf_bytes:
            logger.warning(f"No PDF bytes for {filename}")
            continue

        # Verify PDF opens and get page count
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            n_pages = len(doc)
            doc.close()
        except Exception as e:
            logger.error(f"Failed to open {filename}: {e}")
            continue

        for sheet in file_sheets:
            page_idx = _find_page_index(sheet, classification)
            if page_idx is not None and page_idx < n_pages:
                work_items.append((sheet, page_idx, pdf_bytes))

    def _process_one_page(args):
        sheet, page_idx, pdf_bytes = args
        # Each thread opens its own doc (PyMuPDF is not thread-safe)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_idx]
        page_ocr = TesseractOCR(lang="eng")
        page_tables = _extract_tables_from_page(
            page, page_idx, sheet, client, page_ocr
        )
        doc.close()
        return sheet, page_tables

    MAX_TABLE_WORKERS = 10
    with ThreadPoolExecutor(max_workers=MAX_TABLE_WORKERS) as executor:
        futures = {executor.submit(_process_one_page, w): w for w in work_items}
        for future in as_completed(futures):
            try:
                sheet, page_tables = future.result()
                for t in page_tables:
                    all_tables.append(t)
                    sheet.tables.append(t)
            except Exception as e:
                w = futures[future]
                logger.error(f"Table extraction failed for page {w[1]}: {e}")

    logger.info(f"Extracted {len(all_tables)} tables from {len(sheets)} drawing pages")

    # Log schedule type breakdown
    type_counts: Dict[str, int] = {}
    for t in all_tables:
        type_counts[t.schedule_type] = type_counts.get(t.schedule_type, 0) + 1
    for stype, count in sorted(type_counts.items()):
        logger.info(f"  {stype}: {count} tables")

    return all_tables


def _extract_tables_from_page(
    page,
    page_idx: int,
    sheet: SheetInfo,
    client,
    ocr: TesseractOCR,
) -> List[ExtractedTable]:
    """Extract all tables from a single page using img2table + LLM."""
    tables: List[ExtractedTable] = []

    # Step 1: Render at 150 DPI for img2table detection
    zoom_detect = RENDER_DPI / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom_detect, zoom_detect))
    img_bytes = pix.tobytes("png")
    page_w_detect = pix.width
    page_h_detect = pix.height
    page_area = page_w_detect * page_h_detect

    tmp_path = os.path.join(tempfile.gettempdir(), f"detect_p{page_idx}_{id(page)}.png")
    try:
        with open(tmp_path, "wb") as f:
            f.write(img_bytes)

        # Step 2: Detect table bounding boxes
        t0 = time.time()
        img_doc = Img2TableImage(src=tmp_path)
        detected = img_doc.extract_tables(ocr=ocr, borderless_tables=True)
        detect_time = time.time() - t0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not detected:
        return tables

    # Step 3: Filter out noise before rendering HD image
    valid_crops = []
    for ti, table in enumerate(detected):
        bbox = table.bbox
        crop_w = int((bbox.x2 - bbox.x1) * SCALE)
        crop_h = int((bbox.y2 - bbox.y1) * SCALE)

        # Skip tiny tables (noise)
        if crop_w < MIN_CROP_W or crop_h < MIN_CROP_H:
            continue

        # Skip full-page detections (>95% of page area = page border, not a table)
        bbox_area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        if bbox_area > page_area * 0.95:
            logger.info(
                f"    Page {page_idx} ({sheet.sheet_id}): skipping full-page region "
                f"{crop_w}x{crop_h}px ({bbox_area/page_area:.0%} of page)"
            )
            continue

        # Skip title block (small region in bottom-right corner)
        if (crop_w < 800 and crop_h < 400
                and bbox.x2 > page_w_detect * 0.85
                and bbox.y2 > page_h_detect * 0.85):
            logger.debug(
                f"    Page {page_idx}: skipping title block region {crop_w}x{crop_h}px"
            )
            continue

        valid_crops.append((ti, table, crop_w, crop_h))

    if not valid_crops:
        return tables

    logger.info(
        f"  Page {page_idx} ({sheet.sheet_id}): "
        f"{len(detected)} detected, {len(valid_crops)} valid regions ({detect_time:.1f}s)"
    )

    # Step 4: Render at 300 DPI for high-quality crops
    zoom_crop = CROP_DPI / 72.0
    pix_hd = page.get_pixmap(matrix=fitz.Matrix(zoom_crop, zoom_crop))
    hd_img = Image.open(io.BytesIO(pix_hd.tobytes("png")))

    # Prepare all crops
    crop_jobs = []
    for ti, table, crop_w, crop_h in valid_crops:
        cropped = _crop_table_region(hd_img, table.bbox)
        jpeg_bytes = _image_to_jpeg_bytes(cropped)
        crop_jobs.append((ti, crop_w, crop_h, jpeg_bytes))

    # Step 5: Send all crops to Gemini Flash in parallel
    def _process_crop(args):
        ti, crop_w, crop_h, jpeg_bytes = args
        t0 = time.time()
        raw_response = _extract_table_with_llm(client, jpeg_bytes)
        llm_time = time.time() - t0
        parsed = _parse_llm_response(raw_response)
        page_tables = _llm_result_to_tables(
            parsed, sheet.global_page_number, sheet.sheet_id
        )
        logger.info(
            f"    Table {ti+1}: {crop_w}x{crop_h}px → "
            f"{len(page_tables)} schedules ({llm_time:.1f}s)"
        )
        return page_tables

    with ThreadPoolExecutor(max_workers=min(len(crop_jobs), 4)) as crop_executor:
        crop_futures = {crop_executor.submit(_process_crop, job): job for job in crop_jobs}
        for future in as_completed(crop_futures):
            try:
                page_tables = future.result()
                tables.extend(page_tables)
            except Exception as e:
                job = crop_futures[future]
                logger.error(f"    Table {job[0]+1}: LLM error: {e}")

    return tables


def _find_page_index(
    sheet: SheetInfo,
    classification: DocumentClassificationResult,
) -> Optional[int]:
    """Find the local page index within the PDF for a SheetInfo."""
    for cf in classification.files:
        if cf.filename == sheet.source_file:
            for page in cf.pages:
                if page.global_page_number == sheet.global_page_number:
                    return page.page_number
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Mark column detection
# ════════════════════════════════════════════════════════════════════════════════

_MARK_HEADERS = {
    "MARK", "NO", "NO.", "NUMBER", "DOOR NO", "DOOR NO.",
    "ROOM", "ROOM NO", "ROOM NO.", "ROOM NUMBER", "SPACE",
    "TYPE", "TAG", "SYMBOL", "SET", "HW SET", "HARDWARE SET",
    "CKT", "CIRCUIT", "FIXTURE",
}


def _find_mark_column(headers: List[str]) -> Optional[int]:
    """Find which column contains the primary identifier (mark)."""
    for i, h in enumerate(headers):
        if h.upper().strip() in _MARK_HEADERS:
            return i
    return 0


# ════════════════════════════════════════════════════════════════════════════════
# Schedule row conversion
# ════════════════════════════════════════════════════════════════════════════════

def tables_to_schedule_rows(tables: List[ExtractedTable]) -> List[ExtractedScheduleRow]:
    """Flatten tables into individual rows for downstream linking."""
    rows: List[ExtractedScheduleRow] = []

    for table in tables:
        if table.schedule_type in ("unknown", "other"):
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

    logger.info(f"Converted {len(rows)} schedule rows from {len(tables)} tables")
    return rows


# ════════════════════════════════════════════════════════════════════════════════
# Schedule rows → EstimateItems
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
    """Convert parsed schedule rows to EstimateItem objects.

    Each schedule row becomes one EstimateItem with:
    - extraction_method = "schedule_parse"
    - confidence = 0.90 (LLM extraction)
    - qty = 1 EA per row (count from schedule)
    """
    items: List[EstimateItem] = []

    for row in schedule_rows:
        trade = _SCHEDULE_TYPE_TO_TRADE.get(row.schedule_type, "General Requirements")

        # Build description from row data
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

        # Build material spec from row data
        material_parts = []
        for key in ("MATERIAL", "FRAME", "FRAME MATERIAL", "GLAZING", "GLASS",
                     "MANUFACTURER", "MFR", "MODEL", "CATALOG"):
            val = row.row_data.get(key, "").strip()
            if val:
                material_parts.append(f"{key}: {val}")

        # Build notes from remarks/notes columns
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
        )
        items.append(item)

    logger.info(f"Created {len(items)} estimate items from schedule rows")
    return items
