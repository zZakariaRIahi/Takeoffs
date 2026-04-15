"""Test: Pure Gemini Flash table extraction (no img2table).

Phase 1: Send all page images to Flash in batches of 5 → scan for tables
Phase 2: Re-render pages with tables at 300 DPI → extract table contents

Usage: python test_flash_extraction.py "24-086 Drawings.pdf"
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from google import genai
from google.genai import types as genai_types

# ── Config ──────────────────────────────────────────────────────────────────
SCAN_BATCH = 5       # pages per Flash call in Phase 1
EXTRACT_DPI = 300    # DPI for Phase 2 extraction
WORKERS = 4          # parallel Gemini calls


def _get_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY env var")
    return genai.Client(api_key=api_key)


# ── Phase 1: Scan pages for tables ──────────────────────────────────────────

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


def _scan_batch(client, batch: list[tuple[int, bytes]]) -> list[dict]:
    """Send a batch of page images to Flash to find tables."""
    contents = [SCAN_PROMPT]
    for page_idx, img_bytes in batch:
        contents.append(f"--- page_index={page_idx} ---")
        contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=4096),
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
                # Assume all pages have tables on parse failure
                return [{"page_index": idx, "has_tables": True, "tables": []} for idx, _ in batch]
        else:
            return [{"page_index": idx, "has_tables": True, "tables": []} for idx, _ in batch]

    return results


# ── Phase 2: Extract tables at high DPI ─────────────────────────────────────

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


def _extract_page(client, page_idx: int, img_bytes: bytes) -> list[dict]:
    """Send a high-DPI page image to Flash for table extraction."""
    contents = [
        EXTRACT_PROMPT,
        f"--- Page {page_idx} ---",
        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
    ]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=genai_types.GenerateContentConfig(temperature=0, max_output_tokens=32768),
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


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "24-086 Drawings.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    num_pages = len(doc)
    print(f"PDF: {pdf_path} — {num_pages} pages")

    # ── Render all pages at 150 DPI for scanning ────────────────────────
    print(f"\nRendering {num_pages} pages at 150 DPI...")
    t0 = time.time()
    page_images = {}  # page_idx → png_bytes
    for i in range(num_pages):
        zoom = 150 / 72.0
        pix = doc[i].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        page_images[i] = pix.tobytes("png")
    print(f"Rendered in {time.time() - t0:.1f}s")

    # ── Phase 1: Scan all pages for tables ──────────────────────────────
    print(f"\n=== Phase 1: Scan {num_pages} pages with Gemini Flash (batches of {SCAN_BATCH}) ===")
    t0 = time.time()
    client = _get_client()

    # Build batches
    all_pages = [(i, page_images[i]) for i in range(num_pages)]
    batches = [all_pages[i:i + SCAN_BATCH] for i in range(0, len(all_pages), SCAN_BATCH)]

    pages_with_tables = {}  # page_idx → list of table info dicts

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
                        print(f"  Page {pidx}: {', '.join(titles)}")
            except Exception as e:
                print(f"  Batch {bi} ERROR: {e}")

    elapsed = time.time() - t0
    print(f"Phase 1 done: {len(pages_with_tables)} pages have tables ({elapsed:.1f}s)")

    for pidx in sorted(pages_with_tables):
        tables = pages_with_tables[pidx]
        for t in tables:
            print(f"  Page {pidx}: {t.get('title', '?')} ({t.get('type', '?')})")

    if not pages_with_tables:
        print("No tables found!")
        doc.close()
        return

    # ── Phase 2: Extract tables at 300 DPI ──────────────────────────────
    print(f"\n=== Phase 2: Extract tables from {len(pages_with_tables)} pages at {EXTRACT_DPI} DPI ===")
    t0 = time.time()

    # Render only table pages at high DPI
    hires_images = {}
    for pidx in pages_with_tables:
        zoom = EXTRACT_DPI / 72.0
        pix = doc[pidx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        hires_images[pidx] = pix.tobytes("png")
    print(f"  Rendered {len(hires_images)} pages at {EXTRACT_DPI} DPI")

    all_results = []  # (page_idx, parsed_table_dict)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_extract_page, client, pidx, img): pidx
            for pidx, img in hires_images.items()
        }
        for future in as_completed(futures):
            pidx = futures[future]
            try:
                tables = future.result()
                for t in tables:
                    title = t.get("table_title", "?")
                    n_rows = len(t.get("rows", []))
                    print(f"  Page {pidx}: {title!r} — {n_rows} rows")
                    all_results.append((pidx, t))
            except Exception as e:
                print(f"  Page {pidx} ERROR: {e}")

    elapsed = time.time() - t0
    print(f"Phase 2 done: {len(all_results)} tables extracted ({elapsed:.1f}s)")

    doc.close()

    # ── Output ──────────────────────────────────────────────────────────
    output_path = "test_flash_results.json"
    output = []
    for pidx, t in sorted(all_results, key=lambda x: x[0]):
        output.append({
            "page": pidx,
            "table_title": t.get("table_title", ""),
            "schedule_type": t.get("schedule_type", ""),
            "headers": t.get("headers", []),
            "rows": t.get("rows", []),
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nSummary:")
    total_rows = 0
    for item in output:
        n = len(item["rows"])
        total_rows += n
        print(f"  Page {item['page']}: {item['table_title']!r} ({item['schedule_type']}) — {n} rows")
    print(f"\nTotal: {len(output)} tables, {total_rows} rows")


if __name__ == "__main__":
    main()
