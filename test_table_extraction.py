"""Test script: img2table bbox detection → Gemini Flash filter (150 DPI, 5-at-a-time) → Gemini Flash extraction (300 DPI).

Usage: python test_table_extraction.py "24-086 Drawings.pdf"
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
from PIL import Image
from img2table.document import Image as Img2TableImage
from google import genai
from google.genai import types as genai_types

# ── Config ──────────────────────────────────────────────────────────────────
DETECT_DPI = 150
CROP_DPI = 300
FILTER_BATCH = 5
EXTRACT_BATCH = 5
WORKERS = 4

MIN_CROP_W = 80
MIN_CROP_H = 40


# ── Client ──────────────────────────────────────────────────────────────────
def _get_client():
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY env var")
    return genai.Client(api_key=api_key)


# ── Phase 1: img2table detection ────────────────────────────────────────────
def detect_bboxes(pdf_bytes: bytes, page_idx: int) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = DETECT_DPI / 72.0
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    page_w, page_h = pix.width, pix.height
    page_area = page_w * page_h

    tmp_path = os.path.join(tempfile.gettempdir(), f"det_{page_idx}_{os.getpid()}.png")
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
        if w < MIN_CROP_W or h < MIN_CROP_H:
            continue
        if w * h > page_area * 0.90:
            continue
        # Skip title block (small region bottom-right)
        if (w < 600 and h < 300
                and bbox.x2 > page_w * 0.85
                and bbox.y2 > page_h * 0.85):
            continue
        results.append({
            "page": page_idx,
            "bbox": (bbox.x1, bbox.y1, bbox.x2, bbox.y2),
            "page_w": page_w,
            "page_h": page_h,
        })
    return results


# ── Crop helper ─────────────────────────────────────────────────────────────
def crop_from_page(pdf_bytes: bytes, page_idx: int, bbox: tuple, dpi: int) -> bytes:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    pix = doc[page_idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()

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


# ── Phase 2: Gemini Flash filter (150 DPI, 5-at-a-time) ────────────────────
FILTER_PROMPT = """\
You are filtering cropped regions from construction drawing pages.

For EACH crop image below, classify it as ONE of:
- "sheet_index" — a drawing index / sheet list
- "table" — a schedule or data table useful for construction estimation
  (door schedule, finish schedule, panel schedule, fixture schedule, etc.)
- "not_table" — a plan view, section, detail, note block, title block,
  legend, or anything that is NOT a structured data table

IMPORTANT: Do NOT delete or discard any real table or schedule.
When in doubt, classify as "table". Only mark as "not_table" if it
is clearly NOT a data table.

Return JSON array:
[
  {"crop_id": 0, "type": "table"},
  {"crop_id": 1, "type": "not_table"},
  {"crop_id": 2, "type": "sheet_index"}
]

Return ONLY valid JSON, no markdown fences."""


def filter_batch(client, batch: list[tuple[int, bytes]]) -> list[tuple[int, str]]:
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
                return [(idx, "table") for idx, _ in batch]
        else:
            return [(idx, "table") for idx, _ in batch]

    out = []
    for r in results:
        if isinstance(r, dict):
            out.append((r.get("crop_id", -1), r.get("type", "table")))
    return out


# ── Phase 3: Gemini Flash extraction (300 DPI) ─────────────────────────────
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


def extract_batch(client, batch: list[tuple[int, bytes]]) -> list[tuple[int, dict]]:
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

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    results = []
    for i, item in enumerate(parsed):
        if isinstance(item, dict):
            idx = batch[min(i, len(batch) - 1)][0]
            results.append((idx, item))
    return results


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
    doc.close()
    print(f"PDF: {pdf_path} — {num_pages} pages")

    # ── Phase 1: img2table bbox detection ────────────────────────────────
    print(f"\n=== Phase 1: img2table detection at {DETECT_DPI} DPI ===")
    t0 = time.time()

    all_detections = []

    def _detect(page_idx):
        return detect_bboxes(pdf_bytes, page_idx)

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_detect, i): i for i in range(num_pages)}
        for future in as_completed(futures):
            page_idx = futures[future]
            try:
                dets = future.result()
                if dets:
                    print(f"  Page {page_idx}: {len(dets)} regions")
                all_detections.extend(dets)
            except Exception as e:
                print(f"  Page {page_idx} ERROR: {e}")

    elapsed = time.time() - t0
    pages_with = len(set(d["page"] for d in all_detections))
    print(f"Phase 1 done: {len(all_detections)} candidates on {pages_with} pages ({elapsed:.1f}s)")

    if not all_detections:
        print("No tables detected!")
        return

    # ── Phase 2: Gemini Flash filter at 150 DPI ─────────────────────────
    print(f"\n=== Phase 2: Filter {len(all_detections)} crops with Gemini Flash (150 DPI, batches of {FILTER_BATCH}) ===")
    t0 = time.time()
    client = _get_client()

    # Render 150 DPI crops
    filter_crops = []
    for i, det in enumerate(all_detections):
        crop = crop_from_page(pdf_bytes, det["page"], det["bbox"], DETECT_DPI)
        filter_crops.append((i, crop))

    # Batch and filter in parallel
    batches = [filter_crops[i:i + FILTER_BATCH] for i in range(0, len(filter_crops), FILTER_BATCH)]
    crop_types = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(filter_batch, client, b): bi for bi, b in enumerate(batches)}
        for future in as_completed(futures):
            bi = futures[future]
            try:
                results = future.result()
                for crop_idx, crop_type in results:
                    crop_types[crop_idx] = crop_type
                types_str = ", ".join(f"{idx}={t}" for idx, t in results)
                print(f"  Batch {bi}: {types_str}")
            except Exception as e:
                print(f"  Batch {bi} ERROR: {e}")
                for idx, _ in batches[bi]:
                    crop_types[idx] = "table"

    # Tally
    kept_indices = [i for i in range(len(all_detections))
                    if crop_types.get(i, "table") in ("table", "sheet_index")]
    discarded = len(all_detections) - len(kept_indices)
    elapsed = time.time() - t0
    print(f"Phase 2 done: {len(kept_indices)} kept, {discarded} filtered out ({elapsed:.1f}s)")

    for i in range(len(all_detections)):
        ctype = crop_types.get(i, "?")
        det = all_detections[i]
        print(f"  crop {i}: page={det['page']}, type={ctype}, bbox={det['bbox']}")

    # ── Phase 3: Gemini Flash extraction at 300 DPI ─────────────────────
    kept_detections = [(i, all_detections[i], crop_types.get(i, "table")) for i in kept_indices]
    print(f"\n=== Phase 3: Extract {len(kept_detections)} tables at {CROP_DPI} DPI ===")
    t0 = time.time()

    # Render 300 DPI crops
    extract_crops = []
    for crop_idx, det, ctype in kept_detections:
        crop = crop_from_page(pdf_bytes, det["page"], det["bbox"], CROP_DPI)
        extract_crops.append((crop_idx, crop, det["page"], ctype))

    # Batch and extract in parallel
    ext_batches = [extract_crops[i:i + EXTRACT_BATCH] for i in range(0, len(extract_crops), EXTRACT_BATCH)]

    all_results = []

    def _run_extract(bi, batch):
        batch_for_api = [(idx, crop_bytes) for idx, crop_bytes, _, _ in batch]
        results = extract_batch(client, batch_for_api)
        return [(batch[min(i, len(batch)-1)][2], batch[min(i, len(batch)-1)][3], parsed)
                for i, (_, parsed) in enumerate(results)]

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_run_extract, i, b): i for i, b in enumerate(ext_batches)}
        for future in as_completed(futures):
            bi = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                for page, ctype, parsed in results:
                    title = parsed.get("table_title", "?")
                    n_rows = len(parsed.get("rows", []))
                    print(f"  Batch {bi}: page={page}, type={ctype}, title={title!r}, rows={n_rows}")
            except Exception as e:
                print(f"  Batch {bi} ERROR: {e}")

    elapsed = time.time() - t0
    print(f"Phase 3 done: {len(all_results)} tables extracted ({elapsed:.1f}s)")

    # ── Output ──────────────────────────────────────────────────────────
    output_path = "test_extraction_results.json"
    output = []
    for page, ctype, parsed in all_results:
        output.append({
            "page": page,
            "filter_type": ctype,
            "table_title": parsed.get("table_title", ""),
            "schedule_type": parsed.get("schedule_type", ""),
            "headers": parsed.get("headers", []),
            "rows": parsed.get("rows", []),
        })

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"\nSummary:")
    for item in output:
        print(f"  Page {item['page']}: {item['table_title']!r} ({item['schedule_type']}) — {len(item['rows'])} rows")


if __name__ == "__main__":
    main()
