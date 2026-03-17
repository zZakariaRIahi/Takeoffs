"""Vision Quantifier — Step 3 of the extraction pipeline.

Takes items from Step 2d that have needs_counting=True or needs_measurement=True,
groups them by TRADE + PLAN PAGE, and sends focused Gemini 2.5 Pro vision calls
for each (trade, page) batch.

Grouping by trade keeps each prompt small and focused — the model knows exactly
which discipline's symbols and dimensions to look for.

Input: List[EstimateItem] (from Step 2d) with counting/measurement flags
Output: Same items with qty filled in where vision succeeded
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from google import genai
from google.genai import types as genai_types

from app.core.document_classification import (
    DocumentCategory,
    DocumentClassificationResult,
)
from app.core.estimate_models import EstimateItem, SheetInfo

logger = logging.getLogger(__name__)

RENDER_DPI = 300
MAX_PARALLEL = 5
MODEL = "gemini-2.5-pro"

# ════════════════════════════════════════════════════════════════════════════════
# Gemini client
# ════════════════════════════════════════════════════════════════════════════════

def _get_genai_client() -> genai.Client:
    """Get Gemini client (same pattern as drawing_reader)."""
    api_key: Optional[str] = None
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
        api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured")
    return genai.Client(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════════
# Page rendering (with cache)
# ════════════════════════════════════════════════════════════════════════════════

def _render_page_jpeg(pdf_bytes: bytes, page_idx: int, dpi: int = RENDER_DPI) -> bytes:
    """Render a PDF page to JPEG bytes at the given DPI."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_idx >= len(doc):
        doc.close()
        raise ValueError(f"Page {page_idx} out of range (PDF has {len(doc)} pages)")

    page = doc[page_idx]
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    png_bytes = pix.tobytes("png")
    doc.close()

    img = Image.open(io.BytesIO(png_bytes))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════════
# Prompt builder — trade-focused per page
# ════════════════════════════════════════════════════════════════════════════════

def _build_trade_page_prompt(
    trade: str,
    sheet: SheetInfo,
    items: List[Tuple[int, EstimateItem]],
) -> str:
    """Build a focused prompt for one trade on one plan page.

    Each item shows: Description, Unit, Material, Plan reference, What to do.
    """
    lines = [
        "You are a SENIOR CONSTRUCTION QUANTITY SURVEYOR performing a quantity takeoff.",
        f"You are looking at this drawing page to quantify **{trade}** items ONLY.",
        "",
        f"TRADE: {trade}",
        f"PLAN: Page {sheet.global_page_number} — {sheet.sheet_id} — {sheet.title}",
        f"DISCIPLINE: {sheet.discipline}",
        "SCALE: Read the graphic scale or title block from this page.",
        "",
        "══════════════════════════════════════════",
        f"  {trade.upper()} — ITEMS TO QUANTIFY",
        "══════════════════════════════════════════",
        "",
    ]

    for global_idx, item in items:
        mark = item.schedule_mark or ""
        material = item.material_spec or ""

        if item.needs_counting:
            task_type = "COUNT"
            instruction = item.counting_target or "Count all matching symbols on this page."
        else:
            task_type = "MEASURE"
            instruction = item.review_reason or "Measure from dimensions on this page."

        lines.append(f"  [{global_idx}] {item.item_description}")
        lines.append(f"       Task: {task_type}")
        lines.append(f"       Unit: {item.unit}")
        if mark:
            lines.append(f"       Mark: {mark}")
        if material:
            lines.append(f"       Material: {material}")
        lines.append(f"       Instruction: {instruction}")
        lines.append("")

    lines.extend([
        "══════════════════════════════════════════",
        "HOW TO QUANTIFY:",
        "══════════════════════════════════════════",
        "",
        "FOR COUNT TASKS:",
        "  1. Divide the drawing into 4 quadrants (NW, NE, SW, SE).",
        "  2. Scan each quadrant — count every matching symbol/mark.",
        "  3. Include symbols in enlarged areas, details, and insets.",
        "  4. Report: \"NW: 3, NE: 4, SW: 2, SE: 3 = 12 total\"",
        "",
        "FOR MEASURE TASKS:",
        "  1. Read dimension strings on the drawing (e.g., 15'-0\", 22'-6\").",
        "  2. AREA (SF): find room dimensions → length × width, sum all rooms.",
        "  3. LINEAR (LF): read wall/run dimension callouts, sum total length.",
        "  4. If exact dimensions aren't written, ESTIMATE using the drawing scale.",
        "  5. Show your math: \"Room 101: 15' × 12'6\" = 187.5 SF; Room 103: 20' × 13' = 260 SF\"",
        "",
        "IMPORTANT:",
        "  - TRY YOUR BEST ESTIMATE. An approximate number is far better than null.",
        "  - Only return qty=null when you truly CANNOT determine any reasonable quantity.",
        "  - For lump-sum items (LS) → return qty=1, confidence=\"medium\".",
        "",
        "CONFIDENCE:",
        "  \"high\"   — clearly readable dimensions or easily countable symbols",
        "  \"medium\" — estimated from scale, partially visible, or dense drawing",
        "  \"low\"    — cannot determine accurately → left for MANUAL REVIEW",
        "",
        "Return JSON:",
        "{",
        "  \"page_scale\": \"1/4\\\" = 1'-0\\\"\" or null,",
        "  \"results\": [",
        "    {",
        "      \"item_id\": 0,",
        "      \"qty\": 12,",
        "      \"confidence\": \"high\",",
        "      \"method\": \"NW: 3, NE: 4, SW: 2, SE: 3 = 12 total\",",
        "      \"notes\": \"\"",
        "    }",
        "  ]",
        "}",
        "",
        "item_id = the number in square brackets [N] above each item.",
        "Return ONLY valid JSON. No markdown fences.",
    ])

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Gemini vision call
# ════════════════════════════════════════════════════════════════════════════════

CALL_TIMEOUT = 200  # seconds per Gemini call
MAX_CALL_RETRIES = 2  # retry once on timeout


def _call_vision(
    client: genai.Client,
    prompt: str,
    image_bytes: bytes,
) -> Dict[str, Any]:
    """Send a single page image + prompt to Gemini 2.5 Pro and parse JSON response.

    Retries once on timeout (200s default).
    """
    import concurrent.futures

    contents = [
        prompt,
        genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
    ]

    config = genai_types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=16384,
        thinking_config=genai_types.ThinkingConfig(
            thinking_budget=16384,
        ),
    )

    for attempt in range(MAX_CALL_RETRIES):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(
                    client.models.generate_content,
                    model=MODEL,
                    contents=contents,
                    config=config,
                )
                response = future.result(timeout=CALL_TIMEOUT)

            # Extract text (skip thinking parts)
            raw = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text and (not hasattr(part, "thought") or not part.thought):
                        raw += part.text

            if not raw:
                raw = response.text or "{}"

            return _parse_response(raw)

        except concurrent.futures.TimeoutError:
            logger.warning(
                f"  Gemini call timed out after {CALL_TIMEOUT}s "
                f"(attempt {attempt + 1}/{MAX_CALL_RETRIES})"
            )
            if attempt < MAX_CALL_RETRIES - 1:
                continue
            logger.error("  Gemini call failed after all retries — returning empty")
            return {"results": []}
        except Exception as e:
            logger.error(f"  Gemini call error (attempt {attempt + 1}): {e}")
            if attempt < MAX_CALL_RETRIES - 1:
                continue
            return {"results": []}


def _parse_response(raw: str) -> Dict[str, Any]:
    """Parse JSON response from Gemini."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"Could not parse vision response: {raw[:500]}")
        return {"results": []}


# ════════════════════════════════════════════════════════════════════════════════
# Process one (trade, page) batch
# ════════════════════════════════════════════════════════════════════════════════

def _process_trade_page(
    client: genai.Client,
    image_bytes: bytes,
    page_idx: int,
    sheet: SheetInfo,
    trade: str,
    items: List[Tuple[int, EstimateItem]],
) -> Dict[int, Dict[str, Any]]:
    """Process one trade on one plan page: build prompt, call Gemini, return results."""
    t0 = time.time()

    logger.info(
        f"  [{trade}] Page {page_idx} ({sheet.sheet_id}): {len(items)} items"
    )

    prompt = _build_trade_page_prompt(trade, sheet, items)

    t1 = time.time()
    parsed = _call_vision(client, prompt, image_bytes)
    call_time = time.time() - t1

    logger.info(
        f"    [{trade}] Page {page_idx}: "
        f"Gemini [{call_time:.1f}s], scale={parsed.get('page_scale', '?')}"
    )

    # Map results back to global item indices
    results: Dict[int, Dict[str, Any]] = {}
    for r in parsed.get("results", []):
        item_id = r.get("item_id")
        if item_id is None:
            continue
        results[int(item_id)] = {
            "qty": r.get("qty"),
            "confidence": str(r.get("confidence", "low")),
            "method": str(r.get("method", "")),
            "notes": str(r.get("notes", "")),
        }

    logger.info(f"    [{trade}] Page {page_idx}: {len(results)}/{len(items)} results")
    return results


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

def quantify_items(
    items: List[EstimateItem],
    classification: DocumentClassificationResult,
    sheets: List[SheetInfo],
) -> List[EstimateItem]:
    """Fill in qty for items needing counting or measurement via Gemini 2.5 Pro.

    Groups items by (trade, plan_page) so each vision call is focused on one
    trade's items on one page. Runs up to 5 calls in parallel.

    Args:
        items: EstimateItems from Step 2d (some with needs_counting/needs_measurement)
        classification: For raw PDF bytes
        sheets: SheetInfo list from Step 1 (for page metadata)

    Returns:
        Same items list with qty filled in where vision succeeded.
        Items with confidence="low" keep qty=null for manual review.
    """
    if not items:
        return items

    # ── 1. Find items needing quantification ──────────────────────────────
    needs_work: List[Tuple[int, EstimateItem]] = []
    for i, item in enumerate(items):
        if item.needs_counting or item.needs_measurement:
            needs_work.append((i, item))

    if not needs_work:
        logger.info("Vision quantifier: no items need counting or measurement")
        return items

    logger.info(
        f"Vision quantifier: {len(needs_work)} items need quantification "
        f"({sum(1 for _, i in needs_work if i.needs_counting)} counting, "
        f"{sum(1 for _, i in needs_work if i.needs_measurement)} measurement)"
    )

    # ── 2. Build sheet lookup ─────────────────────────────────────────────
    sheet_by_page: Dict[int, SheetInfo] = {}
    for s in sheets:
        sheet_by_page[s.global_page_number] = s

    # ── 3. Group items by (trade, page) ───────────────────────────────────
    # Key: (trade, page_num) → List[(global_idx, item)]
    trade_page_groups: Dict[Tuple[str, int], List[Tuple[int, EstimateItem]]] = defaultdict(list)

    for idx, item in needs_work:
        target_pages = item.counting_source_pages if item.counting_source_pages else []
        if not target_pages:
            logger.debug(f"  Item {idx} ({item.item_description[:40]}) has no plan pages — skipping")
            continue

        for pg in target_pages:
            if pg not in sheet_by_page:
                continue
            trade_page_groups[(item.trade, pg)].append((idx, item))

    if not trade_page_groups:
        logger.warning("Vision quantifier: no valid (trade, page) groups found")
        return items

    # Log grouping summary
    unique_pages = sorted({pg for _, pg in trade_page_groups})
    unique_trades = sorted({t for t, _ in trade_page_groups})
    logger.info(
        f"  {len(trade_page_groups)} batches across "
        f"{len(unique_pages)} pages and {len(unique_trades)} trades"
    )
    for (trade, pg), batch_items in sorted(trade_page_groups.items()):
        sheet = sheet_by_page[pg]
        logger.info(f"    {trade:25s} | Page {pg:2d} ({sheet.sheet_id}) | {len(batch_items)} items")

    # ── 4. Find PDF bytes ─────────────────────────────────────────────────
    drawing_files = classification.get_files_by_category(
        DocumentCategory.CONSTRUCTION_DRAWINGS
    )
    if not drawing_files:
        logger.warning("No drawing files found")
        return items

    pdf_bytes = None
    for cf in drawing_files:
        if cf.filename in classification.raw_pdf_bytes:
            pdf_bytes = classification.raw_pdf_bytes[cf.filename]
            break

    if not pdf_bytes:
        logger.warning("No raw PDF bytes found for drawing files")
        return items

    # ── 5. Pre-render pages (each page rendered once, shared across trades)
    logger.info("  Rendering pages...")
    t0 = time.time()
    page_images: Dict[int, bytes] = {}
    for pg in unique_pages:
        page_images[pg] = _render_page_jpeg(pdf_bytes, pg)
        logger.info(f"    Page {pg}: {len(page_images[pg]) / 1024:.0f} KB")
    logger.info(f"  Rendered {len(page_images)} pages [{time.time() - t0:.1f}s]")

    # ── 6. Process (trade, page) batches in parallel ──────────────────────
    client = _get_genai_client()
    t0 = time.time()

    # Accumulate results per item across pages
    item_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    def _process(key: Tuple[str, int]):
        trade, pg = key
        batch_items = trade_page_groups[key]
        sheet = sheet_by_page[pg]
        image_bytes = page_images[pg]
        return _process_trade_page(client, image_bytes, pg, sheet, trade, batch_items)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(_process, key): key for key in trade_page_groups}
        for future in as_completed(futures):
            key = futures[future]
            try:
                batch_results = future.result()
                for item_idx, result in batch_results.items():
                    item_results[item_idx].append(result)
            except Exception as e:
                logger.error(f"  Batch {key} failed: {e}")

    total_time = time.time() - t0
    logger.info(f"  Vision calls complete [{total_time:.1f}s]")

    # ── 7. Merge results back into items ──────────────────────────────────
    confidence_map = {"high": 0.85, "medium": 0.65, "low": 0.45}
    updated = 0
    kept_null = 0

    for item_idx, page_results in item_results.items():
        item = items[item_idx]

        # Filter out low-confidence results but keep non-low ones
        good_results = [r for r in page_results if r["confidence"] != "low"]

        if not good_results:
            # ALL pages returned low — keep for manual review
            kept_null += 1
            item.review_reason = "Vision confidence too low — manual takeoff needed"
            continue

        # Sum qty across pages (multi-page items)
        total_qty = 0.0
        all_methods = []
        worst_confidence = "high"

        for r in good_results:
            if r["qty"] is not None:
                total_qty += float(r["qty"])
                all_methods.append(r["method"])
                if r["confidence"] == "medium" and worst_confidence == "high":
                    worst_confidence = "medium"
            else:
                # A non-low page returned null qty — skip this result
                continue

        if total_qty > 0:
            item.qty = total_qty
            item.confidence = confidence_map.get(worst_confidence, 0.65)
            item.extraction_method = "vision_count" if item.needs_counting else "vision_measurement"
            item.notes = " | ".join(all_methods) if all_methods else item.notes
            updated += 1
        else:
            kept_null += 1

    logger.info(
        f"  Vision quantifier: {updated} items updated with qty, "
        f"{kept_null} kept as null (low confidence or no data)"
    )

    return items
