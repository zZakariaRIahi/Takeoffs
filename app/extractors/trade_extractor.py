"""Step 3 — Per-Discipline Plan Reading & Quantification.

Receives discipline packages from Step 2 (schedules, keynotes, symbols already extracted).
Focuses ONLY on visual plan work:
  - Reading dimensions from plans
  - Counting symbols on plans
  - Validating schedule quantities
  - Reporting detailed extraction methods

Does NOT re-read schedules, keynotes, or text — all of that comes from Step 2.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types as genai_types

from app.core.document_classification import DocumentClassificationResult
from app.core.estimate_models import EstimateItem, ExtractedTable, SheetInfo
from app.extractors.context_extractor import DisciplinePackage

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-pro"
MAX_WORKERS = 10
MAX_RETRIES = 2


# ════════════════════════════════════════════════════════════════════════════════
# Client
# ════════════════════════════════════════════════════════════════════════════════

def _get_client() -> genai.Client:
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
# Prompt builder
# ════════════════════════════════════════════════════════════════════════════════

def _build_prompt(
    discipline: str,
    package: DisciplinePackage,
    project_context: str,
) -> Tuple[str, set]:
    """Build a focused plan-reading prompt for one discipline."""

    # ── Page info with views, plans, and focused instructions ──
    page_lines = []
    skip_pages = set()  # pages with only schedules/notes — don't send to Pro
    for pi in sorted(package.page_info, key=lambda x: x.get("page", 0)):
        pg = pi.get("page", -1)
        sid = pi.get("sheet_id", f"PAGE-{pg}")
        plans = pi.get("plans", [])
        views = pi.get("views", [])
        step3 = pi.get("step3_instruction", "")

        # Skip pages that are only schedules/notes (already extracted)
        if step3 and "SKIP" in step3.upper():
            skip_pages.add(pg)
            continue

        line = f"  Page {pg} ({sid})"
        if views:
            line += f"\n    Views: {', '.join(views)}"
        if plans:
            line += f"\n    Plans: {', '.join(plans)}"
        if step3:
            line += f"\n    TASK: {step3}"
        elif plans:
            line += f"\n    TASK: Extract all countable and measurable items from the views on this page."
        page_lines.append(line)
    pages_text = "\n".join(page_lines) or "  (no pages)"

    # ── Schedules already extracted (Step 2) ──
    sched_lines = []
    for s in package.schedules:
        sched_lines.append(f"\n  === {s.get('title', 'Untitled')} ({s.get('type', 'other')}) — Page {s.get('page', '?')} ===")
        headers = s.get("headers", [])
        rows = s.get("rows", [])
        if headers:
            sched_lines.append(f"  Headers: {' | '.join(headers)}")
        for row in rows[:50]:
            vals = [str(row.get(h, "")) for h in headers]
            sched_lines.append(f"  {' | '.join(vals)}")
        if len(rows) > 50:
            sched_lines.append(f"  ... ({len(rows) - 50} more rows)")
    schedule_text = "\n".join(sched_lines) or "  (no schedules for this discipline)"

    # ── Keynotes already extracted (Step 2) ──
    keynote_lines = []
    for k in package.keynotes:
        keynote_lines.append(f"  {k.get('key', '?')}: {k.get('text', '')} (page {k.get('page', '?')})")
    keynote_text = "\n".join(keynote_lines) or "  (no keynotes)"

    # ── Symbols already extracted (Step 2) ──
    symbol_lines = []
    for sym in package.symbols:
        symbol_lines.append(f"  {sym.get('symbol', '?')}: {sym.get('description', '')} [{sym.get('category', '')}]")
    symbol_text = "\n".join(symbol_lines) or "  (no symbols)"

    # ── Discipline context from Step 2 ──
    disc_context = package.context or "No additional context."

    # Map disciplines to CSI trades that appear on their pages
    DISCIPLINE_TRADES = {
        "Architectural": [
            "Concrete", "Masonry", "Metals", "Rough Carpentry", "Finish Carpentry",
            "Waterproofing", "Insulation", "Roofing", "Stucco and Siding",
            "Doors and Windows", "Drywall", "Tile & Solid Surfaces", "Flooring",
            "Painting", "Bath and Accessories", "Cabinets", "Fire Sprinklers",
        ],
        "Civil": ["Site Work", "Concrete", "Landscaping"],
        "Structural": ["Concrete", "Masonry", "Metals", "Rough Carpentry"],
        "Plumbing": ["Plumbing", "Fire Sprinklers"],
        "Mechanical": ["HVAC and Sheet Metals", "Insulation"],
        "Electrical": ["Electrical"],
        "Abatement": ["General Requirements"],
        "Fire Protection": ["Fire Sprinklers"],
        "Landscape": ["Landscaping", "Site Work"],
    }

    trades_for_disc = DISCIPLINE_TRADES.get(discipline, [discipline])
    trades_text = ", ".join(trades_for_disc)

    prompt = f"""\
You are a SENIOR CONSTRUCTION ESTIMATOR performing quantity takeoff from **{discipline}** drawing pages.

These pages contain items for MULTIPLE construction trades. Extract items for ALL of these trades:
  {trades_text}

For example, an Architectural floor plan contains: doors (Doors and Windows), partitions (Drywall),
floor finishes (Flooring), wall paint (Painting), ceiling tile (Drywall), rubber base (Flooring),
toilet accessories (Bath and Accessories), insulation in walls (Insulation), etc.
An estimator breaks these into SEPARATE line items by trade — you must do the same.

All schedules, keynotes, and symbol definitions have ALREADY been extracted and are provided below.
DO NOT re-read or re-extract schedule data from the images — trust the extracted data exactly.

Each page below has a TASK instruction that tells you EXACTLY what to look for on that page.
Follow the TASK instructions precisely — they were written by an estimator who already reviewed
the drawings and identified what needs to be counted, measured, or extracted from each view.

Your job is to:
1. FOLLOW THE TASK for each page — count the specific symbols named, measure the specific areas listed
2. USE THE SCHEDULE DATA below to determine quantities for scheduled items (qty = row count)
3. USE THE KEYNOTES below to identify scope items referenced on plans
4. PRODUCE A COMPLETE ITEM LIST organized by trade, with quantities and detailed extraction methods
5. For EVERY finished surface, extract BOTH the base material AND the finish:
   - GWB wall → Drywall item (SF) + Painting item (SF)
   - Ceiling tile → Drywall item (SF of grid/tile)
   - Concrete floor with seal → Concrete item + Painting/Flooring item (sealed concrete)
6. GROUP identical items — e.g., 4 infrared heaters = 1 line item with qty=4, not 4 separate items

{project_context}

DISCIPLINE CONTEXT:
{disc_context}

═══════════════════════════════════════════════════════════════
PAGES (images attached in order below)
═══════════════════════════════════════════════════════════════
{pages_text}

═══════════════════════════════════════════════════════════════
SCHEDULES (already extracted — TRUST these, do not re-read)
═══════════════════════════════════════════════════════════════
{schedule_text}

═══════════════════════════════════════════════════════════════
KEYNOTES (already extracted)
═══════════════════════════════════════════════════════════════
{keynote_text}

═══════════════════════════════════════════════════════════════
SYMBOLS LEGEND (already extracted)
═══════════════════════════════════════════════════════════════
{symbol_text}

═══════════════════════════════════════════════════════════════
QUANTITY EXTRACTION RULES
═══════════════════════════════════════════════════════════════

FOR SCHEDULE ITEMS:
  - qty = number of rows in the schedule for that item
  - Include full spec from schedule data (size, material, manufacturer)
  - method: "from_schedule: [Schedule Name], [N] rows"
  - source: "schedule:[MARK]"

FOR COUNTING SYMBOLS ON PLANS (EA items):
  - Use QUADRANT method: divide page into NW, NE, SW, SE
  - Count each quadrant separately, then total
  - Check enlarged plans, insets, and detail callouts
  - method: "counted: [symbol] on [sheet_id], NW:X NE:X SW:X SE:X = total"
  - source: "plan_count:[sheet_id]"

FOR DIMENSIONS ON PLANS (SF/LF items):
  - Read dimension strings written on the drawing (15'-0", 22'-6")
  - Calculate: show your math (length × width = area)
  - If dimensions are clearly readable: confidence "high"
  - If estimated from scale: confidence "medium"
  - If NOT readable: set qty = null, confidence "low"
    and review = "Manual takeoff required — measure from: [sheet_id]"
  - method: "measured: [Room] = [dim1] × [dim2] = [result] [unit]"
  - source: "plan_measurement:[sheet_id]"

FOR KEYNOTE ITEMS:
  - Reference the keynote text from above
  - qty from counting keynote references on plans, or 1 LS if not countable
  - method: "from_keynote: [key] on [sheet_id]"
  - source: "keynote:[sheet_id]"

FOR DEMOLITION ITEMS:
  - Extract from demolition plans — each demo item is separate from new work
  - Count items to be removed, or flag for measurement
  - method: "counted: [item] removed on [sheet_id]" or "demolition_scope"
  - source: "plan_count:[sheet_id]" or "keynote:[sheet_id]"

EXISTING ITEMS: Skip items marked (E), (P), or "EXISTING" — they are not new scope.
FBO ITEMS: Include but append "(FBO — Install Only)" to description.

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON array)
═══════════════════════════════════════════════════════════════

Return a JSON array. Each item:
{{
  "trade": "Painting",
  "description": "Paint GWB Walls, 2 coats latex eggshell",
  "qty": 546,
  "unit": "SF",
  "mark": "",
  "source": "plan_measurement:A102",
  "method": "measured: Toilet 101 walls = 213 SF, Office 102 walls = 333 SF. Total = 546 SF",
  "confidence": "high",
  "material_spec": "",
  "review": ""
}}

Rules:
- trade: one of [{trades_text}] — assign each item to its correct CSI trade
- qty: number or null (null only when dimensions are unreadable)
- unit: EA, SF, LF, SY, CY, LS, LBS
- confidence: "high", "medium", "low"
- method: DETAILED — show counting breakdown or dimension math
- review: empty unless manual takeoff needed
- GROUP identical items — combine instances with total qty (e.g., 4 heaters = qty 4, not 4 separate items)
- Every finished surface needs BOTH base material AND finish (GWB wall = Drywall SF + Paint SF)
- Do NOT include existing items unless they say RELOCATE or MODIFY

Return ONLY the JSON array. No markdown fences, no commentary.
"""
    return prompt, skip_pages


# ════════════════════════════════════════════════════════════════════════════════
# Response parsing
# ════════════════════════════════════════════════════════════════════════════════

_CONFIDENCE_MAP = {"high": 0.85, "medium": 0.65, "low": 0.45}


def _parse_response(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error(f"JSON parse failed: {text[:300]}")
                return []
        else:
            logger.error(f"No JSON array: {text[:300]}")
            return []

    if isinstance(parsed, dict):
        parsed = parsed.get("items", [])
    return parsed if isinstance(parsed, list) else []


def _to_items(parsed: List[Dict], discipline: str) -> List[EstimateItem]:
    items = []
    for d in parsed:
        desc = d.get("description", "").strip()
        if not desc:
            continue

        # Use trade from response if provided, fallback to discipline name
        trade = d.get("trade", "").strip() or discipline

        qty = None
        qty_raw = d.get("qty")
        if qty_raw is not None:
            try:
                qty = float(qty_raw)
                if qty <= 0:
                    qty = None
            except (ValueError, TypeError):
                qty = None

        conf_str = d.get("confidence", "medium")
        confidence = _CONFIDENCE_MAP.get(conf_str, 0.65)

        review = d.get("review", "") or ""
        source = d.get("source", "") or ""
        method = d.get("method", "") or ""
        mark = d.get("mark", "") or ""
        material = d.get("material_spec", "") or ""

        needs_measurement = qty is None and d.get("unit", "") in ("SF", "LF", "SY", "CY", "LBS")
        needs_counting = qty is None and d.get("unit", "") == "EA"

        if needs_measurement and not review:
            source_ref = source.split(":")[-1] if ":" in source else "drawings"
            review = f"Manual takeoff required — measure from: {source_ref}"

        items.append(EstimateItem(
            trade=trade,
            item_description=desc,
            qty=qty,
            unit=d.get("unit", "EA"),
            extraction_method=method,
            confidence=confidence,
            source=source,
            schedule_mark=mark,
            material_spec=material,
            needs_measurement=needs_measurement,
            needs_counting=needs_counting,
            review_reason=review,
            notes=method,
        ))
    return items


# ════════════════════════════════════════════════════════════════════════════════
# Process one discipline (with retry)
# ════════════════════════════════════════════════════════════════════════════════

def _process_discipline(
    discipline: str,
    package: DisciplinePackage,
    page_images: Dict[int, bytes],
    project_context: str,
) -> List[EstimateItem]:
    if not package.pages:
        return []

    t0 = time.time()
    client = _get_client()
    prompt, skip_pages = _build_prompt(discipline, package, project_context)

    # Build content: prompt + page images (skip schedule/notes-only pages)
    contents: List[Any] = [prompt]
    pages_sent = 0
    for pg in sorted(package.pages):
        if pg in skip_pages:
            continue
        img = page_images.get(pg)
        if img:
            contents.append(genai_types.Part.from_bytes(data=img, mime_type="image/jpeg"))
            pages_sent += 1

    if pages_sent == 0:
        logger.info(f"  [{discipline}] All pages are schedule/notes — skipping Pro call")
        return []

    # Call with retry
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=16384,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
                ),
            )
            break
        except Exception as e:
            err_str = str(e)
            if attempt < MAX_RETRIES and ("503" in err_str or "UNAVAILABLE" in err_str or "SSL" in err_str or "EOF" in err_str):
                wait = 15 * (attempt + 1)
                logger.warning(f"  [{discipline}] Attempt {attempt+1} failed ({err_str[:80]}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  [{discipline}] Gemini call failed: {e}")
                return []

    raw = response.text or ""
    parsed = _parse_response(raw)
    items = _to_items(parsed, discipline)

    elapsed = time.time() - t0
    logger.info(f"  [{discipline}] {len(items)} items in {elapsed:.1f}s from {len(package.pages)} pages")

    return items


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

CONTEXT_ONLY = {
    "title/index", "cover", "general", "index", "title",
    "unknown", "other", "cover sheet", "title sheet", "general notes",
}


def extract_by_trade(
    classification: DocumentClassificationResult,
    sheets: List[SheetInfo],
    tables: List[ExtractedTable],
    packages: Dict[str, DisciplinePackage],
    project_context: str = "",
) -> List[EstimateItem]:
    """Step 3: Read plans and extract quantities per discipline.

    Args:
        classification: Step 1 output (has raw_pdf_bytes for rendering)
        sheets: Step 2 output
        tables: Step 2 output (not used directly — data is in packages)
        packages: Step 2 discipline packages with schedules, keynotes, symbols
        project_context: formatted string from Step 2 sheet map

    Returns:
        List of all EstimateItem across all disciplines.
    """
    t0 = time.time()

    # Separate context-only vs active disciplines
    active = {}
    context_disciplines = []
    for disc, pkg in packages.items():
        if disc.lower().strip() in CONTEXT_ONLY:
            context_disciplines.append(disc)
            logger.info(f"  {disc:25s} | {len(pkg.pages)} pages → context only")
        elif pkg.pages:
            active[disc] = pkg

    if not active:
        logger.warning("No active disciplines found")
        return []

    # Build project context string if not provided
    if not project_context:
        # Aggregate context from all packages
        ctx_parts = []
        for pkg in packages.values():
            if pkg.context:
                ctx_parts.append(pkg.context)
        project_context = f"PROJECT CONTEXT:\n" + "\n".join(ctx_parts) if ctx_parts else ""

    logger.info(f"Step 3: {len(active)} disciplines to process")
    for disc, pkg in sorted(active.items()):
        logger.info(
            f"  {disc:25s} | {len(pkg.pages)} pages, "
            f"{len(pkg.schedules)} schedules, {len(pkg.keynotes)} keynotes"
        )

    # Collect page images directly from classification (most reliable source)
    # Do NOT rely on SheetInfo.image_bytes — Flash may truncate PAGE INFO responses
    # and skip pages, leaving SheetInfo without images for those pages.
    page_images: Dict[int, bytes] = {}
    for cf in classification.files:
        for page in cf.pages:
            if page.image_bytes and page.has_drawings:
                page_images[page.global_page_number] = page.image_bytes

    # Supplement with any images in SheetInfo (on-demand rendered pages)
    for s in sheets:
        if s.global_page_number not in page_images and s.image_bytes:
            page_images[s.global_page_number] = s.image_bytes

    logger.info(f"  {len(page_images)} page images available (from Step 1, 300 DPI)")

    # Process each discipline in parallel
    all_items: List[EstimateItem] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(
                _process_discipline,
                disc, pkg, page_images, project_context,
            ): disc
            for disc, pkg in active.items()
        }

        for future in as_completed(futures):
            disc = futures[future]
            try:
                items = future.result()
                all_items.extend(items)
            except Exception as e:
                logger.error(f"  [{disc}] Failed: {e}")

    elapsed = time.time() - t0
    logger.info(
        f"Step 3 complete: {len(all_items)} items from "
        f"{len(active)} disciplines in {elapsed:.1f}s"
    )

    return all_items
