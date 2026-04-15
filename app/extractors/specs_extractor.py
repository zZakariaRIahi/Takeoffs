"""Specs Extractor — Process project specifications for quantity takeoff.

Handles three cases:
  Case 1: Specs only (no drawings) → full takeoff from specs
  Case 2: Specs + Drawings → extract spec items + context for drawing disciplines
  Case 3: Drawings only → skip (handled elsewhere)
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types as genai_types

from app.core.document_classification import (
    DocumentCategory,
    DocumentClassificationResult,
)
from app.core.estimate_models import EstimateItem
from app.extractors.context_extractor import DisciplinePackage

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-pro"
MAX_PDF_BYTES = 45 * 1024 * 1024


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
# PDF upload helper
# ════════════════════════════════════════════════════════════════════════════════

def _upload_pdf(client: genai.Client, pdf_bytes: bytes, filename: str):
    """Upload PDF to Gemini Files API and wait until ACTIVE."""
    import fitz
    # Split if too large
    if len(pdf_bytes) > MAX_PDF_BYTES:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        mid = len(doc) // 2
        sub = fitz.open()
        sub.insert_pdf(doc, from_page=0, to_page=mid - 1)
        pdf_bytes = sub.tobytes(garbage=4, deflate=True)
        sub.close()
        doc.close()
        logger.info(f"  Split {filename} for upload: {len(pdf_bytes)/1024/1024:.1f} MB")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        logger.info(f"  Uploading {filename} ({len(pdf_bytes)/1024/1024:.1f} MB)...")
        uploaded = client.files.upload(file=tmp_path)

        # Poll until ACTIVE
        for _ in range(60):
            if uploaded.state.name == "ACTIVE":
                break
            time.sleep(3)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            logger.error(f"  Upload stuck in state: {uploaded.state.name}")
            return None

        logger.info(f"  Uploaded {filename} — ACTIVE")
        return uploaded
    finally:
        os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════════

SPECS_ONLY_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR performing a COMPLETE quantity takeoff
from project specifications. No drawings are available — the specs are your ONLY source.

Read EVERY section of this document: scope summaries, bid forms, spec sections, addenda,
environmental reports, and appendices. Extract ALL items a contractor would need to price.

═══════════════════════════════════════════════════════════════
HOW TO EXTRACT QUANTITIES LIKE AN ESTIMATOR
═══════════════════════════════════════════════════════════════

Quantities are EVERYWHERE in specs — not just in explicit numbers. Read like an estimator:

1. EXPLICIT QUANTITIES — stated directly:
   "Paint walls (approx 35,000 sf)" → qty: 35000, unit: SF
   "Provide 12 hollow metal doors" → qty: 12, unit: EA
   "Install 700 LF of rubber base" → qty: 700, unit: LF

2. COUNT LOCATIONS/ROOMS — when rooms or locations are listed:
   "Remove casework in classrooms A300 and B300" → qty: 2, unit: EA
   "Install sinks in Rooms 101, 102, 103, 104" → qty: 4, unit: EA
   "Paint Band Room, Art Room, and Library" → qty: 3, unit: EA (rooms)
   "Replace windows on north and east facades" → qty: 2, unit: EA (facades)
   Count every room, location, or instance named.

3. COUNT ITEMS IN LISTS — when individual items are enumerated:
   "Remove: 3 water closets, 2 urinals, 4 lavatories" → 3 separate items with qty each
   "Classrooms A100, A101, A102, B100, B101, B102" → qty: 6 classrooms

4. DERIVE FROM CONTEXT — when qty is implied:
   "Remove plumbing sinks and piping in classrooms A300 and B300"
   → qty: 2, unit: EA (one sink per classroom, 2 classrooms)
   "Install fire extinguisher in each stairwell" + building has 4 stairwells
   → qty: 4, unit: EA

5. PER-ITEM BREAKDOWN — split compound scope into individual line items:
   "Remove and replace VCT/carpet floor (5,000 sf) including rubber base (700 lf)"
   → Item 1: Remove existing VCT/carpet flooring — qty: 5000, unit: SF
   → Item 2: Remove existing rubber base — qty: 700, unit: LF
   → Item 3: Install new VCT flooring — qty: 5000, unit: SF
   → Item 4: Install new rubber base — qty: 700, unit: LF
   Demolition and new work are ALWAYS separate line items.

6. AREA/LINEAR CALCULATIONS — when dimensions or room counts + sizes are given:
   "Paint walls in 32 classrooms, average 1,100 SF per room" → qty: 35200, unit: SF
   "Install base in 6 classrooms, 120 LF perimeter each" → qty: 720, unit: LF

7. LUMP SUM — use LS ONLY for items that genuinely cannot be quantified:
   "Performance and Payment Bond" → qty: 1, unit: LS (correct)
   "Daily cleanup and debris removal" → qty: 1, unit: LS (correct — duration-based)
   "Permits and inspections" → qty: 1, unit: LS (correct)

   DO NOT use LS for items where rooms/locations are listed.
   "Remove casework in A300 and B300" is NOT LS — it's 2 EA.

8. ALTERNATES — extract as separate items, prefix with "ALTERNATE X:"
   "Alternate 1: Paint Band Room" → description starts with "ALTERNATE 1: ..."

9. ALLOWANCES — extract with stated dollar amount if given:
   "Allowance of $5,000 for unforeseen conditions" → qty: 5000, unit: DLRS

10. NULL QUANTITY — set qty to null ONLY when:
    - Specs say "as shown on drawings" or "per plans" (needs drawings)
    - Specs say "as required" or "as needed" with no further context
    - No rooms, no dimensions, no counts, no way to derive a number

    When qty is null, explain WHY in the review field.

═══════════════════════════════════════════════════════════════
WHAT TO EXTRACT
═══════════════════════════════════════════════════════════════

Extract from ALL sections of the document:

A. SCOPE SUMMARIES / BID FORMS — often have the most explicit quantities
B. CSI SPEC SECTIONS — material requirements, product specs, installation methods
C. DEMOLITION SECTIONS — everything to be removed (separate from new work)
D. ENVIRONMENTAL/ABATEMENT — asbestos, lead paint, mold (with areas from surveys)
E. GENERAL REQUIREMENTS — permits, bonds, insurance, temporary facilities, cleanup,
   protection, testing, inspections, closeout documentation
F. ADDENDA — scope changes, additions, deletions (if included in the document)
G. SCHEDULES — door schedules, finish schedules, equipment lists (if present)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON array)
═══════════════════════════════════════════════════════════════

For each item:
{
  "description": "Remove and dispose existing casework",
  "qty": 2,
  "unit": "EA",
  "trade": "Site Work",
  "spec_section": "02 41 19",
  "source": "specs:Scope of Work",
  "method": "from_specs: 2 classrooms listed (A300, B300)",
  "review": "",
  "material_spec": ""
}

Fields:
- description: clear, concise item description
- qty: number (derived as described above) or null (ONLY if truly unknowable)
- unit: EA, SF, LF, SY, CY, LS, LBS, DLRS
- trade: standard trade name (General Requirements, Site Work, Concrete, Masonry,
  Metals, Rough Carpentry, Finish Carpentry, Waterproofing, Insulation, Roofing,
  Stucco and Siding, Doors and Windows, Drywall, Tile & Solid Surfaces, Flooring,
  Painting, Bath and Accessories, Appliances, Fire Sprinklers, Plumbing,
  HVAC and Sheet Metals, Electrical, Cabinets)
- spec_section: CSI section number or "Scope of Work" / "Addendum" / "Environmental"
- source: "specs:XX XX XX" or "specs:Scope of Work" etc.
- method: "from_specs: [brief explanation of how qty was derived]"
- review: "" if qty is known. If null: explain why (e.g., "Dimensions not stated — needs drawings")
- material_spec: manufacturer, model, size, finish, standard (preserve exact spec language)

IMPORTANT:
- Split demolition and new work into SEPARATE line items
- Count rooms/locations explicitly — never default to LS when locations are named
- Read the ENTIRE document — scope summary, all spec sections, environmental reports
- Include alternates, allowances, and addendum changes
- Preserve exact manufacturer names, model numbers, and specifications
- The method field must explain HOW you derived the quantity
- Return ONLY the JSON array, no commentary
"""

SPECS_WITH_DRAWINGS_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR reading project specifications.
Drawings exist for this project and will be processed separately.

Your job is to extract from the specifications:
1. ALL scope items with material specs and quantities (when stated)
2. Items that REFERENCE drawings for quantities (e.g., "as shown on drawings",
   "per schedule", "per plans") — flag these so the drawings step can fill in qty
3. Special conditions, allowances, alternates, and exclusions per trade
4. Product/manufacturer requirements that the drawings won't show

For each item:
- description: detailed item with material specs
- qty: quantity if stated in specs, null if "per drawings" or not stated
- unit: EA, SF, LF, SY, CY, LS, LBS
- trade: standard trade name
- spec_section: CSI section number
- source: "specs:Section XX XX XX"
- method: "from_specs" if qty from specs, "specs_ref_drawings" if qty needs drawings
- review: "" if qty known, "Qty per drawings" if referenced, "Qty not stated" if unknown
- material_spec: full material specification

Also extract per-trade CONTEXT that the drawings agent will need:
- Special conditions or requirements per trade
- Substitution rules
- Quality requirements
- Testing/inspection requirements

Return JSON with two keys:
{
  "items": [...array of items...],
  "trade_context": {
    "Doors and Windows": "All doors to be hollow metal per Section 08 11 13. Hardware Group 1 per 08 71 00. Existing doors marked (E) to remain.",
    "Flooring": "VCT per 09 65 13, carpet tile per 09 68 13. All flooring areas to receive moisture testing per 09 05 13.",
    "Electrical": "Provide 200A panel per 26 24 16. All wiring to be copper. Emergency generator per 26 32 13.",
    ...
  }
}

IMPORTANT:
- Read EVERY specification section
- For items saying "per drawings" or "as shown" → method = "specs_ref_drawings", qty = null
- Include general conditions, temporary facilities, cleanup
- Include allowances and alternates
- The trade_context helps the drawing reading agent understand requirements
- Return ONLY valid JSON
"""


# ════════════════════════════════════════════════════════════════════════════════
# Response parsing
# ════════════════════════════════════════════════════════════════════════════════

_CONFIDENCE_MAP = {"high": 0.85, "medium": 0.65, "low": 0.45}


def _parse_json_response(raw: str) -> Any:
    """Parse JSON from model response with repair."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to fix unescaped quotes
    repaired = re.sub(r'(\d)"(\s*[,}\]\n])', r'\1\\"\2', text)
    repaired = re.sub(r'(\d)"(\s*[A-Za-z])', r'\1\\"\2', repaired)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Try to find JSON array or object
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue

    logger.error(f"Failed to parse specs response: {text[:500]}")
    return None


def _items_from_parsed(parsed: List[Dict], default_method: str = "from_specs") -> List[EstimateItem]:
    """Convert parsed JSON dicts to EstimateItem objects."""
    items = []
    for d in parsed:
        desc = d.get("description", "").strip()
        if not desc:
            continue

        qty = None
        qty_raw = d.get("qty")
        if qty_raw is not None:
            try:
                qty = float(qty_raw)
                if qty <= 0:
                    qty = None
            except (ValueError, TypeError):
                qty = None

        method = d.get("method", default_method) or default_method
        review = d.get("review", "") or ""
        source = d.get("source", "") or ""
        mark = d.get("mark", "") or ""
        material = d.get("material_spec", "") or ""
        spec_section = d.get("spec_section", "") or ""

        if not review and qty is None:
            review = "Qty not stated in specs — verify from drawings or field"

        confidence = 0.75 if qty is not None else 0.45

        items.append(EstimateItem(
            trade=d.get("trade", "General Requirements"),
            item_description=desc,
            qty=qty,
            unit=d.get("unit", "LS"),
            extraction_method=method,
            confidence=confidence,
            source=source or f"specs:{spec_section}",
            schedule_mark=mark,
            material_spec=material,
            spec_section=spec_section,
            review_reason=review,
        ))
    return items


# ════════════════════════════════════════════════════════════════════════════════
# Case 1: Specs only
# ════════════════════════════════════════════════════════════════════════════════

def extract_from_specs_only(
    classification: DocumentClassificationResult,
) -> List[EstimateItem]:
    """Case 1: No drawings — full takeoff from specifications alone."""
    t0 = time.time()
    client = _get_client()

    # Find specs files
    specs_files = []
    for cf in classification.files:
        if cf.categories and DocumentCategory.PROJECT_SPECIFICATIONS in cf.categories:
            pdf_bytes = classification.raw_pdf_bytes.get(cf.filename)
            if pdf_bytes:
                specs_files.append((cf.filename, pdf_bytes))

    if not specs_files:
        logger.warning("No specification files found")
        return []

    all_items: List[EstimateItem] = []

    for fname, pdf_bytes in specs_files:
        logger.info(f"Extracting takeoff from specs: {fname} ({len(pdf_bytes)/1024/1024:.1f} MB)")

        # Upload to Gemini
        uploaded = _upload_pdf(client, pdf_bytes, fname)
        if not uploaded:
            continue

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[SPECS_ONLY_PROMPT, uploaded],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
                ),
            )

            parsed = _parse_json_response(response.text or "")
            if isinstance(parsed, list):
                items = _items_from_parsed(parsed)
                all_items.extend(items)
                logger.info(f"  {fname}: {len(items)} items extracted")
            else:
                logger.error(f"  {fname}: unexpected response format")

        except Exception as e:
            logger.error(f"  {fname}: extraction failed: {e}")
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    elapsed = time.time() - t0
    logger.info(f"Specs-only extraction: {len(all_items)} items in {elapsed:.0f}s")
    return all_items


# ════════════════════════════════════════════════════════════════════════════════
# Case 2: Specs + Drawings
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class SpecsResult:
    """Results from specs extraction when drawings also exist."""
    items: List[EstimateItem] = field(default_factory=list)
    trade_context: Dict[str, str] = field(default_factory=dict)


def extract_from_specs_with_drawings(
    classification: DocumentClassificationResult,
) -> SpecsResult:
    """Case 2: Extract spec items + per-trade context for the drawings agent."""
    t0 = time.time()
    client = _get_client()

    specs_files = []
    for cf in classification.files:
        if cf.categories and DocumentCategory.PROJECT_SPECIFICATIONS in cf.categories:
            pdf_bytes = classification.raw_pdf_bytes.get(cf.filename)
            if pdf_bytes:
                specs_files.append((cf.filename, pdf_bytes))

    if not specs_files:
        logger.warning("No specification files found for specs+drawings extraction")
        return SpecsResult()

    all_items: List[EstimateItem] = []
    all_trade_context: Dict[str, str] = {}

    for fname, pdf_bytes in specs_files:
        logger.info(f"Extracting specs context: {fname} ({len(pdf_bytes)/1024/1024:.1f} MB)")

        uploaded = _upload_pdf(client, pdf_bytes, fname)
        if not uploaded:
            continue

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[SPECS_WITH_DRAWINGS_PROMPT, uploaded],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
                ),
            )

            parsed = _parse_json_response(response.text or "")
            if isinstance(parsed, dict):
                # Extract items
                items_data = parsed.get("items", [])
                if isinstance(items_data, list):
                    items = _items_from_parsed(items_data, default_method="from_specs")
                    all_items.extend(items)
                    logger.info(f"  {fname}: {len(items)} spec items")

                # Extract trade context
                ctx = parsed.get("trade_context", {})
                if isinstance(ctx, dict):
                    for trade, text in ctx.items():
                        if trade in all_trade_context:
                            all_trade_context[trade] += " " + str(text)
                        else:
                            all_trade_context[trade] = str(text)
                    logger.info(f"  {fname}: context for {len(ctx)} trades")

            elif isinstance(parsed, list):
                # Fallback: just items, no trade context
                items = _items_from_parsed(parsed)
                all_items.extend(items)
                logger.info(f"  {fname}: {len(items)} spec items (no trade context)")

        except Exception as e:
            logger.error(f"  {fname}: specs extraction failed: {e}")
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    elapsed = time.time() - t0
    logger.info(
        f"Specs+drawings extraction: {len(all_items)} items, "
        f"context for {len(all_trade_context)} trades in {elapsed:.0f}s"
    )
    return SpecsResult(items=all_items, trade_context=all_trade_context)


# ════════════════════════════════════════════════════════════════════════════════
# Merge specs into drawing discipline packages
# ════════════════════════════════════════════════════════════════════════════════

def merge_specs_into_packages(
    specs_result: SpecsResult,
    packages: Dict[str, DisciplinePackage],
) -> Tuple[Dict[str, DisciplinePackage], List[EstimateItem]]:
    """Merge spec items and context into drawing discipline packages.

    - Spec trade_context → appended to DisciplinePackage.context
    - Spec items with method="specs_ref_drawings" → added as hints for Step 3
    - Spec items with method="from_specs" and qty → returned as final items (skip Step 3)

    Returns:
        enriched_packages: drawing packages with spec context added
        spec_final_items: spec items that don't need drawings (have qty from specs)
    """
    # Enrich packages with spec trade context
    for trade, context_text in specs_result.trade_context.items():
        # Find matching discipline package (trade names might not match exactly)
        matched = False
        for disc, pkg in packages.items():
            # Try matching: "Doors and Windows" might map to "Architectural"
            if trade.lower() in disc.lower() or disc.lower() in trade.lower():
                pkg.context = (pkg.context + "\n\nFROM SPECIFICATIONS:\n" + context_text).strip()
                matched = True
                break
        if not matched:
            # Try partial match on keywords
            trade_lower = trade.lower()
            for disc, pkg in packages.items():
                disc_lower = disc.lower()
                if any(kw in disc_lower for kw in trade_lower.split()):
                    pkg.context = (pkg.context + "\n\nFROM SPECIFICATIONS:\n" + context_text).strip()
                    matched = True
                    break
        if not matched:
            # Add to all packages as general context
            for pkg in packages.values():
                pkg.context = (pkg.context + f"\n\nSPECS ({trade}):\n{context_text}").strip()

    # Separate spec items: those with qty (final) vs those needing drawings
    spec_final_items = []
    spec_hints_by_trade = {}

    for item in specs_result.items:
        if item.qty is not None and item.extraction_method == "from_specs":
            # Has qty from specs — this is a final item
            spec_final_items.append(item)
        else:
            # Needs drawings for qty — add as a hint/keynote to the relevant package
            trade = item.trade
            spec_hints_by_trade.setdefault(trade, []).append(item)

    # Add spec hints as keynotes in matching discipline packages
    for trade, hints in spec_hints_by_trade.items():
        for disc, pkg in packages.items():
            if trade.lower() in disc.lower() or disc.lower() in trade.lower():
                for hint in hints:
                    pkg.keynotes.append({
                        "key": f"SPEC:{hint.spec_section}",
                        "text": f"{hint.item_description} — {hint.material_spec}".strip(" —"),
                        "page": -1,
                    })
                break

    logger.info(
        f"Specs merge: {len(spec_final_items)} items with qty (final), "
        f"{sum(len(v) for v in spec_hints_by_trade.values())} items as hints for drawings"
    )

    return packages, spec_final_items
