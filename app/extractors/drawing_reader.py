"""Drawing Reader — Step 2d.

Uploads the full drawings PDF(s) to Gemini via the Files API, sends
pre-extracted schedule data (from Step 2a) + page index (from Step 1),
and asks Gemini Pro to:
  1. LINK schedules to plans (which schedules relate to which plan pages)
  2. CLASSIFY every page (plan, schedule, detail, elevation, etc.)
  3. EXTRACT explicit QTY values from schedule rows
  4. FLAG items as needs_measurement (dimension-based) or needs_counting
     (plan-based symbol counting — the model identifies WHAT to count
     and WHERE, but does NOT attempt to count itself)
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types

from app.core.document_classification import (
    DocumentCategory,
    DocumentClassificationResult,
)
from app.core.estimate_models import EstimateItem, ExtractedTable, SheetInfo

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-pro"
MAX_RETRIES = 3

# ════════════════════════════════════════════════════════════════════════════════
# Client
# ════════════════════════════════════════════════════════════════════════════════

def _get_genai_client() -> genai.Client:
    """Get Gemini client (same pattern as table_extractor)."""
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
# Prompt
# ════════════════════════════════════════════════════════════════════════════════

DRAWING_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR performing a COMPLETE scope extraction from
a set of construction drawings. Your goal is to produce a FULL Bill of Materials (BOM)
— every estimable item that a contractor would need to price this project.

You have:
1. The full drawings PDF file(s) — you can see every page at native quality.
2. A PAGE INDEX listing every sheet with its ID, title, and discipline.
3. PRE-EXTRACTED SCHEDULE DATA — structured tables already parsed from schedule pages.

IMPORTANT: Schedules are only ONE source of items. Most construction scope lives in
plans, keynotes, general notes, details, and legends — NOT in schedules. A drawing
set will often explicitly state "items shown on plans but omitted from schedules
must still be provided and installed." You MUST extract from ALL sources.

═══════════════════════════════════════════════════════════════════
EXTRACTION STRATEGY — 4 PASSES OVER THE DRAWINGS
═══════════════════════════════════════════════════════════════════

PASS 1 — SCHEDULE-BASED ITEMS:
  Pre-extracted schedule data is provided below as REFERENCE to help you
  identify scheduled items. Use it as a starting point, but ALWAYS verify
  against the actual drawing pages. The PDF is the ultimate source of truth.
  For each scheduled item:
  - Read the actual schedule on the drawing page to confirm types and specs
  - Link to plan pages where the mark appears (scan plans for mark symbols)
  - Include full material spec (manufacturer, model, finish, size)
  - Flag as needs_counting if qty depends on counting symbols on plans
  - Flag as needs_measurement if qty depends on area/length
  Set source_type="schedule" for all items from this pass.

PASS 2 — PLAN-DRIVEN ITEMS (keynotes, symbols, annotations):
  Scan EVERY plan page for items NOT in any schedule. This is critical —
  most electrical devices, demolition scope, and architectural elements
  have no schedule. Extract:

  DEMOLITION (AD/D-series pages):
    - Walls to demolish (needs_measurement, LF)
    - Ceiling to demolish (needs_measurement, SF)
    - Flooring to demolish (needs_measurement, SF)
    - Doors/frames to remove (needs_counting, EA)
    - Fixtures/equipment to remove (needs_counting, EA)
    - Light fixtures to remove (needs_counting, EA)
    - MEP devices to remove/relocate (needs_counting, EA)
    - Items to salvage for reuse (EA)
    - Patching/repair after removals (needs_measurement)

  ARCHITECTURAL (A-series plan pages):
    - Wall types by type mark (e.g., Type A, C3) — needs_measurement, LF per type
    - Door marks on plans (needs_counting per mark, EA)
    - Ceiling types on RCP (ACT, GWB, open/exposed) — needs_measurement, SF per type
    - Millwork/casework visible on plans and elevations
    - Backing/blocking/support plates for wall-mounted items
    - Sound insulation at restroom/lobby/stair perimeters
    - Soffits, bulkheads, furr-downs visible on RCP
    - Access panels — flush frameless GWB access panels in partitions/ceilings for MEP
      access (if called out on plans, details, or notes) — needs_counting, EA
    - Glazing / glass partitions — tempered glass, sidelites, borrowed lites,
      glass partition walls with framing, silicone sealant at joints — needs_measurement

  SIGNAGE & EGRESS:
    - Tactile / ADA room signs (needs_counting, EA)
    - Exit signs, illuminated (needs_counting, EA)
    - Directional exit signs (needs_counting, EA)
    - "TO EXIT" wall-mounted signs on egress plans (needs_counting, EA)
    - Any other signage callouts in keynotes or notes

  ELECTRICAL (E-series pages):
    - Duplex receptacles, GFI receptacles, dedicated outlets (needs_counting)
    - Light switches — single-pole, 3-way, dimmer (needs_counting)
    - Data/telecom outlets, telephone jacks (needs_counting)
    - Exit signs, directional exit signs (needs_counting)
    - Fire alarm devices — pull stations, strobes, horns, speakers (needs_counting)
    - Smoke detectors, CO detectors (needs_counting)
    - Occupancy/vacancy sensors (needs_counting)
    - Thermostats / temperature sensors / BAS tie-ins (needs_counting)
    - Junction boxes, disconnect switches (needs_counting)
    - Conduit/wire runs (needs_measurement, LF)
    - Electrical panels, transformers (needs_counting)
    - Low-voltage: security, access control, card readers (needs_counting)

  MECHANICAL/HVAC (M-series pages):
    - Diffusers, grilles, registers not in schedule (needs_counting)
    - Ductwork runs (needs_measurement, LBS or LF)
    - Fire dampers, smoke dampers (needs_counting)
    - Access panels/doors in ductwork or ceiling (needs_counting)
    - Piping insulation (needs_measurement)
    - Thermostats, temperature sensors, controls, BAS/DDC tie-ins (needs_counting)
    - Relocated existing thermostats or controls (needs_counting, EA)
    - Ceiling exhaust fans / inline fans discharging to cavity (needs_counting)
    - Refrigerant piping, condensate lines (needs_measurement)

  PLUMBING (P-series pages):
    - Fixtures not in schedule — floor drains, cleanouts, hose bibs (needs_counting)
    - Piping runs by type (needs_measurement, LF)
    - Water heater connections, gas connections
    - Backflow preventers, shut-off valves, dual check valves (needs_counting)
    - Plumbing accessories: drain pans with leak detection, water bug/solenoid
      valves, vacuum relief valves, trap primers (needs_counting)
    - Coffee maker / ice maker cold water connections with shutoff (needs_counting)
    - Cleanout adjustments, floor drain adjustments at new finishes (needs_counting)

  FIRE PROTECTION (FP-series pages):
    - Sprinkler heads (needs_counting)
    - Fire extinguisher cabinets (needs_counting)
    - Standpipe connections (needs_counting)

PASS 2B — ELEVATION & DETAIL SHEETS (building envelope):
  Explicitly scan ALL elevation sheets and detail/section sheets for building
  envelope scope. These items are CRITICAL and only appear on elevations/sections,
  NOT on floor plans. Set source_type="elevation" or "detail" for these items.

  ROOFING:
    - Metal roof panels / standing seam — type, gauge, profile (needs_measurement, SF)
    - Roof underlayment / ice & water shield (needs_measurement, SF)
    - Ridge cap, hip cap, valley flashing (needs_measurement, LF)
    - Ridge vent (needs_measurement, LF)
    - Roof penetration flashings — pipe boots, curbs (needs_counting, EA)
    - Snow guards / snow retention system (needs_measurement, LF)

  SIDING & CLADDING:
    - Metal wall panels / siding — type, profile, gauge (needs_measurement, SF)
    - Metal liner panels (needs_measurement, SF)
    - Wall sheathing / weather barrier (needs_measurement, SF)

  GUTTERS & DRAINAGE:
    - Gutters — profile, size, material (needs_measurement, LF)
    - Downspouts — size, material (needs_counting + needs_measurement, EA/LF)

  TRIM & FLASHING:
    - Corner trim, J-channel, starter strip (needs_measurement, LF)
    - Window/door head flashing, sill flashing (needs_measurement, LF)
    - Drip edge at eaves and rakes (needs_measurement, LF)
    - Fascia — material, size (needs_measurement, LF)

  SOFFIT:
    - Soffit panels — vented or solid, material (needs_measurement, SF)

  STRUCTURAL SHOWN ON DETAILS:
    - Concrete pads at overhead doors or equipment (needs_measurement, EA/SF)
    - New columns or posts shown in sections (needs_counting, EA)
    - Lintels, headers, steel connections shown in details (needs_counting, EA)

PASS 3 — GENERAL NOTES, DETAILS & NOTE-ONLY SCOPE:
  Read general notes pages, construction notes, and detail pages for scope that
  is STATED IN TEXT but not shown as symbols on plans. These are real estimate
  items — output each as a separate item with source_type="general_note" or
  source_type="detail".

  DRYWALL / GWB:
    - Water-resistant GWB (greenboard/densglass) at wet/tile areas
    - Level-5 finish on GWB where specified
    - Abuse-resistant GWB in high-traffic areas

  FINISHES & ACCESSORIES:
    - Painting of exposed deck/structure in open ceiling areas
    - Floor prep/leveling/patching before new finishes
    - Floor transition strips — rubber reducing strips between different finishes
    - Stainless steel edge trim (Schluter or similar) at tile edges
    - Ceiling perimeter trim — Armstrong Axiom or similar at ACT/cove conditions
    - Caulking/sealant at perimeters, joints, penetrations
    - FRP (fiberglass reinforced panels) at service/kitchen areas

  GLAZING ACCESSORIES:
    - Silicone sealant at glass partition sides and butt-joints
    - Setting blocks, gaskets, trim for glass installations

  ACOUSTIC & FIRE:
    - Acoustic requirements (STC ratings, insulation at walls)
    - Seismic bracing for ceiling grid, ductwork, piping
    - Firestopping / fire caulk at all MEP penetrations through rated assemblies (LS)
    - Fire-rated framing at rated walls (double-layer GWB, mineral wool, etc.)

  GENERAL CONSTRUCTION / REPAIR:
    - Patch and repair of damaged existing construction after demolition
    - Floor patching / leveling at removed partition locations
    - Infill framing at removed door openings
    - Temporary dust barriers / protection during construction (LS)
    - Final cleaning (LS)

  ADA / CODE:
    - ADA compliance items (grab bars, signage, clearances)
    - Permits, inspections, testing noted on drawings

PASS 4 — CROSS-REFERENCE & LINKING:
  After extracting from all sources, cross-reference:
    - Link schedule items to their plan pages
    - Link demolition scope to corresponding new-work items
    - Identify which plan pages contain items still needing counting/measurement
    - Classify every page by type

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — Return a JSON object:
═══════════════════════════════════════════════════════════════════
{
  "items": [
    {
      "trade": "one of the 23 standard trades",
      "item_description": "clear description (max 100 chars)",
      "material_spec": "manufacturer, model, spec if known",
      "qty": <number or null>,
      "unit": "EA|SF|LF|SY|CY|LS|LBS",
      "schedule_mark": "F-1 or CP-1 etc. or empty string if no schedule",
      "schedule_page": 13 or null,
      "plan_pages": [
        {"page": 7, "sheet_id": "A1.11", "title": "Enlarged Plans - Suite 100"},
        {"page": 8, "sheet_id": "A2.10", "title": "Reflected Ceiling Plans"}
      ],
      "source_pages": [7, 8, 13],
      "source_type": "schedule|keynote|plan_symbol|general_note|detail|elevation|legend",
      "needs_measurement": false,
      "measurement_note": "",
      "needs_counting": true,
      "counting_note": "Count F-1 symbols on RCP page 8",
      "confidence": "high|medium|low",
      "reasoning": "brief explanation"
    }
  ],
  "page_classification": {
    "0": {"type": "cover", "description": "Title sheet"},
    "7": {"type": "plan", "description": "Floor finish plan", "linked_schedules": [13]},
    "13": {"type": "schedule", "description": "Finish, lighting, plumbing schedules"}
  }
}

═══════════════════════════════════════════════════════════════════
23 STANDARD TRADES
═══════════════════════════════════════════════════════════════════
General Requirements, Site Work, Masonry, Concrete, Metals, Rough Carpentry,
Finish Carpentry, Plumbing, Electrical, HVAC and Sheet Metals, Insulation,
Doors and Windows, Drywall, Cabinets, Stucco and Siding, Painting, Roofing,
Tile & Solid Surfaces, Bath and Accessories, Appliances, Flooring,
Fire Sprinklers, Landscaping

═══════════════════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════════════════
1. ONE ITEM PER UNIQUE TYPE — combine all instances into one item with correct qty.
2. TRUST the pre-extracted schedule data for schedule-sourced items.
3. For countable items on plans → needs_counting=true, qty=null. DO NOT guess counts.
4. For area/length items → needs_measurement=true, qty=null. DO NOT compute areas.
5. For items with explicit QTY in schedule → extract the number, confidence="high".
6. Include schedule_mark for schedule items, empty string for non-schedule items.
7. source_pages = ALL relevant pages for this item.
8. SKIP non-estimable items (title blocks, revision tables, code references).
9. Cross-reference across ALL pages — schedules link to plans.
10. Extract from EVERY page type — plans, notes, details, legends, keynotes, schedules.
    Do NOT focus only on schedules. The majority of scope is in plans and notes.
11. Include source_type to track where each item was found.
12. For note-driven items (backing, insulation, prep, patching), reference the page
    where the note appears and the plan pages where the work applies.
13. For note-only / lump-sum items (firestopping, patch & repair, temp protection,
    final cleaning, permits), use qty=1, unit="LS", source_type="general_note".
    These are real scope items that affect pricing — do NOT skip them.
14. For relocated / existing items to be reconnected (thermostats, controls),
    create a separate item with source_type="keynote" or "plan_symbol".
15. For plumbing/mechanical accessories (drain pans, check valves, shutoffs,
    trap primers, vacuum relief), extract even if quantity is 1 EA — they are
    individually priced items that estimators need to see.
16. BE THOROUGH. An experienced estimator reviewing your output should not find
    major scope gaps. Think about what a contractor would need to BUILD this project.

17. CONSTRUCTION ABBREVIATIONS — understand these standard markings:
    (P) = existing / previously installed — DO NOT include as new BOM item
    (E) = existing — DO NOT include as new BOM item
    EXIST / EXISTING = not new scope — DO NOT include
    (N) = new work — include in BOM
    NIC = not in contract — DO NOT include
    FBO = furnished by owner — include but prefix description with "FBO — Install only:"
          and set material_spec to "Furnished by Owner"
    BY OWNER = same as FBO
    If an existing item is being RELOCATED or RECONNECTED, include it with
    description prefixed "Relocate:" and source_type="keynote".

18. The PRE-EXTRACTED SCHEDULES below are REFERENCE only — always verify against
    the actual PDF. If the PDF shows different data than the pre-extracted schedules,
    trust the PDF. Read schedule abbreviations exactly (HM, SC, RB, OHD, etc.).

19. SCAN ELEVATIONS AND DETAILS for building envelope items (roofing, siding, gutters,
    trim, soffit, flashing, metal panels). These are high-value items that ONLY appear
    on elevation and detail/section sheets, never on floor plans.

Return ONLY valid JSON. No markdown fences, no explanation.
"""


# ════════════════════════════════════════════════════════════════════════════════
# File upload
# ════════════════════════════════════════════════════════════════════════════════

def _upload_pdf(client: genai.Client, pdf_bytes: bytes, filename: str):
    """Upload a PDF to Gemini Files API and wait until ACTIVE."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        logger.info(f"  Uploading {filename} ({len(pdf_bytes) / 1024 / 1024:.1f} MB)...")
        uploaded = client.files.upload(file=tmp_path)

        # Poll until ACTIVE (typically 5-30s for a drawing PDF)
        max_wait = 120
        waited = 0
        while uploaded.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(3)
            waited += 3
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(
                f"File upload failed for {filename}: state={uploaded.state.name}"
            )

        logger.info(f"  Uploaded {filename} — ACTIVE")
        return uploaded
    finally:
        os.unlink(tmp_path)


# ════════════════════════════════════════════════════════════════════════════════
# Build prompt context
# ════════════════════════════════════════════════════════════════════════════════

def _build_page_index(sheets: List[SheetInfo]) -> str:
    """Build page index text from sheet info."""
    lines = ["PAGE INDEX:"]
    for s in sheets:
        lines.append(
            f"  Page {s.global_page_number}: {s.sheet_id} | "
            f"{s.title or 'Untitled'} | {s.discipline}"
        )
    return "\n".join(lines)


def _build_schedule_context(tables: List[ExtractedTable]) -> str:
    """Build schedule context from pre-extracted tables (Step 2a)."""
    if not tables:
        return "PRE-EXTRACTED SCHEDULES: None detected."

    # Group tables by page
    by_page: Dict[int, List[ExtractedTable]] = {}
    for t in tables:
        by_page.setdefault(t.page_number, []).append(t)

    lines = ["PRE-EXTRACTED SCHEDULES (from Step 2a — trust as ground truth):"]
    for page_num in sorted(by_page):
        page_tables = by_page[page_num]
        sheet_id = page_tables[0].sheet_id if page_tables else ""
        lines.append(f"\n  Page {page_num} (Sheet {sheet_id}):")

        for i, t in enumerate(page_tables):
            lines.append(f"    Table {i + 1}: {t.schedule_type} schedule")
            if t.headers:
                lines.append(f"      Headers: {', '.join(h for h in t.headers if h)}")
            lines.append(f"      Rows ({len(t.rows)}):")
            for j, row in enumerate(t.rows):
                vals = [f"{k}={v}" for k, v in row.items() if v and str(v).strip()]
                lines.append(f"        {j + 1}. {'; '.join(vals[:8])}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Response parsing
# ════════════════════════════════════════════════════════════════════════════════

def _parse_response(raw: str) -> Dict[str, Any]:
    """Parse JSON response from model."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
        # If model returned a list, wrap it
        if isinstance(result, list):
            return {"items": result, "page_classification": {}}
    except json.JSONDecodeError:
        # Try to find JSON object in response
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"Could not parse JSON from drawing reader: {raw[:500]}")

    return {"items": [], "page_classification": {}}


def _dicts_to_estimate_items(parsed: Dict[str, Any]) -> List[EstimateItem]:
    """Convert parsed JSON to EstimateItem objects."""
    items_list = parsed.get("items", [])
    if not isinstance(items_list, list):
        return []

    confidence_map = {"high": 0.85, "medium": 0.65, "low": 0.45}
    items: List[EstimateItem] = []

    for d in items_list:
        if not isinstance(d, dict):
            continue

        qty_raw = d.get("qty")
        qty = float(qty_raw) if qty_raw is not None else None

        needs_measurement = bool(d.get("needs_measurement", False))
        needs_counting = bool(d.get("needs_counting", False))

        conf_str = str(d.get("confidence", "medium")).lower()
        confidence = confidence_map.get(conf_str, 0.65)

        # Lower confidence for items needing further work
        if needs_measurement or needs_counting:
            confidence = min(confidence, 0.50)

        # Determine extraction method from source_type + flags
        source_type = str(d.get("source_type", ""))
        source_type_method_map = {
            "schedule": "drawing_schedule",
            "keynote": "drawing_keynote",
            "plan_symbol": "drawing_plan_symbol",
            "general_note": "drawing_general_note",
            "detail": "drawing_detail",
            "legend": "drawing_legend",
        }
        if needs_counting:
            method = "drawing_needs_counting"
        elif needs_measurement:
            method = "drawing_needs_measurement"
        elif qty is not None:
            method = source_type_method_map.get(source_type, "drawing_schedule_qty")
        else:
            method = source_type_method_map.get(source_type, "drawing_linked")

        # Build source page from schedule_page + plan_pages
        source_pages = d.get("source_pages", [])
        source_page = source_pages[0] if source_pages else d.get("schedule_page", 0)

        # Plan pages — may be enriched objects or flat ints
        raw_plan_pages = d.get("plan_pages", [])
        plan_page_numbers: List[int] = []
        for pp in raw_plan_pages:
            if isinstance(pp, dict):
                plan_page_numbers.append(int(pp.get("page", 0)))
            elif isinstance(pp, (int, float)):
                plan_page_numbers.append(int(pp))

        # Build source provenance tag
        sheet_id_for_source = ""
        if raw_plan_pages and isinstance(raw_plan_pages[0], dict):
            sheet_id_for_source = str(raw_plan_pages[0].get("sheet_id", ""))

        if source_type == "schedule":
            sched_mark = str(d.get("schedule_mark", ""))
            source_tag = f"schedule:{sched_mark}" if sched_mark else "schedule"
        elif needs_counting:
            source_tag = f"plan_count:{sheet_id_for_source}" if sheet_id_for_source else "plan_count"
        elif needs_measurement:
            source_tag = f"plan_measurement:{sheet_id_for_source}" if sheet_id_for_source else "plan_measurement"
        elif source_type == "elevation":
            source_tag = f"elevation:{sheet_id_for_source}" if sheet_id_for_source else "elevation"
        elif source_type == "detail":
            source_tag = f"detail:{sheet_id_for_source}" if sheet_id_for_source else "detail"
        elif source_type == "general_note":
            source_tag = "note"
        elif source_type == "keynote":
            source_tag = f"keynote:{sheet_id_for_source}" if sheet_id_for_source else "keynote"
        else:
            source_tag = source_type or "drawing"

        item = EstimateItem(
            trade=str(d.get("trade", "General Requirements")),
            item_description=str(d.get("item_description", ""))[:100],
            qty=qty,
            unit=str(d.get("unit", "")),
            extraction_method=method,
            confidence=confidence,
            source_page=int(source_page) if source_page else 0,
            sheet_id=str(d.get("sheet_id", "")),
            material_spec=str(d.get("material_spec", "")),
            schedule_mark=str(d.get("schedule_mark", "")),
            needs_measurement=needs_measurement,
            needs_counting=needs_counting,
            counting_target=str(d.get("counting_note", "")),
            counting_source_pages=plan_page_numbers,
            needs_field_verification=False,
            review_reason=str(d.get("measurement_note", "") or d.get("counting_note", "")),
            notes=str(d.get("reasoning", "")),
            source=source_tag,
        )
        items.append(item)

    return items


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

def read_drawings(
    classification: DocumentClassificationResult,
    sheets: List[SheetInfo],
    tables: List[ExtractedTable],
) -> List[EstimateItem]:
    """Read drawing PDFs via Gemini file upload for linking + flagging.

    Uploads all drawing PDFs to Gemini, sends page index + pre-extracted
    schedules, and gets back linked items with needs_counting/needs_measurement
    flags.

    Args:
        classification: Document classification (has raw_pdf_bytes)
        sheets: SheetInfo list from Step 1
        tables: Extracted tables from Step 2a

    Returns:
        List of EstimateItem with linking and flagging.
    """
    if not sheets:
        return []

    # Find drawing PDF files
    drawing_files = classification.get_files_by_category(
        DocumentCategory.CONSTRUCTION_DRAWINGS
    )
    if not drawing_files:
        logger.warning("No drawing files found in classification")
        return []

    # Collect PDF bytes for drawing files
    pdf_entries: List[tuple] = []  # (filename, bytes)
    for cf in drawing_files:
        if cf.filename in classification.raw_pdf_bytes:
            pdf_entries.append((cf.filename, classification.raw_pdf_bytes[cf.filename]))

    if not pdf_entries:
        logger.warning("No raw PDF bytes found for drawing files")
        return []

    logger.info(
        f"Drawing reader: {len(pdf_entries)} PDF(s), "
        f"{len(sheets)} sheets, {len(tables)} pre-extracted tables"
    )

    client = _get_genai_client()

    # Upload all drawing PDFs
    uploaded_files = []
    try:
        for filename, pdf_bytes in pdf_entries:
            uploaded = _upload_pdf(client, pdf_bytes, filename)
            uploaded_files.append(uploaded)

        # Build prompt context
        page_index = _build_page_index(sheets)
        schedule_context = _build_schedule_context(tables)

        prompt_text = (
            f"{DRAWING_PROMPT}\n\n"
            f"{page_index}\n\n"
            f"{schedule_context}"
        )

        # Build contents: prompt + all uploaded file refs
        contents: list = [prompt_text]
        for uf in uploaded_files:
            contents.append(uf)

        # Call Gemini Pro
        t0 = time.time()
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(
                    f"  Sending to {MODEL} "
                    f"({len(uploaded_files)} file(s), {len(sheets)} sheets)"
                    f"{' retry ' + str(attempt + 1) if attempt > 0 else ''}"
                )

                response = client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        temperature=0,
                        max_output_tokens=65536,
                        thinking_config=genai_types.ThinkingConfig(
                            thinking_budget=32768,
                        ),
                    ),
                )

                # Extract text (skip thinking parts)
                raw_text = ""
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if part.text and (not hasattr(part, "thought") or not part.thought):
                            raw_text += part.text

                if not raw_text:
                    raw_text = response.text or "{}"

                dur = time.time() - t0
                logger.info(f"  Response received [{dur:.1f}s], {len(raw_text)} chars")

                parsed = _parse_response(raw_text)
                items = _dicts_to_estimate_items(parsed)

                # Log page classification if available
                page_class = parsed.get("page_classification", {})
                if page_class:
                    logger.info(f"  Page classifications: {len(page_class)} pages")

                # Log summary
                n_counting = sum(1 for i in items if i.needs_counting)
                n_measurement = sum(1 for i in items if i.needs_measurement)
                n_qty = sum(1 for i in items if i.qty is not None)
                logger.info(
                    f"  Drawing reader: {len(items)} items "
                    f"(qty={n_qty}, needs_counting={n_counting}, "
                    f"needs_measurement={n_measurement}) [{dur:.1f}s]"
                )

                return items

            except Exception as e:
                logger.error(f"  Drawing reader attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

    finally:
        # Clean up uploaded files
        for uf in uploaded_files:
            try:
                client.files.delete(name=uf.name)
                logger.debug(f"  Deleted uploaded file: {uf.name}")
            except Exception as e:
                logger.warning(f"  Failed to delete uploaded file: {e}")

    return []
