"""Step 2b — Build Sheet Map.

Groups pages by discipline from the sheet index (Architectural, Structural,
Mechanical, Electrical, Plumbing, Civil, etc.) and uses one Gemini Flash call
to add extraction hints (what to count/measure on each page).
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types as genai_types

from app.core.estimate_models import ExtractedTable, SheetInfo

logger = logging.getLogger(__name__)

MODEL = "gemini-2.0-flash"


# ════════════════════════════════════════════════════════════════════════════════
# Client
# ════════════════════════════════════════════════════════════════════════════════

def _get_genai_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        try:
            from app.config.settings import settings
            api_key = settings.GOOGLE_API_KEY
        except Exception:
            pass
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not configured")
    return genai.Client(api_key=api_key)


# ════════════════════════════════════════════════════════════════════════════════
# Prompt — only asks for project context + per-page extraction hints
# ════════════════════════════════════════════════════════════════════════════════

SHEET_MAP_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR analyzing a set of construction drawings.

Below you have:
1. A SHEET INDEX — every sheet with page number, sheet ID, title, and discipline.
2. EXTRACTED TABLES — structured data already parsed from schedule pages.
3. UPLOADED DATA — any CSV/XLSX data uploaded by the user (may be empty).

Your job is to produce:
1. PROJECT CONTEXT — overall project info
2. PER-PAGE EXTRACTION HINTS — for each page, what items should be counted or measured

═══════════════════════════════════════════════════════════════
INSTRUCTIONS
═══════════════════════════════════════════════════════════════

1. PROJECT CONTEXT:
   - type: "new_construction", "renovation", "addition", or "mixed"
   - total_sf: building area in SF if found in code data tables or notes (0 if unknown)
   - buildings: list of building names/numbers if multiple
   - notes: key project rules (existing items marked (E)/(P), FBO items, demolition, phasing)

2. PER-PAGE ANALYSIS — For EACH page in the sheet index:
   - plans: what views are on this page (floor plan, RCP, roof plan, elevation, section, detail)
   - count_items: what EA items to count (be SPECIFIC — name fixture types, device types, symbol types)
   - measure_items: what SF/LF items to measure (name the material/system)
   - notes: any special instructions (reference legend pages, abbreviation pages, etc.)
   - Only include pages that have actual extractable content (skip pure note pages, spec pages, abbreviation-only pages)

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON)
═══════════════════════════════════════════════════════════════

{
  "project_context": {
    "type": "renovation",
    "total_sf": 15000,
    "buildings": ["Building 1A"],
    "notes": "Existing items marked (E)/(P). FBO items install only."
  },
  "pages": {
    "14": {
      "plans": ["First Floor Plan", "Large Scale Toilet Plan"],
      "count_items": ["doors by mark", "plumbing fixtures by mark", "fire extinguisher cabinets"],
      "measure_items": ["partition lengths by type mark", "flooring areas by finish type"],
      "notes": "Door marks visible on plan. Reference abbreviations from page 14."
    },
    "33": {
      "plans": ["First Floor Lighting Plan"],
      "count_items": ["light fixtures by type (A1, A2, B1, EM, EX)", "exit signs", "occupancy sensors"],
      "measure_items": [],
      "notes": "Dense plan — use symbol legend from electrical symbols page."
    }
  }
}

IMPORTANT:
- Page numbers are 0-based integers (matching the sheet index below).
- count_items: be specific — "light fixtures by type (A1, A2, B1)" not "electrical devices"
- measure_items: name what to measure — "ACT ceiling areas by type" not "areas"
- Skip pages that are pure notes, abbreviations, or specifications (nothing to count/measure)
- DO include schedule pages if they contain useful data tables
- DO include demolition plans — they have items to count (things to remove)
"""


# ════════════════════════════════════════════════════════════════════════════════
# Build inputs
# ════════════════════════════════════════════════════════════════════════════════

def _build_sheet_index_text(sheets: List[SheetInfo]) -> str:
    lines = []
    for s in sorted(sheets, key=lambda x: x.global_page_number):
        lines.append(
            f"  Page {s.global_page_number}: {s.sheet_id} | {s.title} | {s.discipline}"
        )
    return "\n".join(lines) if lines else "  (no sheets found)"


def _build_table_summary(tables: List[ExtractedTable]) -> str:
    lines = []
    for t in tables:
        page_label = f"Page {t.page_number}" if t.page_number >= 0 else "Uploaded file"
        type_label = f" ({t.schedule_type})" if t.schedule_type else ""
        title = t.table_title if t.table_title else "(untitled)"
        lines.append(
            f"  {page_label}: {title}{type_label} — "
            f"{len(t.headers)} columns, {len(t.rows)} rows"
        )
        if t.headers:
            lines.append(f"    Headers: {', '.join(t.headers[:10])}")
    return "\n".join(lines) if lines else "  (no tables extracted)"


def _group_by_discipline(sheets: List[SheetInfo]) -> Dict[str, List[int]]:
    """Group pages by discipline from the sheet index. This is the trade→pages mapping."""
    groups = defaultdict(list)
    for s in sheets:
        disc = s.discipline or "Unknown"
        groups[disc].append(s.global_page_number)
    return dict(groups)


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

def build_sheet_map(
    sheets: List[SheetInfo],
    tables: List[ExtractedTable],
) -> Dict[str, Any]:
    """Build sheet map: discipline grouping from sheet index + Flash extraction hints.

    Returns:
      - project_context: {type, total_sf, buildings, notes}
      - discipline_pages: {discipline: [page_numbers]} — from sheet index directly
      - pages: {page_str: {plans, count_items, measure_items, notes}} — from Flash
    """
    import time
    t0 = time.time()

    # ── Build discipline→pages mapping from sheet index (no AI needed) ──
    discipline_pages = _group_by_discipline(sheets)

    logger.info(f"Discipline grouping from sheet index:")
    for disc, pages in sorted(discipline_pages.items()):
        logger.info(f"  {disc:20s}: {len(pages)} pages — {pages[:8]}{'...' if len(pages) > 8 else ''}")

    # ── Build page metadata from sheets ──
    sheet_by_page = {s.global_page_number: s for s in sheets}

    # ── Build table lookup by page ──
    tables_by_page = defaultdict(list)
    for t in tables:
        if t.page_number >= 0:
            tables_by_page[t.page_number].append(t)

    # ── Call Flash for project context + extraction hints ──
    client = _get_genai_client()

    sheet_index_text = _build_sheet_index_text(sheets)
    table_summary = _build_table_summary(tables)

    uploaded_tables = [t for t in tables if t.filter_type == "uploaded_file"]
    if uploaded_tables:
        uploaded_text = "\n".join(
            f"  {t.table_title}: {len(t.rows)} rows, headers: {', '.join(t.headers[:10])}"
            for t in uploaded_tables
        )
    else:
        uploaded_text = "  (none)"

    full_prompt = f"""{SHEET_MAP_PROMPT}

═══════════════════════════════════════════════════════════════
SHEET INDEX
═══════════════════════════════════════════════════════════════
{sheet_index_text}

═══════════════════════════════════════════════════════════════
EXTRACTED TABLES (from drawing pages)
═══════════════════════════════════════════════════════════════
{table_summary}

═══════════════════════════════════════════════════════════════
UPLOADED DATA (CSV/XLSX)
═══════════════════════════════════════════════════════════════
{uploaded_text}

Return ONLY valid JSON. No markdown fences, no commentary."""

    response = client.models.generate_content(
        model=MODEL,
        contents=[full_prompt],
        config=genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            response_mime_type="application/json",
        ),
    )

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        flash_result = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse sheet map JSON: {e}")
        logger.error(f"Raw response: {raw[:500]}")
        flash_result = {}

    # ── Assemble final sheet map ──
    project_context = flash_result.get("project_context", {
        "type": "unknown", "total_sf": 0, "buildings": [], "notes": ""
    })

    # Pages info from Flash (extraction hints)
    pages_info = flash_result.get("pages", {})

    # Enrich pages_info with sheet metadata
    for s in sheets:
        pg_str = str(s.global_page_number)
        if pg_str not in pages_info:
            pages_info[pg_str] = {}
        pg = pages_info[pg_str]
        pg["sheet_number"] = s.sheet_id
        pg["sheet_name"] = s.title
        pg["discipline"] = s.discipline
        # Add tables on this page
        pg["tables_on_page"] = [t.table_title for t in tables_by_page.get(s.global_page_number, [])]

    sheet_map = {
        "project_context": project_context,
        "discipline_pages": discipline_pages,  # from sheet index, NOT from AI
        "pages": pages_info,
    }

    elapsed = time.time() - t0
    logger.info(
        f"Sheet map built in {elapsed:.1f}s: "
        f"{len(discipline_pages)} disciplines, {len(pages_info)} pages, "
        f"project type: {project_context.get('type', 'unknown')}"
    )

    return sheet_map
