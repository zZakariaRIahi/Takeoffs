"""Step 2 — Full Context Extraction.

Single step that replaces the old table_extractor + sheet_mapper.
For each discipline group, sends all pages to Gemini Flash and extracts
EVERYTHING text-based in one pass:
  - Sheet index + discipline mapping
  - Schedules (headers, rows, exact values)
  - Keynotes and general notes
  - Plan names per page
  - Symbol definitions and legends
  - Context summary for Step 3's plan-reading agent

Step 3 receives these packages and focuses ONLY on visual plan work
(dimensions, symbol counting) — no re-reading of text.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
    ExtractedScheduleRow,
    ExtractedTable,
    SheetInfo,
)

logger = logging.getLogger(__name__)

WORKERS = 10
MODEL = "gemini-2.5-flash"


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
# Discipline mapping (code, no AI)
# ════════════════════════════════════════════════════════════════════════════════

_PREFIX_MAP: Dict[str, str] = {
    "AD": "Architectural Demolition", "A": "Architectural", "S": "Structural",
    "D": "Demolition", "M": "Mechanical", "E": "Electrical", "P": "Plumbing",
    "FP": "Fire Protection", "L": "Landscape", "C": "Civil", "G": "General",
    "T": "Title/Index", "I": "Interior",
    "V": "Mechanical", "VD": "Mechanical Demolition",
    "PD": "Plumbing Demolition", "ED": "Electrical Demolition",
    "MD": "Mechanical Demolition", "CD": "Civil Demolition",
    "SD": "Structural Demolition",
}

_SHEET_ID_PATTERNS = [
    re.compile(r'\b((?:AD|FP|VD|PD|ED|MD|CD|SD)[.-]?\d{1,3}[.-]?\d{0,2}[A-Z]?)\b', re.IGNORECASE),
    re.compile(r'\b([A-Z][.-]?\d{1,3}[.-]?\d{0,2}[A-Z]?)\b'),
    re.compile(r'(?:SHEET|DWG)\s*[:#]?\s*([A-Z]{1,2}[.-]?\d+[.-]?\d*)', re.IGNORECASE),
]


def _detect_discipline(sheet_id: str) -> str:
    sid = sheet_id.upper()
    if sid.startswith("PAGE-"):
        return "Unknown"
    for prefix in sorted(_PREFIX_MAP, key=len, reverse=True):
        if sid.startswith(prefix):
            return _PREFIX_MAP[prefix]
    return "Unknown"


# ════════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════════

SHEET_INDEX_PROMPT = """\
You are reading the FIRST PAGE of a construction document. Look for a SHEET INDEX table.

A sheet index is a TABLE on the title page that lists ALL drawing sheets in the set.
It typically has columns like: Sheet Number, Sheet Title, and sometimes Discipline.
Example of a real sheet index:
  T101  TITLE SHEET
  C1    GENERAL NOTES
  A102  FLOOR PLAN AND SCHEDULES
  E101  FIRST FLOOR LIGHTING PLAN

CRITICAL RULES:
- ONLY extract a sheet index if you see an ACTUAL TABLE listing sheets on this page
- If this page is a bid form, scope summary, specification, floor plan, or any page
  WITHOUT a sheet index table → return an EMPTY array: []
- Do NOT invent or guess sheet entries. Do NOT create entries from room names,
  keynotes, or other content on the page.
- If you are unsure whether something is a sheet index → return []

If a real sheet index table IS present, extract each entry:

Group sheets into their REAL disciplines:
- Civil, Architectural (includes AD- demo), Structural
- Plumbing (includes PD- demo), Mechanical (includes V- ventilation + VD- demo)
- Electrical (includes ED- demo), Fire Protection
- Abatement (ASB-, LBP-), Landscape

Return a JSON array:
[
  {"sheet_id": "G-001", "title": "TITLE SHEET", "discipline": "Architectural", "page": 0},
  {"sheet_id": "C-001", "title": "GENERAL CIVIL NOTES", "discipline": "Civil", "page": 1},
  {"sheet_id": "A-102", "title": "FLOOR PLAN AND SCHEDULES", "discipline": "Architectural", "page": 5}
]

Rules:
- sheet_id: the sheet number exactly as shown in the table
- title: the full sheet title/description from the table
- discipline: one of the standard disciplines listed above
- page: 0-based page number (title page = 0, sheets listed sequentially after)
- If NO sheet index table exists on this page → return []
- Return ONLY the JSON array, no commentary
"""

DISCIPLINE_EXTRACT_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR reading {discipline} drawing pages.

You have TWO jobs:
1. EXTRACT all text content (schedules, keynotes, symbols) — an AI vision model
   will read these pages next, so it won't need to re-read any text you extract here.
2. CLASSIFY each view on each page and write FOCUSED INSTRUCTIONS for what the
   vision model should count, measure, or extract from each view.

═══════════════════════════════════════════════════════════════
WHAT TO EXTRACT
═══════════════════════════════════════════════════════════════

DO NOT extract schedule tables — they will be extracted separately in a focused pass.
Just identify which pages have schedules in the PAGE INFO section.

KEYNOTES — Every keynote, general note, numbered note relevant to scope
SYMBOLS — Symbol definitions from legend pages
CONTEXT — Brief summary of this discipline's scope

═══════════════════════════════════════════════════════════════
PAGE CLASSIFICATION
═══════════════════════════════════════════════════════════════

Each page may contain MULTIPLE views. Classify each view using these types:

  site_plan        — overhead map showing property, utilities, grading
  floor_plan       — horizontal cut showing room layouts, walls, doors, fixtures
  demolition_plan  — floor plan showing items to be removed
  elevation        — flat exterior facade view showing heights, materials, openings
  section          — vertical cut through building showing internal construction
  detail           — enlarged view of a specific component or assembly
  structural       — foundation layouts, columns, beams, reinforcement
  MEP              — mechanical/electrical/plumbing system layouts
  RCP              — reflected ceiling plan showing ceiling fixtures, finishes, diffusers
  roof_plan        — overhead view of roof showing slopes, drains, equipment
  landscape        — exterior landscaping, paths, outdoor structures
  schedule         — tabular data (door schedule, finish schedule, etc.)
  notes            — general notes, abbreviations, legends, symbols
  abatement        — environmental/hazardous material plans

A single page can have: a floor plan + a schedule + an enlarged detail.
List ALL views on each page.

═══════════════════════════════════════════════════════════════
STEP 3 INSTRUCTIONS
═══════════════════════════════════════════════════════════════

For each non-schedule, non-notes view, write a SPECIFIC instruction telling the
vision model exactly what to look for. Be precise — name the symbols, materials,
and items to count or measure.

Examples of GOOD instructions:
  "Count door marks by type (101, 103A, 103B etc). Count plumbing fixtures: WC-1,
   UR-1, LAV-1 symbols. Count fire extinguisher cabinets (FEC)."
  "Count light fixtures by type: A1, A1-EM, A2, B1, EM, EX. Count occupancy
   sensors (OS). Measure ACT-1 and ACT-2 ceiling areas."
  "Measure north facade area for metal panel cladding. Count windows by type.
   Measure roof edge/parapet length."
  "Count items to REMOVE: doors (marked X), walls (dashed), ceiling areas (hatched).
   Reference keynotes 1-15 for demolition scope."
  "Extract insulation types and locations from wall section. Note vapor barrier
   requirements. No counting — material specs only."
  "Count receptacles (duplex, GFCI, dedicated). Count switches. Count data outlets.
   Reference symbol legend from E-001."

Examples of BAD instructions (too vague):
  "Count electrical items" — which items? What symbols?
  "Measure areas" — which areas? For what material?
  "Extract scope" — what scope specifically?

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (use this EXACT markdown structure)
═══════════════════════════════════════════════════════════════

# CONTEXT
Brief summary of scope, special conditions, references to other disciplines.

# PAGE INFO
- Page 14 (A102):
  Views: floor_plan, schedule, detail
  Plans: First Floor Plan, Large Scale Toilet Plan, Door Schedule, Finish Schedule
  STEP3: Count door marks on floor plan (match to door schedule). Count plumbing
  fixtures: WC-1, UR-1, LAV-1. Count fire extinguisher cabinets. Read room
  dimensions for flooring takeoff (Room 101: toilet, Room 102: tool storage,
  Room 103: storage).
- Page 15 (A103):
  Views: RCP
  Plans: Reflected Ceiling Plan
  STEP3: Count light fixtures by type (A1, A1-EM, A2, B1-EM, EM, EX). Count
  supply/return diffusers. Measure ACT-1 ceiling area, ACT-2 ceiling area,
  GWB ceiling area.
- Page 17 (A105):
  Views: elevation
  Plans: North Elevation, South Elevation, East Elevation, West Elevation
  STEP3: Measure each facade area for metal panel cladding (Color 1, Color 2).
  Count windows by type. Measure roof edge length. Identify soffit materials.
- Page 18 (A106):
  Views: section, detail
  Plans: Building Section A, Wall Section Details
  STEP3: Extract wall assembly components (insulation, sheathing, vapor barrier).
  Note flashing details. No symbol counting — material specs and assembly info.
- Page 20 (A108):
  Views: detail
  Plans: Door Head/Jamb/Sill Details, Roof Edge Details
  STEP3: Extract assembly components only. No counting or measuring needed.
- Page 13 (A101):
  Views: notes, schedule
  Plans: General Notes, Abbreviations, Code Data
  STEP3: SKIP — notes and schedules already extracted. No vision work needed.

# SYMBOLS
- WC-1: Wall-mounted water closet, Zurn Z5615 [fixture]
- A1: 2x4 LED troffer, Lithonia 2BLT4 [lighting]
- EX: Exit sign, thermoplastic [lighting]

# KEYNOTES
- 1 (page 7): REMOVE EXISTING DOOR, FRAME AND HARDWARE
- 5 (page 14): NEW 6" RUBBER BASE
- 12 (page 14): PATCH AND REPAIR WALL TO MATCH EXISTING

IMPORTANT RULES:
- Page numbers are 0-based integers shown above each image
- DO NOT extract schedule table contents — just flag pages that have schedules in Views
- Preserve exact abbreviations in keynotes and symbols
- Include EVERY page in PAGE INFO with views, plans, and STEP3 instruction
- STEP3 instruction must be SPECIFIC — name exact symbols, materials, items to count/measure
- For pages with only schedules/notes: write "STEP3: SKIP — already extracted"
"""


# ════════════════════════════════════════════════════════════════════════════════
# Discipline package dataclass
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DisciplinePackage:
    """Everything extracted for one discipline — input to Step 3."""
    discipline: str = ""
    pages: List[int] = field(default_factory=list)
    page_info: List[Dict[str, Any]] = field(default_factory=list)
    schedules: List[Dict[str, Any]] = field(default_factory=list)
    keynotes: List[Dict[str, Any]] = field(default_factory=list)
    symbols: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""


# ════════════════════════════════════════════════════════════════════════════════
# Sheet index extraction
# ════════════════════════════════════════════════════════════════════════════════

def _extract_sheet_index(client: genai.Client, title_page_image: bytes) -> List[Dict[str, str]]:
    """Extract sheet index from title page image."""
    contents = [
        SHEET_INDEX_PROMPT,
        genai_types.Part.from_bytes(data=title_page_image, mime_type="image/jpeg"),
    ]
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                ),
            )
            raw = response.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            parsed = _parse_json_robust(raw, "sheet_index")
            if parsed is None:
                return []
            if isinstance(parsed, list):
                return parsed
            return parsed.get("sheets", parsed.get("index", []))
        except Exception as e:
            err_str = str(e)
            if attempt < 2 and ("503" in err_str or "SSL" in err_str or "UNAVAILABLE" in err_str):
                wait = 15 * (attempt + 1)
                logger.warning(f"  Sheet index attempt {attempt+1} failed ({err_str[:80]}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"Sheet index extraction failed: {e}")
                return []
    return []


# ════════════════════════════════════════════════════════════════════════════════
# Robust JSON parsing (handles construction drawing quirks)
# ════════════════════════════════════════════════════════════════════════════════

def _parse_json_robust(raw: str, label: str = "") -> Optional[Dict]:
    """Parse JSON with fallback repair for common construction data issues.

    Construction drawings produce values like 3'-0", 1-3/4", 1/4" = 1'-0"
    which contain unescaped double quotes that break JSON parsing.
    """
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to repair: fix unescaped quotes inside string values
    # Pattern: find " inside a JSON string value that's followed by common construction suffixes
    repaired = raw
    # Replace inch marks inside values: digits followed by " that isn't a JSON delimiter
    repaired = re.sub(r'(\d)"(\s*[,}\]\n])', r'\1\\"\2', repaired)
    repaired = re.sub(r'(\d)"(\s*[A-Z])', r'\1\\"\2', repaired)  # 3'-0" WIDTH
    repaired = re.sub(r'(\d)"(\s*[a-z])', r'\1\\"\2', repaired)  # 1/4" thick
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Try truncation repair: if JSON was cut off, close it
    truncated = repaired.rstrip()
    # Try adding closing brackets
    for suffix in ['"}]}', '"}],"keynotes":[],"page_info":[],"symbols":[],"context":""}',
                   '],"keynotes":[],"page_info":[],"symbols":[],"context":""}',
                   ']}', '}]', '}']:
        try:
            return json.loads(truncated + suffix)
        except json.JSONDecodeError:
            continue

    # Last resort: try to extract each section independently
    result = {"schedules": [], "keynotes": [], "page_info": [], "symbols": [], "context": ""}
    for key in ["schedules", "keynotes", "page_info", "symbols"]:
        pattern = rf'"{key}"\s*:\s*(\[.*?\])(?:\s*[,}}])'
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                result[key] = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    context_match = re.search(r'"context"\s*:\s*"(.*?)"(?:\s*[,}])', raw, re.DOTALL)
    if context_match:
        result["context"] = context_match.group(1)

    if any(result[k] for k in ["schedules", "keynotes", "page_info", "symbols"]):
        logger.warning(f"  [{label}] Partial JSON recovery: {len(result['schedules'])} schedules, "
                       f"{len(result['keynotes'])} keynotes, {len(result['symbols'])} symbols")
        return result

    logger.error(f"  [{label}] JSON parse failed completely")
    logger.error(f"  [{label}] Raw (first 500): {raw[:500]}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# Markdown response parser
# ════════════════════════════════════════════════════════════════════════════════

def _parse_markdown_table(lines: List[str]) -> Tuple[List[str], List[Dict[str, str]]]:
    """Parse a markdown pipe-delimited table into headers + rows."""
    headers = []
    rows = []
    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]  # skip empty first/last from split
        if not cells:
            continue
        # Skip separator lines (|---|---|)
        if all(re.match(r'^[-:]+$', c) for c in cells):
            continue
        if not headers:
            headers = cells
        else:
            row = {}
            for i, h in enumerate(headers):
                row[h] = cells[i] if i < len(cells) else ""
            rows.append(row)
    return headers, rows


def _parse_markdown_response(raw: str, discipline: str, page_indices: List[int]) -> DisciplinePackage:
    """Parse the markdown-formatted response from Flash into a DisciplinePackage."""
    pkg = DisciplinePackage(discipline=discipline, pages=page_indices)

    lines = raw.split("\n")
    current_section = None
    current_schedule_title = ""
    current_schedule_type = ""
    current_schedule_page = -1
    table_lines: List[str] = []

    def _flush_schedule():
        """Save accumulated table lines as a schedule."""
        if table_lines and current_schedule_title:
            headers, rows = _parse_markdown_table(table_lines)
            if headers and rows:
                pkg.schedules.append({
                    "title": current_schedule_title,
                    "type": current_schedule_type,
                    "page": current_schedule_page,
                    "headers": headers,
                    "rows": rows,
                })

    for line in lines:
        stripped = line.strip()

        # Detect section headers
        if stripped.startswith("# CONTEXT"):
            _flush_schedule()
            current_section = "context"
            table_lines = []
            continue
        elif stripped.startswith("# PAGE INFO"):
            _flush_schedule()
            current_section = "page_info"
            table_lines = []
            continue
        elif stripped.startswith("# SYMBOLS"):
            _flush_schedule()
            current_section = "symbols"
            table_lines = []
            continue
        elif stripped.startswith("# KEYNOTES"):
            _flush_schedule()
            current_section = "keynotes"
            table_lines = []
            continue
        elif stripped.startswith("# SCHEDULE:"):
            _flush_schedule()
            current_section = "schedule"
            table_lines = []
            # Parse: # SCHEDULE: DOOR SCHEDULE (door) [page 14]
            m = re.match(r'#\s*SCHEDULE:\s*(.+?)\s*\((\w+)\)\s*\[page\s*(\d+)\]', stripped)
            if m:
                current_schedule_title = m.group(1).strip()
                current_schedule_type = m.group(2).strip()
                current_schedule_page = int(m.group(3))
            else:
                # Fallback: just grab the title
                title_part = stripped.replace("# SCHEDULE:", "").strip()
                current_schedule_title = re.sub(r'\s*\(.*?\)\s*\[.*?\]', '', title_part).strip()
                current_schedule_type = "other"
                current_schedule_page = -1
            continue

        # Skip empty lines
        if not stripped:
            continue

        # Parse content based on current section
        if current_section == "context":
            if pkg.context:
                pkg.context += " " + stripped
            else:
                pkg.context = stripped

        elif current_section == "page_info":
            # New multi-line format:
            # - Page 14 (A102):
            #   Views: floor_plan, schedule, detail
            #   Plans: First Floor Plan, Large Scale Toilet Plan
            #   STEP3: Count door marks...
            # OR old single-line format:
            # - Page 14 (A102): First Floor Plan [has_plans] [has_schedules]

            # New page entry
            m = re.match(r'-\s*Page\s+(\d+)\s*\(([^)]+)\):\s*(.*)', stripped)
            if m:
                pg = int(m.group(1))
                sid = m.group(2).strip()
                rest = m.group(3).strip()

                # Check if old format (single line with [has_*] tags)
                if "[has_" in rest:
                    has_sched = "[has_schedules]" in rest
                    has_plans = "[has_plans]" in rest
                    plan_text = re.sub(r'\[has_\w+\]', '', rest).strip().rstrip(',')
                    plans = [p.strip() for p in plan_text.split(",") if p.strip()] if plan_text else []
                    pkg.page_info.append({
                        "page": pg, "sheet_id": sid, "plans": plans,
                        "has_schedules": has_sched, "has_plans": has_plans,
                        "views": [], "step3_instruction": "",
                    })
                else:
                    # New multi-line format — start a new page entry
                    pkg.page_info.append({
                        "page": pg, "sheet_id": sid, "plans": [],
                        "has_schedules": False, "has_plans": False,
                        "views": [], "step3_instruction": "",
                    })

            # Sub-fields of current page entry
            elif pkg.page_info and stripped.startswith("Views:"):
                views_text = stripped.replace("Views:", "").strip()
                views = [v.strip() for v in views_text.split(",") if v.strip()]
                pkg.page_info[-1]["views"] = views
                pkg.page_info[-1]["has_plans"] = any(
                    v in views for v in ["floor_plan", "elevation", "section", "RCP",
                                         "roof_plan", "site_plan", "MEP", "structural",
                                         "demolition_plan", "landscape", "abatement"]
                )
                pkg.page_info[-1]["has_schedules"] = "schedule" in views

            elif pkg.page_info and stripped.startswith("Plans:"):
                plans_text = stripped.replace("Plans:", "").strip()
                plans = [p.strip() for p in plans_text.split(",") if p.strip()]
                pkg.page_info[-1]["plans"] = plans

            elif pkg.page_info and stripped.startswith("STEP3:"):
                instruction = stripped.replace("STEP3:", "").strip()
                pkg.page_info[-1]["step3_instruction"] = instruction

            elif pkg.page_info and pkg.page_info[-1].get("step3_instruction") is not None:
                # Continuation of STEP3 instruction (multi-line)
                last = pkg.page_info[-1]
                if last["step3_instruction"] and not stripped.startswith("-") and not stripped.startswith("Views:") and not stripped.startswith("Plans:"):
                    last["step3_instruction"] += " " + stripped

        elif current_section == "symbols":
            # - WC-1: Wall-mounted water closet, Zurn Z5615 [fixture]
            m = re.match(r'-\s*(.+?):\s*(.+?)(?:\s*\[(\w+)\])?\s*$', stripped)
            if m:
                pkg.symbols.append({
                    "symbol": m.group(1).strip(),
                    "description": m.group(2).strip(),
                    "category": m.group(3).strip() if m.group(3) else "",
                })

        elif current_section == "keynotes":
            # - 1 (page 7): REMOVE EXISTING DOOR, FRAME AND HARDWARE
            m = re.match(r'-\s*(.+?)\s*\(page\s*(\d+)\):\s*(.+)', stripped)
            if m:
                pkg.keynotes.append({
                    "key": m.group(1).strip(),
                    "page": int(m.group(2)),
                    "text": m.group(3).strip(),
                })
            else:
                # Fallback: - K5: some text
                m2 = re.match(r'-\s*(.+?):\s*(.+)', stripped)
                if m2:
                    pkg.keynotes.append({
                        "key": m2.group(1).strip(),
                        "page": -1,
                        "text": m2.group(2).strip(),
                    })

        elif current_section == "schedule":
            # Accumulate table lines
            if stripped.startswith("|"):
                table_lines.append(stripped)

    # Flush last schedule
    _flush_schedule()

    return pkg


# ════════════════════════════════════════════════════════════════════════════════
# Per-discipline extraction
# ════════════════════════════════════════════════════════════════════════════════

def _extract_discipline(
    client: genai.Client,
    discipline: str,
    page_indices: List[int],
    page_images: Dict[int, bytes],
    sheet_lookup: Dict[int, Dict[str, str]],
) -> DisciplinePackage:
    """Extract all text-based content for one discipline."""
    if not page_indices:
        return DisciplinePackage(discipline=discipline)

    import time
    t0 = time.time()

    # Build prompt
    prompt = DISCIPLINE_EXTRACT_PROMPT.format(discipline=discipline)

    # Add page labels and images
    contents: List[Any] = [prompt]
    for pg in sorted(page_indices):
        sheet_info = sheet_lookup.get(pg, {})
        sid = sheet_info.get("sheet_id", f"PAGE-{pg}")
        title = sheet_info.get("title", "")
        label = f"\n--- Page {pg}: {sid} — {title} ---"
        contents.append(label)
        img = page_images.get(pg)
        if img:
            contents.append(genai_types.Part.from_bytes(data=img, mime_type="image/jpeg"))

    # Call Flash with retry
    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                ),
            )
            break
        except Exception as e:
            err_str = str(e)
            if attempt < 2 and ("503" in err_str or "SSL" in err_str or "EOF" in err_str or "UNAVAILABLE" in err_str):
                wait = 15 * (attempt + 1)
                logger.warning(f"  [{discipline}] Attempt {attempt+1} failed ({err_str[:80]}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  [{discipline}] Flash extraction failed: {e}")
                return DisciplinePackage(discipline=discipline, pages=page_indices)

    if not response:
        return DisciplinePackage(discipline=discipline, pages=page_indices)

    # Parse markdown response
    raw = response.text or ""
    pkg = _parse_markdown_response(raw, discipline, page_indices)

    elapsed = time.time() - t0
    logger.info(
        f"  [{discipline}] {len(pkg.schedules)} schedules, {len(pkg.keynotes)} keynotes, "
        f"{len(pkg.symbols)} symbols in {elapsed:.1f}s from {len(page_indices)} pages"
    )

    return pkg


# ════════════════════════════════════════════════════════════════════════════════
# Pass 2: Focused schedule extraction (one page at a time)
# ════════════════════════════════════════════════════════════════════════════════

SCHEDULE_EXTRACT_PROMPT = """\
You are reading a construction drawing page. Extract EVERY schedule/table on this page.

A schedule is any structured table with headers and rows — door schedules, finish schedules,
equipment schedules, fixture schedules, panel schedules, lighting fixture schedules,
abbreviation tables, code data tables, etc.

For EACH table, return a markdown section:

# SCHEDULE: [TABLE TITLE] ([type]) [page {page_num}]
| HEADER1 | HEADER2 | HEADER3 |
|---------|---------|---------|
| value1 | value2 | value3 |

Type must be one of: door, window, finish, fixture, equipment, panel, lighting,
hardware, plumbing_fixture, mechanical, ventilation, abbreviation, code_data, other

Rules:
- Extract ALL rows — do not truncate or summarize
- Preserve exact abbreviations (HM, SC, RB, OHD, STL, GYP/PT, etc.)
- Preserve exact dimensions (3'-0", 1-3/4", etc.)
- Flatten merged cells or sub-headers into simple rows
- If no tables exist on this page, return "NO SCHEDULES"
"""


def _extract_one_schedule_page(
    client: genai.Client,
    page_num: int,
    img_bytes: bytes,
    sheet_info: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Extract schedules from a single page."""
    sid = sheet_info.get("sheet_id", f"PAGE-{page_num}")
    prompt = SCHEDULE_EXTRACT_PROMPT.replace("{page_num}", str(page_num))

    contents = [
        f"--- Page {page_num}: {sid} — {sheet_info.get('title', '')} ---",
        prompt,
        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
    ]

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(temperature=0.1),
        )
    except Exception as e:
        logger.error(f"    Schedule extraction failed for page {page_num}: {e}")
        return []

    raw = response.text or ""
    if "NO SCHEDULES" in raw.upper():
        return []

    # Parse markdown tables from response
    schedules = []
    lines = raw.split("\n")
    current_title = ""
    current_type = ""
    current_page = page_num
    table_lines = []

    def flush():
        if table_lines and current_title:
            headers, rows = _parse_markdown_table(table_lines)
            if headers and rows:
                schedules.append({
                    "title": current_title,
                    "type": current_type,
                    "page": current_page,
                    "headers": headers,
                    "rows": rows,
                })

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# SCHEDULE:"):
            flush()
            table_lines = []
            m = re.match(r'#\s*SCHEDULE:\s*(.+?)\s*\((\w+)\)\s*\[page\s*(\d+)\]', stripped)
            if m:
                current_title = m.group(1).strip()
                current_type = m.group(2).strip()
                current_page = int(m.group(3))
            else:
                title_part = stripped.replace("# SCHEDULE:", "").strip()
                current_title = re.sub(r'\s*\(.*?\)\s*\[.*?\]', '', title_part).strip()
                current_type = "other"
                current_page = page_num
        elif stripped.startswith("|"):
            table_lines.append(stripped)

    flush()
    return schedules


def _extract_schedules_pass2(
    client: genai.Client,
    page_nums: List[int],
    page_images: Dict[int, bytes],
    page_to_sheet: Dict[int, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Extract schedules from multiple pages in parallel (one page per call)."""
    all_schedules = []

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for pg in page_nums:
            img = page_images.get(pg)
            if not img:
                continue
            sheet_info = page_to_sheet.get(pg, {})
            futures[pool.submit(
                _extract_one_schedule_page, client, pg, img, sheet_info
            )] = pg

        for future in as_completed(futures):
            pg = futures[future]
            try:
                schedules = future.result()
                if schedules:
                    logger.info(f"    Page {pg}: {len(schedules)} schedules extracted")
                    all_schedules.extend(schedules)
            except Exception as e:
                logger.error(f"    Page {pg}: schedule extraction failed: {e}")

    return all_schedules


# ════════════════════════════════════════════════════════════════════════════════
# Convert to existing data models
# ════════════════════════════════════════════════════════════════════════════════

def _packages_to_models(
    packages: Dict[str, DisciplinePackage],
    sheet_index: List[Dict[str, str]],
    all_page_images: Dict[int, bytes],
) -> Tuple[List[SheetInfo], List[ExtractedTable], List[ExtractedScheduleRow]]:
    """Convert discipline packages to the existing data models (SheetInfo, ExtractedTable, etc.)."""

    sheets: List[SheetInfo] = []
    tables: List[ExtractedTable] = []
    schedule_rows: List[ExtractedScheduleRow] = []

    # Build sheets from sheet index
    seen_pages = set()
    for pkg in packages.values():
        for pi in pkg.page_info:
            pg = pi.get("page", -1)
            if pg in seen_pages:
                continue
            seen_pages.add(pg)
            sid = pi.get("sheet_id", f"PAGE-{pg}")
            plans = pi.get("plans", [])
            title = pi.get("sheet_name", "") or ", ".join(plans) if plans else ""

            # Fall back to sheet index for title
            if not title:
                for idx_entry in sheet_index:
                    if idx_entry.get("sheet_id", "").upper() == sid.upper():
                        title = idx_entry.get("title", "")
                        break

            sheets.append(SheetInfo(
                sheet_id=sid,
                title=title,
                discipline=pkg.discipline,
                global_page_number=pg,
                image_bytes=all_page_images.get(pg),
            ))

    # Add any sheets from sheet index that weren't in page_info
    for idx_entry in sheet_index:
        sid = idx_entry.get("sheet_id", "")
        if not any(s.sheet_id.upper() == sid.upper() for s in sheets):
            disc = idx_entry.get("discipline", "").strip() or _detect_discipline(sid)
            sheets.append(SheetInfo(
                sheet_id=sid,
                title=idx_entry.get("title", ""),
                discipline=disc,
            ))

    sheets.sort(key=lambda s: s.global_page_number)

    # Build tables from schedule data
    _MARK_HEADERS = {"MARK", "NO", "NUMBER", "DOOR NO", "ROOM", "TAG", "SYMBOL",
                     "SET", "PLAN NO", "EQUIPMENT", "FIXTURE", "TYPE", "DOOR NO."}

    for pkg in packages.values():
        for sched in pkg.schedules:
            headers = sched.get("headers", [])
            rows = sched.get("rows", [])
            if not rows:
                continue

            table = ExtractedTable(
                page_number=sched.get("page", -1),
                sheet_id=sched.get("sheet_id", ""),
                schedule_type=sched.get("type", "other"),
                headers=headers,
                rows=rows,
                confidence=0.90,
                table_title=sched.get("title", ""),
            )
            tables.append(table)

            # Convert to schedule rows
            mark_col = None
            for h in headers:
                if h.upper().replace(".", "").strip() in _MARK_HEADERS:
                    mark_col = h
                    break

            for row in rows:
                mark = str(row.get(mark_col, "")).strip() if mark_col else ""
                schedule_rows.append(ExtractedScheduleRow(
                    schedule_type=sched.get("type", "other"),
                    row_data=row,
                    mark=mark,
                    page_number=sched.get("page", -1),
                    sheet_id=sched.get("sheet_id", ""),
                ))

    return sheets, tables, schedule_rows


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

def extract_context(
    classification: DocumentClassificationResult,
) -> Tuple[
    List[SheetInfo],
    List[ExtractedTable],
    List[ExtractedScheduleRow],
    Dict[str, DisciplinePackage],
]:
    """Step 2: Extract all text-based content from drawing pages.

    Returns:
        sheets: List[SheetInfo] — one per drawing sheet
        tables: List[ExtractedTable] — all extracted schedules
        rows: List[ExtractedScheduleRow] — flattened schedule rows
        packages: Dict[discipline, DisciplinePackage] — rich context for Step 3
    """
    import time
    t0 = time.time()
    client = _get_client()

    # ── Collect all drawing pages with images ──
    _KEEP_CATEGORIES = {
        DocumentCategory.PROJECT_SPECIFICATIONS,
        DocumentCategory.CONSTRUCTION_DRAWINGS,
        DocumentCategory.COVER_SHEET,
    }

    all_pages: List[Tuple[int, bytes]] = []  # (global_page_number, image_bytes)
    page_images: Dict[int, bytes] = {}

    for cf in classification.files:
        if cf.categories and not cf.categories.intersection(_KEEP_CATEGORIES):
            continue
        for page in cf.pages:
            if page.image_bytes and page.has_drawings:
                all_pages.append((page.global_page_number, page.image_bytes))
                page_images[page.global_page_number] = page.image_bytes

    if not all_pages:
        logger.warning("No drawing pages found")
        return [], [], [], {}

    all_pages.sort(key=lambda x: x[0])
    logger.info(f"Step 2: {len(all_pages)} drawing pages from {len(classification.files)} files")

    # ── Extract sheet index from first page of the file (always page 0) ──
    # The sheet index is always on page 0 (title/cover page), not the first drawing page.
    # If page 0 doesn't have an image yet, render it now.
    logger.info("  Extracting sheet index from first page of file...")
    title_img = None
    for cf in classification.files:
        if cf.pages:
            p0 = cf.pages[0]
            if p0.image_bytes:
                title_img = p0.image_bytes
                break
            # Page 0 not rendered — render it now
            pdf_bytes = classification.raw_pdf_bytes.get(cf.filename)
            if pdf_bytes:
                import fitz as _fitz
                doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
                if len(doc) > 0:
                    zoom = 300 / 72
                    pix = doc[0].get_pixmap(matrix=_fitz.Matrix(zoom, zoom))
                    title_img = pix.tobytes("jpeg", 85)
                doc.close()
                break
    if not title_img:
        title_img = all_pages[0][1]  # last resort: first drawing page
    sheet_index = _extract_sheet_index(client, title_img)
    logger.info(f"  Sheet index: {len(sheet_index)} entries")

    # ── Build page mapping using SEQUENTIAL assignment ──
    # Flash's guessed page numbers are unreliable (off-by-one, wrong base, etc.).
    # Sequential assignment is far more reliable: sheet_index[i] -> sorted_drawing_pages[i].
    # This works because sheet indexes list sheets in the same order as they appear in the PDF.
    page_to_sheet: Dict[int, Dict[str, str]] = {}
    discipline_pages: Dict[str, List[int]] = defaultdict(list)

    sorted_gpn = sorted(gpn for gpn, _ in all_pages)

    for idx, entry in enumerate(sheet_index):
        if idx >= len(sorted_gpn):
            break  # More index entries than actual drawing pages
        gpn = sorted_gpn[idx]
        sid = entry.get("sheet_id", "").strip()
        disc = entry.get("discipline", "").strip()
        title = entry.get("title", "").strip()

        # Fallback discipline if Flash didn't provide one
        if not disc or disc.lower() in ("", "unknown", "other"):
            disc = _detect_discipline(sid) if sid else "Unknown"

        page_to_sheet[gpn] = {
            "sheet_id": sid,
            "title": title,
            "discipline": disc,
        }
        discipline_pages[disc].append(gpn)

    # Add any drawing pages not covered by the sheet index
    for gpn in sorted_gpn[len(sheet_index):]:
        page_to_sheet[gpn] = {"sheet_id": f"PAGE-{gpn}", "title": "", "discipline": "Unknown"}
        discipline_pages["Unknown"].append(gpn)

    logger.info("  Discipline mapping:")
    for disc, pages in sorted(discipline_pages.items()):
        logger.info(f"    {disc:25s}: {len(pages)} pages — {pages[:6]}{'...' if len(pages) > 6 else ''}")

    # ── Extract per discipline (parallel) ──
    logger.info(f"  Extracting context per discipline ({len(discipline_pages)} groups)...")
    packages: Dict[str, DisciplinePackage] = {}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(
                _extract_discipline,
                client, disc, pages, page_images, page_to_sheet,
            ): disc
            for disc, pages in discipline_pages.items()
        }

        for future in as_completed(futures):
            disc = futures[future]
            try:
                pkg = future.result()
                packages[disc] = pkg
            except Exception as e:
                logger.error(f"  [{disc}] Extraction failed: {e}")
                packages[disc] = DisciplinePackage(
                    discipline=disc,
                    pages=discipline_pages[disc],
                )

    # ── Pass 2: Extract schedules from pages that have them ──
    # Pass 1 often misses schedules due to output size limits.
    # This pass sends ONLY schedule pages individually for focused table extraction.
    schedule_pages = set()
    for pkg in packages.values():
        for pi in pkg.page_info:
            if pi.get("has_schedules") or "schedule" in (pi.get("views") or []):
                schedule_pages.add(pi.get("page", -1))
    # Also add pages that Pass 1 already got schedules from (don't re-extract those)
    pages_with_schedules = set()
    for pkg in packages.values():
        for s in pkg.schedules:
            pg = s.get("page", -1)
            if pg >= 0:
                pages_with_schedules.add(pg)

    # Only extract from pages that were flagged but NOT already extracted
    pages_needing_extraction = schedule_pages - pages_with_schedules
    pages_needing_extraction = {pg for pg in pages_needing_extraction if pg in page_images}

    if pages_needing_extraction:
        logger.info(f"  Pass 2: Extracting schedules from {len(pages_needing_extraction)} pages "
                     f"(already have schedules from {len(pages_with_schedules)} pages)...")
        new_schedules = _extract_schedules_pass2(client, sorted(pages_needing_extraction),
                                                  page_images, page_to_sheet)
        # Merge into packages by discipline
        for sched in new_schedules:
            pg = sched.get("page", -1)
            sheet_info = page_to_sheet.get(pg, {})
            disc = sheet_info.get("discipline", "Unknown")
            if disc in packages:
                packages[disc].schedules.append(sched)
        logger.info(f"  Pass 2: extracted {len(new_schedules)} additional schedules")
    else:
        logger.info(f"  Pass 2: no additional schedule pages to extract "
                     f"(already have {len(pages_with_schedules)} pages with schedules)")

    # ── Convert to existing data models ──
    sheets, tables, rows = _packages_to_models(packages, sheet_index, page_images)

    elapsed = time.time() - t0
    total_sched = sum(len(p.schedules) for p in packages.values())
    total_notes = sum(len(p.keynotes) for p in packages.values())
    total_syms = sum(len(p.symbols) for p in packages.values())
    logger.info(
        f"Step 2 done in {elapsed:.0f}s: {len(sheets)} sheets, {len(tables)} schedules, "
        f"{total_notes} keynotes, {total_syms} symbols"
    )

    return sheets, tables, rows, packages
