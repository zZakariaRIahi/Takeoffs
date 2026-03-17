"""Sheet Index Parser — Step 1 of the extraction pipeline.

Parses the drawing sheet index (T-sheet or title page) to build a
sheet_id → title mapping, then assigns metadata to every drawing page.

After Step 1 document classification, keeps only SPECS + DRAWINGS files
and drops admin/bid/insurance pages.

Parsing strategy: Vision-only — Gemini Flash reads the title page image.
Tries page 0 first, then page 1 if no index found.
"""
from __future__ import annotations

import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.document_classification import (
    ClassifiedFile,
    DocumentCategory,
    DocumentClassificationResult,
    PageInfo,
)
from app.core.estimate_models import ExtractedTable, SheetInfo

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Discipline prefix map
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

# ════════════════════════════════════════════════════════════════════════════════
# Sheet ID detection
# ════════════════════════════════════════════════════════════════════════════════

_SHEET_ID_PATTERNS = [
    # "AD1.10", "FP1.00" — multi-letter prefix
    re.compile(r'\b((?:AD|FP)[.-]?\d{1,3}[.-]?\d{0,2}[A-Z]?)\b', re.IGNORECASE),
    # "A9.00", "E1.00", "S1.0", "A-101" — single-letter prefix
    re.compile(r'\b([A-Z][.-]?\d{1,3}[.-]?\d{0,2}[A-Z]?)\b'),
    # Explicit label: "SHEET: A-101" or "DWG# A9.00"
    re.compile(r'(?:SHEET|DWG)\s*[:#]?\s*([A-Z]{1,2}[.-]?\d+[.-]?\d*)', re.IGNORECASE),
]


def _detect_sheet_id(text: str, page_number: int) -> str:
    """Extract sheet ID from page text."""
    for pat in _SHEET_ID_PATTERNS:
        matches = pat.findall(text)
        if matches:
            matches.sort(key=len, reverse=True)
            return matches[0].upper()
    return f"PAGE-{page_number}"


def _detect_discipline(sheet_id: str, text: str) -> str:
    """Detect drawing discipline from sheet ID prefix."""
    sid = sheet_id.upper()

    if sid.startswith("PAGE-"):
        text_l = text.lower()
        if "specification" in text_l or "division" in text_l:
            return "Specification"
        return "Unknown"

    # Check multi-char prefixes first (AD, FP)
    for prefix in sorted(_PREFIX_MAP, key=len, reverse=True):
        if sid.startswith(prefix):
            return _PREFIX_MAP[prefix]

    return "Unknown"


# ════════════════════════════════════════════════════════════════════════════════
# Vision-based sheet index parsing (Gemini Flash)
# ════════════════════════════════════════════════════════════════════════════════

_SHEET_INDEX_VISION_PROMPT = """\
You are reading a construction drawing TITLE SHEET / COVER SHEET.

Find the SHEET INDEX (also called DRAWING INDEX or LIST OF DRAWINGS) on this page.
It is a table listing all drawing sheets in the set, grouped by discipline sections.

Extract EVERY sheet entry with its discipline section header.

Discipline section headers are lines like: GENERAL, ARCHITECTURAL, ARCHITECTURAL SITE,
ELECTRICAL, MECHANICAL, PLUMBING, FIRE PROTECTION, STRUCTURAL, CIVIL, LANDSCAPE, etc.
They appear as group labels above the sheets in that discipline.

Return a JSON array where each entry is:
{
  "sheet_id": "T1.0",
  "title": "Cover Sheet",
  "discipline": "General"
}

Rules:
1. Copy EXACT sheet IDs and titles — do not abbreviate or paraphrase.
2. Each sheet inherits the discipline from the section header above it.
3. If no sheet index is visible on this page, return: []
4. Return ONLY a JSON array. No markdown fences, no explanation.
"""


def _parse_sheet_index_from_vision(
    image_bytes: bytes,
) -> Dict[str, Tuple[str, str]]:
    """Parse sheet index from page image using Gemini vision."""
    try:
        import os
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning("google-genai not installed — cannot use vision sheet index parsing")
        return {}

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
        logger.warning("GOOGLE_API_KEY not available for vision-based sheet index parsing")
        return {}

    # Downscale image for the API
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size
    max_dim = 3000
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    jpeg_bytes = buf.getvalue()

    contents = [
        _SHEET_INDEX_VISION_PROMPT,
        genai_types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg"),
    ]

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=8192,
            ),
        )
        raw = response.text or "[]"
        return _parse_vision_response(raw)
    except Exception as e:
        logger.error(f"Vision-based sheet index parsing failed: {e}")
        return {}


def _parse_vision_response(raw: str) -> Dict[str, Tuple[str, str]]:
    """Parse the Gemini vision response into sheet_id → (title, discipline)."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    try:
        entries = json.loads(raw)
        if not isinstance(entries, list):
            entries = [entries]
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                entries = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning(f"Could not parse vision sheet index response: {raw[:300]}")
                return {}
        else:
            return {}

    mapping: Dict[str, Tuple[str, str]] = {}
    for entry in entries:
        sid = str(entry.get("sheet_id", "")).strip().upper()
        title = str(entry.get("title", "")).strip()
        discipline = str(entry.get("discipline", "General")).strip().title()
        if sid and re.match(r'^[A-Z]', sid):
            mapping[sid] = (title, discipline)

    return mapping


# ════════════════════════════════════════════════════════════════════════════════
# Page-to-index matching
# ════════════════════════════════════════════════════════════════════════════════

def _match_pages_to_index(
    all_pages: List[Tuple[Any, PageInfo]],
    sheet_index_map: Dict[str, Tuple[str, str]],
    index_page_idxs: set,
) -> Dict[int, Tuple[str, str, str]]:
    """Match PDF pages to sheet index entries.

    Uses the known sheet IDs from the parsed index to find which page
    each sheet is on, rather than guessing IDs from page text.

    Args:
        all_pages: List of (ClassifiedFile, PageInfo) tuples
        sheet_index_map: sheet_id → (title, discipline)
        index_page_idxs: Set of page indices that are index/title pages
                         (these get assigned to the first index entry)

    Returns:
        Dict mapping page_index → (sheet_id, title, discipline)
    """
    n_pages = len(all_pages)
    index_ids = list(sheet_index_map.keys())

    # ── Pass 1: Search each page for known sheet IDs ───────────────────
    page_to_matched_ids: Dict[int, List[str]] = {}
    sid_to_pages: Dict[str, List[int]] = {sid: [] for sid in index_ids}

    for pg_idx, (_cf, page) in enumerate(all_pages):
        if pg_idx in index_page_idxs:
            continue  # Skip index pages — they contain all IDs
        text_upper = page.extracted_text.upper()
        for sid in index_ids:
            pattern = r'\b' + re.escape(sid) + r'\b'
            if re.search(pattern, text_upper):
                page_to_matched_ids.setdefault(pg_idx, []).append(sid)
                sid_to_pages[sid].append(pg_idx)

    # ── Pass 2: Assign best page for each index ID ─────────────────────
    assigned_pages: Dict[int, str] = {}
    assigned_ids: Dict[str, int] = {}

    # Pre-assign index pages to the first index entry (title sheet)
    for pg_idx in sorted(index_page_idxs):
        if index_ids and index_ids[0] not in assigned_ids and pg_idx not in assigned_pages:
            assigned_pages[pg_idx] = index_ids[0]
            assigned_ids[index_ids[0]] = pg_idx
            logger.info(f"  Index page {pg_idx} → {index_ids[0]} (title sheet)")

    # First pass: assign IDs that appear on exactly one page
    for sid in index_ids:
        if sid in assigned_ids:
            continue
        candidates = sid_to_pages[sid]
        if len(candidates) == 1:
            pg = candidates[0]
            if pg not in assigned_pages:
                assigned_pages[pg] = sid
                assigned_ids[sid] = pg

    # Second pass: for IDs on multiple pages, pick the best unassigned page
    for sid in index_ids:
        if sid in assigned_ids:
            continue
        candidates = [p for p in sid_to_pages[sid] if p not in assigned_pages]
        if candidates:
            best = min(candidates, key=lambda p: len(page_to_matched_ids.get(p, [])))
            assigned_pages[best] = sid
            assigned_ids[sid] = best

    # ── Pass 3: Fill gaps using consecutive ordering ───────────────────
    unassigned_pages = sorted(i for i in range(n_pages) if i not in assigned_pages)
    unassigned_ids = [sid for sid in index_ids if sid not in assigned_ids]

    if unassigned_pages and unassigned_ids:
        for pg, sid in zip(unassigned_pages, unassigned_ids):
            assigned_pages[pg] = sid
            assigned_ids[sid] = pg

    # ── Build result ───────────────────────────────────────────────────
    result: Dict[int, Tuple[str, str, str]] = {}
    for pg_idx, sid in assigned_pages.items():
        title, discipline = sheet_index_map[sid]
        result[pg_idx] = (sid, title, discipline)

    matched = len(result)
    total = len(index_ids)
    logger.info(f"Page matching: {matched}/{total} index entries mapped to pages")
    if unassigned_ids:
        logger.warning(f"  Unmatched index entries: {unassigned_ids}")
    unmatched_pages = [i for i in range(n_pages) if i not in result]
    if unmatched_pages:
        logger.info(f"  Pages not in index: {unmatched_pages}")

    return result


# ════════════════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════════════════

_KEEP_CATEGORIES = {
    DocumentCategory.PROJECT_SPECIFICATIONS,
    DocumentCategory.CONSTRUCTION_DRAWINGS,
}


def build_sheet_index(
    classification: DocumentClassificationResult,
) -> List[SheetInfo]:
    """Build sheet index from classified documents.

    1. Filter to keep only SPECS + DRAWINGS files
    2. Try vision on page 0, then page 1 to parse the sheet index
    3. For each drawing page: assign sheet_id, discipline, title
    4. Return List[SheetInfo] — one per drawing page

    Args:
        classification: Output from Step 1 (document classification)

    Returns:
        List of SheetInfo objects, one per drawing page
    """
    # ── Step 1: Filter to specs + drawings ────────────────────────────────
    kept_files: List[ClassifiedFile] = []
    for f in classification.files:
        if f.categories & _KEEP_CATEGORIES:
            kept_files.append(f)

    if not kept_files:
        logger.warning("No spec or drawing files found after filtering")
        return []

    drawing_files = [f for f in kept_files if DocumentCategory.CONSTRUCTION_DRAWINGS in f.categories]
    spec_files = [f for f in kept_files if DocumentCategory.PROJECT_SPECIFICATIONS in f.categories]

    logger.info(
        f"Kept {len(drawing_files)} drawing files, {len(spec_files)} spec files "
        f"(dropped {len(classification.files) - len(kept_files)} admin/bid files)"
    )

    # ── Step 2: Parse sheet index via vision (page 0, then page 1) ────────
    sheet_index_map: Dict[str, Tuple[str, str]] = {}
    index_page_number: Optional[int] = None  # which page we found the index on

    for cf in drawing_files:
        # Try page 0 first, then page 1
        for try_page_num in (0, 1):
            if try_page_num >= len(cf.pages):
                continue

            page = cf.pages[try_page_num]
            if not page.image_bytes:
                continue

            logger.info(
                f"Trying vision on page {page.global_page_number} "
                f"({cf.filename}, local page {try_page_num})"
            )

            vision_map = _parse_sheet_index_from_vision(page.image_bytes)
            if vision_map:
                logger.info(
                    f"Vision parsed {len(vision_map)} sheets from "
                    f"page {page.global_page_number}"
                )
                sheet_index_map.update(vision_map)
                index_page_number = try_page_num
                break

        if sheet_index_map:
            break  # Found index, stop searching files

    if sheet_index_map:
        logger.info(f"Sheet index: {len(sheet_index_map)} total sheets mapped")
    else:
        logger.warning("No sheet index found — will use detected sheet IDs only")

    # ── Step 3: Map pages to sheet index entries ──────────────────────────
    all_pages: List[Tuple[ClassifiedFile, PageInfo]] = []
    for cf in drawing_files:
        for page in cf.pages:
            all_pages.append((cf, page))

    page_assignments: Dict[int, Tuple[str, str, str]] = {}

    if sheet_index_map:
        # The page we parsed the index from is the "index page"
        known_index_pages = {index_page_number} if index_page_number is not None else set()
        page_assignments = _match_pages_to_index(
            all_pages, sheet_index_map, known_index_pages
        )

    # Build SheetInfo for every page
    sheets: List[SheetInfo] = []
    for i, (cf, page) in enumerate(all_pages):
        if i in page_assignments:
            sheet_id, title, discipline = page_assignments[i]
        else:
            # Fallback for pages not in the index
            sheet_id = _detect_sheet_id(page.extracted_text, page.global_page_number)
            title = ""
            discipline = _detect_discipline(sheet_id, page.extracted_text)

        sheet = SheetInfo(
            sheet_id=sheet_id,
            title=title,
            discipline=discipline,
            global_page_number=page.global_page_number,
            source_file=cf.filename,
            extracted_text=page.extracted_text,
            image_bytes=page.image_bytes,
            tables=[],  # Populated by table_extractor.py
        )
        sheets.append(sheet)

    logger.info(
        f"Built {len(sheets)} SheetInfo objects from "
        f"{len(drawing_files)} drawing files"
    )

    # Log discipline breakdown
    disciplines: Dict[str, int] = {}
    for s in sheets:
        disciplines[s.discipline] = disciplines.get(s.discipline, 0) + 1
    for disc, count in sorted(disciplines.items()):
        logger.info(f"  {disc}: {count} sheets")

    return sheets


# ════════════════════════════════════════════════════════════════════════════════
# Spec page helpers (for downstream use)
# ════════════════════════════════════════════════════════════════════════════════

def get_spec_pages(classification: DocumentClassificationResult) -> List[PageInfo]:
    """Return all pages classified as project specifications."""
    return classification.get_pages_by_category(DocumentCategory.PROJECT_SPECIFICATIONS)


def get_spec_text(classification: DocumentClassificationResult) -> str:
    """Return concatenated text from all spec pages."""
    return classification.get_text_for_category(DocumentCategory.PROJECT_SPECIFICATIONS)
