"""Data models for the new 4-step pipeline — Step 1: Document Classification.

Defines the 8 bid document categories and the data structures that hold
classified pages/files for downstream consumption by Steps 2-4.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


# ════════════════════════════════════════════════════════════════════════════════
# Document categories
# ════════════════════════════════════════════════════════════════════════════════

class DocumentCategory(str, Enum):
    COVER_SHEET = "cover_sheet"
    INSTRUCTIONS_TO_BIDDER = "instructions_to_bidder"
    PROJECT_SPECIFICATIONS = "project_specifications"
    CONSTRUCTION_DRAWINGS = "construction_drawings"
    GENERAL_CONDITIONS = "general_conditions"
    SPECIAL_CONDITIONS = "special_conditions"
    BID_FORM = "bid_form"
    BID_SECURITY = "bid_security"


# Routing helpers — which categories need text vs vision vs both
TEXT_CATEGORIES: Set[DocumentCategory] = {
    DocumentCategory.COVER_SHEET,
    DocumentCategory.INSTRUCTIONS_TO_BIDDER,
    DocumentCategory.GENERAL_CONDITIONS,
    DocumentCategory.SPECIAL_CONDITIONS,
    DocumentCategory.BID_FORM,
    DocumentCategory.BID_SECURITY,
}

VISUAL_CATEGORIES: Set[DocumentCategory] = {
    DocumentCategory.CONSTRUCTION_DRAWINGS,
}

MIXED_CATEGORIES: Set[DocumentCategory] = {
    DocumentCategory.PROJECT_SPECIFICATIONS,
}


# ════════════════════════════════════════════════════════════════════════════════
# Per-page data
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class PageInfo:
    """One page of an uploaded file after ingestion."""
    page_number: int                        # 0-based within file
    global_page_number: int                 # 0-based across all files
    extracted_text: str = ""
    has_drawings: bool = False              # heuristic TEXT / DRAWING flag
    image_bytes: Optional[bytes] = None     # raw rendered image (kept in memory)
    image_base64: Optional[str] = None      # base64 for OpenAI vision calls
    categories: Set[DocumentCategory] = field(default_factory=set)  # per-page categories (from chunk classification)

    def ensure_base64(self) -> str:
        """Lazily encode image_bytes → base64 if not already done."""
        if self.image_base64 is None and self.image_bytes is not None:
            self.image_base64 = base64.b64encode(self.image_bytes).decode("utf-8")
        return self.image_base64 or ""


# ════════════════════════════════════════════════════════════════════════════════
# Per-file data
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassifiedFile:
    """A single uploaded file after classification."""
    filename: str
    categories: Set[DocumentCategory] = field(default_factory=set)
    has_visual_content: bool = False
    pages: List[PageInfo] = field(default_factory=list)

    # ── convenience properties ──────────────────────────────────────────────
    @property
    def is_text_only(self) -> bool:
        return not self.has_visual_content

    @property
    def needs_vision(self) -> bool:
        return self.has_visual_content

    @property
    def visual_pages(self) -> List[PageInfo]:
        return [p for p in self.pages if p.has_drawings]

    @property
    def text_pages(self) -> List[PageInfo]:
        return [p for p in self.pages if not p.has_drawings]


# ════════════════════════════════════════════════════════════════════════════════
# Overall result of Step 1
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class DocumentClassificationResult:
    """Full output of the Document Classification step."""
    files: List[ClassifiedFile] = field(default_factory=list)
    raw_pdf_bytes: Dict[str, bytes] = field(default_factory=dict)  # filename → raw PDF bytes

    # ── query helpers ───────────────────────────────────────────────────────
    @property
    def total_files(self) -> int:
        return len(self.files)

    @property
    def total_pages(self) -> int:
        return sum(len(f.pages) for f in self.files)

    def get_files_by_category(self, cat: DocumentCategory) -> List[ClassifiedFile]:
        return [f for f in self.files if cat in f.categories]

    def get_pages_by_category(self, cat: DocumentCategory) -> List[PageInfo]:
        """Return pages that belong to a category.

        Uses page-level categories when available (chunked files),
        falls back to file-level categories (small files).
        """
        pages: List[PageInfo] = []
        for f in self.files:
            for p in f.pages:
                if p.categories:
                    # Page has its own categories (from chunk classification)
                    if cat in p.categories:
                        pages.append(p)
                else:
                    # Fall back to file-level categories (non-chunked files)
                    if cat in f.categories:
                        pages.append(p)
        return pages

    def get_text_for_category(self, cat: DocumentCategory) -> str:
        return "\n".join(
            p.extracted_text
            for p in self.get_pages_by_category(cat)
            if p.extracted_text
        )

    def get_visual_pages_for_category(self, cat: DocumentCategory) -> List[PageInfo]:
        return [p for p in self.get_pages_by_category(cat) if p.has_drawings]

    def get_text_pages_for_category(self, cat: DocumentCategory) -> List[PageInfo]:
        return [p for p in self.get_pages_by_category(cat) if not p.has_drawings]

    def summary(self) -> str:
        lines = [f"Classification Result: {self.total_files} files, {self.total_pages} pages"]
        for f in self.files:
            cats = ", ".join(sorted(c.value for c in f.categories)) or "UNCLASSIFIED"
            tp = len(f.text_pages)
            vp = len(f.visual_pages)
            lines.append(f"  {f.filename}: [{cats}] — {len(f.pages)} pages (text={tp}, drawing={vp})")
        return "\n".join(lines)
