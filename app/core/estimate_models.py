"""Data models for the new extraction pipeline.

Structures:
  - SheetInfo           — one drawing sheet with metadata + extracted data
  - ExtractedTable      — one table found on a page via PyMuPDF find_tables()
  - ExtractedScheduleRow — one row from a parsed schedule table
  - EstimateItem        — universal output item (every extraction → one of these)
  - ExtractionResult    — full pipeline output
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtractedTable:
    """One table found on a drawing page via PyMuPDF find_tables()."""
    page_number: int = 0                # global page number
    sheet_id: str = ""
    schedule_type: str = ""             # "door"|"window"|"finish"|"fixture"|"equipment"|"panel"|"lighting"|"hardware"|"unknown"
    headers: List[str] = field(default_factory=list)
    rows: List[Dict[str, str]] = field(default_factory=list)  # {header: cell_value}
    confidence: float = 0.0             # parse quality: cell fill rate, header detection
    table_title: str = ""               # e.g., "DOOR SCHEDULE"
    filter_type: str = "detected"       # "detected" | "uploaded_file"


@dataclass
class ExtractedScheduleRow:
    """One row from a parsed schedule table."""
    schedule_type: str = ""
    row_data: Dict[str, str] = field(default_factory=dict)  # {column_header: cell_value}
    mark: str = ""                      # primary identifier (door mark "103A", room "201", fixture "WC-1")
    page_number: int = 0
    sheet_id: str = ""


@dataclass
class SheetInfo:
    """One drawing sheet identified from the sheet index."""
    sheet_id: str = ""                  # "A9.00", "AD1.10", "E1.00"
    title: str = ""                     # "Schedules and Details", "Demolition Key Plans"
    discipline: str = ""                # "Architectural", "Electrical", etc.
    global_page_number: int = 0         # maps to PageInfo
    source_file: str = ""
    extracted_text: str = ""
    image_bytes: Optional[bytes] = None
    tables: List[ExtractedTable] = field(default_factory=list)


@dataclass
class EstimateItem:
    """Universal output — every extracted quantity becomes one of these."""
    trade: str = ""
    item_description: str = ""
    qty: Optional[float] = None         # None if measurement required
    unit: str = ""                      # EA, LF, SF, SY, CY, etc.

    # Provenance
    extraction_method: str = ""         # "schedule_parse" | "text_explicit" | "vision_count" |
                                        # "vision_dimension" | "spec_stated" | "keynote"
    confidence: float = 0.0             # 0.0–1.0
    source_file: str = ""
    source_page: int = 0
    sheet_id: str = ""

    # Material
    material_spec: str = ""
    spec_section: str = ""              # CSI section if known

    # Review flags
    needs_measurement: bool = False     # qty depends on scaled geometry
    needs_counting: bool = False        # qty depends on symbol counting from plans
    counting_target: str = ""           # what to count: "F-1 troffers", "duplex receptacles"
    counting_source_pages: List[int] = field(default_factory=list)  # which plan pages to count on
    needs_field_verification: bool = False  # "V.I.F.", "field verify", "as required"
    review_reason: str = ""

    # Linking
    schedule_mark: str = ""             # links to ExtractedScheduleRow.mark
    notes: str = ""

    # Source provenance
    source: str = ""                    # "schedule:<name>", "plan_count:<sheet_id>",
                                        # "plan_measurement:<sheet_id>", "detail:<sheet_id>",
                                        # "elevation:<sheet_id>", "note"


@dataclass
class ExtractionResult:
    """Full pipeline output."""
    sheets: List[SheetInfo] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    items: List[EstimateItem] = field(default_factory=list)

    # Stats
    total_items: int = 0
    auto_accepted: int = 0              # confidence >= threshold
    needs_review: int = 0               # confidence < threshold
    items_by_trade: Dict[str, int] = field(default_factory=dict)
    items_by_method: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Extraction Result: {self.total_items} items",
            f"  Auto-accepted: {self.auto_accepted} ({self._pct(self.auto_accepted)}%)",
            f"  Needs review:  {self.needs_review} ({self._pct(self.needs_review)}%)",
            f"  Sheets: {len(self.sheets)}, Tables: {len(self.tables)}",
        ]
        if self.items_by_method:
            lines.append("  By method:")
            for method, count in sorted(self.items_by_method.items()):
                lines.append(f"    {method}: {count}")
        if self.items_by_trade:
            lines.append("  By trade:")
            for trade, count in sorted(self.items_by_trade.items()):
                lines.append(f"    {trade}: {count}")
        return "\n".join(lines)

    def _pct(self, n: int) -> int:
        return n * 100 // max(self.total_items, 1)
