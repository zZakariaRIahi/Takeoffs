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
# Scope boundary dataclass
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class ScopeBoundary:
    """Scope constraints extracted from Invitation to Bid / bid instructions.

    Used to guide extraction (prevent pulling excluded scope from spec sections)
    and as a post-extraction filter (drop anything that slipped through).
    """
    in_scope: List[str] = field(default_factory=list)    # explicit inclusions
    excluded: List[str] = field(default_factory=list)    # explicit exclusions / NIC
    alternates: List[str] = field(default_factory=list)  # alternates to price separately
    notes: List[str] = field(default_factory=list)       # other pricing guidance
    source: str = ""                                     # which file(s) produced this


# ════════════════════════════════════════════════════════════════════════════════
# Prompts
# ════════════════════════════════════════════════════════════════════════════════

SPECS_ONLY_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR preparing a complete BILL OF MATERIALS
for a construction bid. Read EVERY section of the provided documents: scope
summaries, all CSI spec sections, bid forms, addenda, environmental reports,
appendices, and any embedded drawing pages.

Your output must go beyond scope-level line items. For every scope operation,
produce BOTH the installation labor line AND all material, accessory, and
consumable line items needed to complete it — each as its own JSON entry.

═══════════════════════════════════════════════════════════════
STEP 1 — EXTRACT QUANTITIES LIKE AN ESTIMATOR
═══════════════════════════════════════════════════════════════

Quantities are EVERYWHERE — not just in explicit numbers:

1. EXPLICIT QUANTITIES — stated directly:
   "Paint walls (approx 35,000 sf)" → qty: 35000, unit: SF
   "Provide 12 hollow metal doors" → qty: 12, unit: EA
   "Install 700 LF of rubber base" → qty: 700, unit: LF

2. COUNT LOCATIONS/ROOMS:
   "Remove casework in classrooms A300 and B300" → qty: 2, unit: EA
   "Install sinks in Rooms 101, 102, 103, 104" → qty: 4, unit: EA
   Count every room, location, or instance named.

3. COUNT ITEMS IN LISTS:
   "Remove: 3 water closets, 2 urinals, 4 lavatories" → 3 separate items
   "Classrooms A100, A101, A102, B100, B101, B102" → qty: 6 classrooms

4. DERIVE FROM CONTEXT:
   "Remove plumbing sinks in classrooms A300 and B300" → qty: 2 EA
   "Fire extinguisher in each stairwell" + 4 stairwells → qty: 4 EA

5. SPLIT COMPOUND SCOPE — demolition and new work are ALWAYS separate:
   "Remove and replace VCT (5,000 sf) including rubber base (700 lf)"
   → Remove VCT: 5000 SF | Remove base: 700 LF | Install VCT: 5000 SF | Install base: 700 LF

6. CALCULATE FROM DIMENSIONS:
   "Paint walls in 32 classrooms, avg 1,100 SF per room" → qty: 35200 SF
   "Install base in 6 classrooms, 120 LF perimeter each" → qty: 720 LF

7. LUMP SUM — ONLY when quantity genuinely cannot be derived:
   "Daily cleanup" → 1 LS    "Temporary hoisting" → 1 LS
   DO NOT use LS when rooms or locations are listed.

8. ALTERNATES — prefix description with "ALTERNATE X:"
9. ALLOWANCES — extract with dollar amount: qty: 5000, unit: DLRS
10. NULL — set qty: null ONLY when truly unknowable; explain in review field.
11. ASSUMED QUANTITIES — if you are counting items that are NOT named in a
    list, schedule, or explicit statement (e.g., you are guessing how many
    tack boards are in a hallway, or inferring a count from a photo
    description), you MUST flag the item as ALLOWANCE with confidence
    0.45–0.60 and explain your assumption in the review field. Never present
    an assumed count as if it were extracted directly from the documents.
12. FBO / OWNER-FURNISHED ITEMS — items designated "Furnished by Owner"
    (FBO), "Owner-Furnished / Contractor-Installed" (OFCI), or "NIC"
    (Not in Contract for supply):
    → prefix description with "INSTALL ONLY — "
    → material_spec: "Furnished by Owner"
    → confidence: same as equivalent installation work
    DO NOT omit FBO items — they represent real installation labor that
    must be priced.
13. EXISTING ITEMS — items marked "(E)", "(Ex)", "existing to remain",
    "existing to be reused", or "existing — no work":
    → DO NOT create a supply or installation item for them
    → If the spec says "protect existing X" → add a PROTECTION item (LS)
    → If the spec says "remove existing X" → add a DEMOLITION item only

QUANTITY PRIORITY — always prefer explicit over calculated:
  1. Spec states qty explicitly → use it (highest authority)
  2. Drawings provide dimension → measure and calculate
  3. Spec + drawing together → use drawing dimension × spec coverage rate
  4. Neither → estimate from context with low confidence and ALLOWANCE note

═══════════════════════════════════════════════════════════════
STEP 2 — EXTRACT MATERIALS FOR EVERY SCOPE ITEM
═══════════════════════════════════════════════════════════════

After extracting each labor/installation line, add the materials needed.
Apply coverage rates below to calculate quantities. Show math in method field.
Only extract trades present in this project — skip the rest.

── DEMOLITION ───────────────────────────────────────────────────
  Demolition items are LABOR + DISPOSAL ONLY — no new material supply.
  Always extract separately from new installation work (Rule 5).
  → Selective partition / wall demo: SF or LF (note material: CMU, drywall, plaster)
  → Floor covering removal (VCT, carpet, tile, wood flooring): SF
  → Ceiling system removal (ACT grid + tiles, drywall ceiling): SF
  → Plumbing fixture removal: EA
  → HVAC equipment / unit removal: EA
  → Electrical fixture / panel / conduit removal: EA
  → Door and window removal: EA
  → Concrete sawcutting: LF
  → Debris hauling and disposal: CY or LS
  → Dumpster / rolloff container rental: EA × weeks or LS
  → Dust / contamination protection — temporary containment barriers: SF or LS

── CONCRETE ─────────────────────────────────────────────────────
  → Concrete (state mix design per spec, e.g., 4000 psi): CY
  → Formwork (slabs, walls, columns): SF contact area + 10% waste
  → Rebar / reinforcing steel: TON — standard density:
      slabs: 0.6–1.0 lb/SF × SF ÷ 2000; beams/columns: 3–5 lb/SF × SF ÷ 2000
  → Wire mesh / WWF: SF + 10% (if specified instead of rebar)
  → Vapor retarder under slab: SF + 15%
  → Curing compound: GAL — SF ÷ 200 SF/gal
  → Control joint sealant: LF
  → Epoxy dowels / mechanical anchors at connections: EA if noted in spec

── PAINTING ─────────────────────────────────────────────────────
  SPLIT by substrate — CMU and gypsum board require different systems:
  CMU substrate:
  → Block filler (coat 1): GAL — SF ÷ 200 SF/gal + 5% waste
  → Finish coat (coat 2): GAL — SF ÷ 350 SF/gal + 5% waste
  Gypsum board / drywall substrate:
  → Primer (coat 1): GAL — SF ÷ 300 SF/gal + 5% waste
  → Finish coat (coat 2): GAL — SF ÷ 350 SF/gal + 5% waste
  All painted surfaces:
  → Roller covers, brushes, trays: 1 LS
  → Masking tape and plastic sheeting: 1 LS
  → Surface patching compound / spackling: 1 LS if spec requires prep
  → RRP lead-safe setup (poly sheeting, HEPA vacuum, disposal bags): 1 LS
    if spec or local code requires lead-safe practices
  Note: if the spec calls for different products on different substrates,
  produce SEPARATE labor + material lines for each substrate type.

── DRYWALL ──────────────────────────────────────────────────────
  → Drywall sheets: EA — SF ÷ 32 SF/sheet (4×8) + 10% waste
  → Joint compound: BAG — 1 bag per 100 SF
  → Drywall tape: LF — SF × 2 (approx)
  → Corner bead: LF of outside corners

── FLOORING ─────────────────────────────────────────────────────
  → Flooring material (VCT, LVT, carpet tile, etc.): SF + 10% waste
  → Adhesive: GAL — SF ÷ 175 SF/gal (or per spec)
  → Floor leveling compound: BAG — if subfloor prep required
  → Transition strips / thresholds: LF at doorways
  → Wall base / cove base: LF of room perimeters

── TILE & SOLID SURFACES ────────────────────────────────────────
  → Tile: SF + 10% waste
  → Thinset mortar: BAG — 40-50 SF per 50 lb bag
  → Grout: BAG — 50-100 SF per 25 lb bag (varies by joint width)
  → Grout sealer: GAL

── MASONRY (brick, block, CMU, tuckpointing) ────────────────────
  → Brick / CMU: EA — modular brick 6.75/SF face; 8" CMU 1.125/SF + 10% waste
  → Setting/rebuilding mortar: BAG — 6-7 SF per 70 lb bag
  → Pointing/repointing mortar: BAG — 40-50 SF per 70 lb bag at 3/4" depth
  → Mortar pigment: 1 LS when color matching required
  → Bonding agent: GAL or LS
  → Setting buttons/shims: 1 LS
  → Masonry cleaner: 1 LS if spec requires final cleaning

── WATERPROOFING & JOINT SEALANTS ──────────────────────────────
  Per sealant location (window perimeters, expansion joints, flashing
  terminations, shelf angles — each as a SEPARATE line):
  → Sealant: EA cartridges (10.1 oz ≈ 25-30 LF at 1/4"×1/4" bead; scale
    for larger beads stated in spec) OR LF if spec gives a total
  → Backer rod (closed-cell): LF — total of all sealant joint LF
  → Sealant primer: 1 LS if spec requires
  → Bond-breaker tape: 1 LS
  → Masking tape: 1 LS
  For sheet membrane waterproofing:
  → Membrane: SF + 10% waste | Primer: GAL | Termination bar: LF of edges

── SHEET METAL FLASHING & TRIM ──────────────────────────────────
  Per flashing location (each SEPARATELY — lintel, shelf angle, parapet, etc.):
  → Flashing material: LF of that run + 5% for laps
  → Termination bar: LF — same as flashing run
  → Drip edge: LF — same as flashing run where exposed
  → End dams: EA — 2 per opening served by that flashing run
  → Fasteners / cleats / blind rivets: 1 LS per flashing system
  → Solder + flux: 1 LS if spec requires soldered joints
  → Synthetic underlayment / slip sheet: SF where flashing contacts
    cementitious substrate (run width × LF)
  → Bituminous coating: 1 LS if spec requires on concealed face
  → Weep vents / weep holes: EA — flashing LF ÷ spacing (use spec spacing;
    note which spacing you used and flag any spec vs. drawing conflict)

── ROUGH CARPENTRY ──────────────────────────────────────────────
  → Lumber: LF by size (2×4, 2×6, LVL) + 15% waste
  → Sheathing/plywood: SF + 10% waste
  → Fasteners (nails, screws, bolts): LB or LS
  → Metal connectors (joist hangers, post caps): EA

── FINISH CARPENTRY & MILLWORK ──────────────────────────────────
  (Visible, finished woodwork — separate from Rough Carpentry.)
  → Door casing / trim sets (both faces of opening): LF or EA
      LF per opening = 2 × door height + door width; multiply by opening count
  → Base trim / base cap (wood): LF — perimeter of rooms with wood base
      (separate from rubber base in Flooring)
  → Window stool and apron: LF per window
  → Chair rail: LF of total run if spec requires
  → Wainscoting / wood wall paneling: SF if spec requires
  → Finish fasteners (brad nails, finish screws): 1 LS
  → Caulk and wood filler for painted trim: 1 LS
  → Backing / blocking in drywall for trim attachment: LF if required by spec

── DOORS AND WINDOWS ────────────────────────────────────────────
  Per door from schedule (or estimated count):
  → Door/window unit: EA
  → Frame / buck: EA or LF if separate
  → Hardware set: EA — (hinges, lockset, closer, threshold, stops)
  → Weatherstripping: LF
  → Perimeter sealant: LF

── INSULATION ───────────────────────────────────────────────────
  → Batt insulation: SF | Rigid insulation: SF + 5% | Spray foam: BF
  → Vapor barrier: SF + 10%

── ACOUSTICAL CEILINGS ──────────────────────────────────────────
  → Ceiling tiles (2×2 or 2×4 per spec): SF + 5% waste
  → Main tees: LF — grid SF ÷ 2 ft spacing + 5%
  → Cross tees (2-ft): LF — grid SF ÷ 2 ft + 5%
  → Wall angle / trim molding: LF — perimeter of all rooms with ACT
  → Hanger wire: LF — 1 hanger per 4 SF of grid (ASTM C635) × avg 4 ft drop
  → Concrete inserts / toggle bolts for hangers: EA — 1 per hanger
  If spec requires seismic bracing:
  → Diagonal seismic brace wires: LF — 4 per 144 SF of grid area

── ROOFING ──────────────────────────────────────────────────────
  → Membrane / shingles: SQ (100 SF) + 10% waste
  → Underlayment: SQ | Fasteners: LS
  → Flashing at penetrations: EA or LF | Edge metal / drip edge: LF

── PLUMBING ─────────────────────────────────────────────────────
  → Fixtures: EA (from count)
  → Supply pipe: LF by size | Drain/waste/vent pipe: LF by size
  → Fittings: 1 LS | Shutoff valves: EA (one per supply connection)
  → Escutcheons: EA (one per pipe penetration visible)
  Drinking fountains / bottle fillers — extract as SEPARATE items per unit:
  → Drinking fountain / bottle filler combo unit: EA
  → Bottle filler assembly (if separate from fountain): EA
  → Back panel / mounting bracket: EA per unit
  → Carrier / floor support (if freestanding or semi-recessed): EA
  → Cane detection apron (ADA requirement): EA per ADA-accessible unit
  → Hardwired sensor / electrical connection: EA if spec requires
  → Trap and waste fitting: EA per unit
  → Shutoff valve at supply: EA per unit
  → Escutcheon plates at wall/floor penetrations: EA

── ELECTRICAL ───────────────────────────────────────────────────
  → Conduit: LF by size | Wire: LF by gauge
  → Devices (outlets, switches, fixtures): EA
  → Panel / breaker: EA if new circuits | Junction boxes: EA

── HVAC & SHEET METALS ──────────────────────────────────────────
  → Equipment units: EA | Ductwork: LF or SF sheet metal
  → Diffusers / grilles / registers: EA | Insulation on duct: LF

── CABINETS / LOCKERS / CASEWORK ────────────────────────────────
  → Units: EA (rooms × units per room — count by UNIT not compartment)
  → Countertop: LF or SF | Hardware per unit: EA (if purchased separately)
  → Blocking / backing: LF or LS | Fasteners: 1 LS
  Lockers — also extract these spec-driven sub-items:
  → ADA / accessible lockers: EA — spec typically requires 5% of total
    locker count to be ADA-accessible; calculate and extract separately
  → RFID / electronic locks for ADA lockers: EA — one per ADA locker
    if spec requires contactless or RFID hardware for accessible units
  → RFID media / key cards / fobs for ADA lockers: EA — spec often
    requires 2 per ADA locker; extract as separate line
  → Standard combination or key locks: EA — one per standard locker unit
    if spec requires furnished locks
  → Number / name plate frames: EA — one per locker opening (openings =
    units × tiers; e.g., 90 double-tier units = 180 openings = 180 plates)
  → Flush panel closures: EA — required where locker end panels or top
    panels are exposed (corners, end-of-row); count from layout if available
  → Filler strips: EA or LF — required at gaps between lockers and walls
    or between locker banks; flag as ALLOWANCE if layout not provided
  → Sloped tops / flat tops: EA — extract if spec distinguishes between
    sloped-top and flat-top configurations

── BATH ACCESSORIES & TOILET PARTITIONS ─────────────────────────
  Toilet partitions / compartments:
  → Toilet compartment units: EA — count from fixture schedule; if not
    available, count WC fixtures from drawings or spec
  → Urinal screens: EA — one per urinal fixture
  → Material per spec: phenolic, HDPE, solid plastic, baked enamel, stainless
  Accessories (CSI 10 28 00 or similar):
  → Grab bars: EA — 2 per ADA-accessible WC stall (1 side + 1 rear wall);
    calculate ADA count from spec or apply 5% of total WCs as minimum
  → Toilet paper holders: EA — 1 per WC stall
  → Paper towel dispensers: EA — 1 per lavatory (or per spec)
  → Hand dryers: EA — if spec specifies instead of dispensers
  → Soap dispensers: EA — 1 per lavatory
  → Mirrors: EA — 1 per lavatory
  → Sanitary napkin dispensers: EA if spec requires (1 per women's toilet room)
  → Coat hooks: EA if spec requires
  → Shower curtain rods + curtains: EA if shower facilities specified
  → Mounting hardware / backing: 1 LS per toilet room

── FIRE SPRINKLERS ──────────────────────────────────────────────
  → Sprinkler heads: EA — use count from drawings if available; otherwise
    calculate from hazard class:
    light hazard (offices, schools): 225 SF/head; ordinary hazard: 130 SF/head
    (per NFPA 13 — flag as ALLOWANCE if total protected area not stated)
  → Escutcheons / trim plates: EA — 1 per head
  → Main distribution pipe (4" or 6"): LF
  → Branch line pipe (1"–2"): LF
  → Pipe hangers: EA — approx 1 per 12 LF of branch pipe
  → Zone control valve / tamper switch: EA per floor or zone
  → Backflow preventer: EA
  → Fire department connection (FDC siamese): EA
  → Acceptance testing and hydraulic calculations: 1 LS

── GENERAL REQUIREMENTS ─────────────────────────────────────────
  INCLUDE (separately priceable items only):
  → Scaffolding / access equipment: MO — derive from contract duration;
    specify type (swing stage, frame, mast climber) from spec
  → Temporary toilet facilities: EA per 20 workers, or MO
  → Temporary hoisting / crane: MO or LS with defined scope
  → Mockup panels: EA with dimensions stated in spec
  → Independent testing / inspections: EA or LS with test type
  → Photographic / video documentation: 1 LS if spec requires

  DO NOT INCLUDE: bid bonds, performance bonds, payment bonds, insurance,
  CAF fees, permits, shop drawings review time, supervision, as-builts,
  O&M manuals, project management overhead.

═══════════════════════════════════════════════════════════════
WHAT TO READ
═══════════════════════════════════════════════════════════════

A. SCOPE SUMMARIES / BID FORMS — most explicit quantities
B. ALL CSI SPEC SECTIONS — materials, standards, coverage rates, accessories
C. DEMOLITION SECTIONS — everything removed (separate from new work)
D. ENVIRONMENTAL/ABATEMENT — asbestos, lead paint, mold with survey areas
E. ADDENDA — scope changes, additions, deletions
F. SCHEDULES — door, finish, equipment schedules
G. EMBEDDED OR PROVIDED DRAWINGS — extract dimensions, counts, locations

═══════════════════════════════════════════════════════════════
CONFIDENCE — VARY IT, NEVER UNIFORM
═══════════════════════════════════════════════════════════════

0.90–0.95  Qty explicitly stated in spec text
0.75–0.85  Calculated from stated dimensions + standard coverage rates
0.60–0.75  Estimated from drawings (measured or counted visible portions)
0.45–0.60  ALLOWANCE — partial info, key assumption stated in method
0.40–0.50  Null qty — dimension not found in any document

═══════════════════════════════════════════════════════════════
OUTPUT FORMAT (JSON array)
═══════════════════════════════════════════════════════════════

{
  "description": "Interior wall paint, finish coat — Annex Classrooms",
  "qty": 11000,
  "unit": "SF",
  "confidence": 0.90,
  "trade": "Painting",
  "spec_section": "09 91 23",
  "source": "specs:Scope of Work",
  "method": "from_specs: Stated in scope summary. 2 coats per spec 09 91 23.",
  "review": "",
  "material_spec": "Eggshell latex, ASTM D3960. Manufacturers: Benjamin Moore, Sherwin-Williams, PPG.",
  "trigger_type": "explicit"
},
{
  "description": "Paint — finish coat gallons, 2 coats at 11,000 SF",
  "qty": 65,
  "unit": "GAL",
  "confidence": 0.80,
  "trade": "Painting",
  "spec_section": "09 91 23",
  "source": "specs:09 91 23",
  "method": "calculated: 11,000 SF x 2 coats / 350 SF/gal = 62.9 gal + 5% waste = 66 GAL.",
  "review": "",
  "material_spec": "Same product as labor line.",
  "trigger_type": "explicit"
}

Fields:
- description: item name + location (e.g., "SS flashing at window lintels", not just "flashing")
- qty: number or null
- unit: EA, SF, LF, SY, CY, LS, BAG, GAL, LB, TON, MO, SQ, BF, DLRS
- confidence: 0.40–0.95 — NEVER assign the same value to every item
- trade: General Requirements, Site Work, Demolition, Concrete, Masonry, Metals,
  Rough Carpentry, Finish Carpentry, Waterproofing, Insulation, Acoustical Ceilings,
  Roofing, Stucco and Siding, Doors and Windows, Drywall, Tile & Solid Surfaces,
  Flooring, Painting, Bath and Accessories, Appliances, Fire Sprinklers,
  Plumbing, HVAC and Sheet Metals, Electrical, Cabinets
- spec_section: CSI section number or "Scope of Work" / "Addendum" / "Environmental"
- source: "specs:XX XX XX" or "drawings:sheet" or "specs:Scope of Work"
- method: show calculation steps — inputs × rate = result. Be explicit.
- review: "" if confident. "ALLOWANCE — [missing info + RFI needed]" if uncertain.
- material_spec: ASTM standard, product name, size, grade, approved manufacturers
  (preserve exact spec language)
- trigger_type: classify HOW this item is required — one of four values:
    "explicit"     — the documents directly say to furnish, install, provide, or replace this item.
                     Use for: definitive bid scope, addendum changes, firm quantities from bid forms.
    "conditional"  — the requirement is gated on a condition that may or may not apply:
                     "if required", "if damaged", "as needed", "where applicable",
                     "if directed by engineer", "at GC's discretion", "if found during demo".
                     Use for: hazmat if-required language, repairs contingent on field conditions,
                     items the spec mentions only as a possibility.
    "inferred"     — no direct instruction exists; you are reasoning that this work must exist
                     (e.g., "demo of existing lockers" inferred because new ones are installed,
                     but the documents never say to remove old ones).
                     Use when: you are adding an item the documents do not mention.
    "referenced"   — the document mentions this work only to exclude it, disclaim it, cross-reference
                     it to another contract, or note it is NIC / by others.
                     Use for: "plumbing is a separate contract", "painting not in scope", "by owner".

TRIGGER TYPE RULES:
- Default to "explicit" only when the scope instruction is unambiguous and unconditional.
- Any "if", "when required", "as needed", "where damaged", "contingent upon", or similar
  hedging language → MUST be "conditional", never "explicit".
- Items you add because they logically must exist (but are unstated) → "inferred".
- Items present in documents only to disclaim, exclude, or redirect → "referenced" — and
  you should generally NOT extract these as bid line items at all.

RULES:
- Each scope operation gets a labor line PLUS separate material/accessory lines
- Demolition and new work: ALWAYS separate line items
- Never default to LS when rooms or locations are named
- Read the ENTIRE document set before extracting
- Include alternates (prefix "ALTERNATE X:"), allowances, addendum changes
- ONE ITEM PER SCOPE: if the same work appears in scope summary AND a CSI
  section, extract ONCE from the most detailed source
- Show all calculation math in the method field
- Return ONLY the JSON array, no commentary
"""

# Prepended to SPECS_ONLY_PROMPT when multiple documents are bundled into one call.
BUNDLED_PREAMBLE = """\
IMPORTANT: Multiple project documents are provided below. Read ALL of them together
and produce ONE unified takeoff list covering the complete project scope.

DOCUMENT AUTHORITY (most authoritative first — later documents supersede earlier ones):
  1. Addenda / amendments — always override the original spec
  2. Walkthrough notes / scope-of-work letters — clarify actual intent
  3. Main project specifications / project manual — base scope
  4. Bid forms — may contain explicit quantities

CROSS-DOCUMENT RULES:
  - If an addendum deletes scope (e.g., "remove ceiling painting from scope"), honor
    that deletion — do NOT include the deleted item.
  - If two documents give conflicting quantities for the same item, use the most
    authoritative document and note the conflict in the "review" field.
  - Do NOT produce duplicate items — if the same physical scope appears in both the
    spec and an addendum, extract it ONCE with the final correct quantity.
  - Each document is labeled with === DOCUMENT: filename === before it begins.

"""

BUNDLE_SIZE_LIMIT = 40 * 1024 * 1024  # 40 MB total — above this, fall back to per-file

ITB_SCOPE_PROMPT = """\
You are reading one or more Invitation to Bid (ITB) or Bid Instructions documents for a construction project.

Your job is to extract the SCOPE BOUNDARY — what the general contractor should and should not price.

Return a single JSON object (not an array):

{
  "in_scope": [
    "Furnish and install 250 new single-tier metal lockers in 2nd and 3rd floor corridors",
    "Remove existing chair rail and wall base — salvage to building engineer"
  ],
  "excluded": [
    "Plumbing work — separate contract, do not price any plumbing",
    "Abatement — by licensed abatement contractor, not in this bid",
    "Specialty gymnasium equipment installation by owner's vendors: backstops, curtain dividers, elevated running track, scoreboards"
  ],
  "alternates": [
    "Alternate 1: Replace 2nd floor west-end drinking fountain with new unit and bottle filler"
  ],
  "notes": [
    "Salvage all removed materials to the building engineer unless noted otherwise",
    "Owner will provide access on weekends only"
  ]
}

═══════════════════════════════════════════════════════════════
EXCLUSION EXTRACTION — READ THIS CAREFULLY
═══════════════════════════════════════════════════════════════

Every exclusion string you write will later be used to filter the BOM. A vague or
overly broad exclusion will accidentally delete legitimate scope. Write each exclusion
to describe EXACTLY what is excluded — no more, no less.

RULE 1 — WHOLE-TRADE EXCLUSIONS
When an entire trade is excluded as a separate contract, state that clearly:
  ✓ "Plumbing — separate contract, do not price any plumbing"
  ✓ "All electrical work — by others, not in this bid"
  ✓ "Abatement — licensed abatement contractor, not this GC"

RULE 2 — LISTS OF SPECIFIC ITEMS (the most common source of error)
Construction ITBs often write: "work by other vendors including X, Y, Z and <trade name>"
where the trade name refers only to the specific items listed, NOT the entire trade.

When you see a trade name (painting, plumbing, electrical, flooring, etc.) appear
inside a list of specific equipment or vendor items, DO NOT write a blanket trade
exclusion. Instead, name only the specific items that are actually excluded.

  Document says: "work by other vendors including backstops, curtain, elevated track,
                  scoreboards and painting"
  WRONG: "painting by others — do not price any painting work"
  RIGHT: "Specialty gymnasium equipment by owner's vendors: backstops, curtain dividers,
          elevated running track, scoreboards, and any painted graphics that are
          physically part of those equipment packages"

  Document says: "plumbing fixtures furnished by owner — drinking fountains, hose bibs"
  WRONG: "plumbing — not in contract"
  RIGHT: "Owner-furnished plumbing fixtures: drinking fountains and hose bibs (supply only
          excluded; installation labor remains in scope)"

  Document says: "electrical by others except low-voltage data cabling"
  WRONG: "electrical work excluded"
  RIGHT: "Line-voltage electrical work — by others; low-voltage data cabling remains in scope"

RULE 3 — PARTIAL EXCLUSIONS
When only part of a trade is excluded, name the boundary precisely:
  ✓ "Painting of existing walls in occupied areas — not in scope; new construction
     surfaces only"
  ✓ "Concrete flatwork beyond building footprint — site work contract, not this bid"
  ✓ "Roofing warranty repairs — owner's maintenance contract"

RULE 4 — OWNER / FBO SUPPLY EXCLUSIONS
When the owner furnishes material but the contractor installs:
  ✓ "Owner-furnished equipment (supply only excluded): [list items]. Installation
     remains in scope."
  Never write these as blanket exclusions of the entire trade.

RULE 5 — ADDENDA DELETIONS
When an addendum removes scope from a prior bid:
  ✓ "Flooring in Room 204 — deleted by Addendum 2"

═══════════════════════════════════════════════════════════════
GENERAL RULES
═══════════════════════════════════════════════════════════════
- Include ALL exclusions and ALL alternates/add-deducts/allowances described
- If a document explicitly states a quantity or scope item, include it in in_scope
- If the ITB is an addendum version that overrides an earlier ITB, its exclusions
  take precedence
- Return ONLY the JSON object, no commentary
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


DEDUP_PROMPT = """\
You are reviewing a list of construction quantity takeoff items extracted from MULTIPLE
documents for the same project. Because different documents may describe the same scope,
there may be duplicate entries.

Your task: consolidate the list by removing true duplicates, keeping the BEST version
of each unique scope item. Do not lose any genuinely distinct work.

═══════════════════════════════════════════════════════════
WHAT IS A DUPLICATE
═══════════════════════════════════════════════════════════

Two items are duplicates if they describe the SAME physical work at the SAME location,
regardless of which document they came from or what trade they were assigned to.

Examples of duplicates (merge these):
- "Furnish and install 250 lockers" + "Install new lockers 250 EA" + "Install new single-tier metal lockers 250 EA" → same physical lockers
- "Remove wooden wall base, null LF" + "Remove existing wall base 250 LF" → same removal work (keep the one with a quantity)
- "Secure permits 1 LS" + "Labor to secure necessary permits 1 LS" → same GC task
- "Abatement by licensed contractor, null LS" + "RRP/abatement work, 1 LS" → same environmental item
- "Move CPS materials 1 LS" + "Move CPS materials 2 EA" → same GC task (keep 1 LS — cannot count moves)

═══════════════════════════════════════════════════════════
WHAT IS NOT A DUPLICATE — NEVER MERGE THESE
═══════════════════════════════════════════════════════════

- Different operations: "Remove 250 lockers" vs "Install 250 lockers" → keep both
- Different locations: "1st floor" vs "2nd floor" → keep both
- ALTERNATE vs non-ALTERNATE: "ALTERNATE: DF 2nd floor" vs "DF 1st floor" → keep both
- Different scope: "Remove chair railing" vs "Remove wall base" → keep both
- Demolition vs new work: always separate items
- Genuinely different quantities for the same trade from different floor/area counts
- A material/accessory sub-item vs its parent installation line: "Install 250 lockers,
  250 EA" vs "RFID locks for ADA lockers, 13 EA" → keep BOTH (sub-items are never
  duplicates of their parent — they are separate physical products)
- Labor line vs material line for the same scope: "Paint walls 11,000 SF" vs
  "Paint — finish coat gallons, 65 GAL" → keep BOTH
- Different accessories for the same assembly: "RFID electronic locks, 13 EA" vs
  "RFID key cards / fobs, 26 EA" → keep BOTH; they are different physical products

═══════════════════════════════════════════════════════════
HOW TO MERGE
═══════════════════════════════════════════════════════════

When merging duplicates:
1. Keep the most detailed, accurate description
2. Prefer a real qty (non-null) over null — if one has qty and the other doesn't, keep the qty
3. Keep the highest confidence value
4. Keep the most complete material_spec
5. Combine sources (e.g., "specs:Project Manual, specs:ITB, specs:Addendum 02")
6. Keep the trade that best reflects the actual work

═══════════════════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════════════════

Return JSON with exactly two keys:
{
  "items": [
    {
      "trade": "...",
      "description": "...",
      "qty": <number or null>,
      "unit": "...",
      "confidence": <number>,
      "source": "...",
      "method": "...",
      "review": "...",
      "material_spec": "..."
    }
  ],
  "merge_log": [
    {
      "kept": "brief description of the item kept",
      "merged": ["description of dropped duplicate 1", "description of dropped duplicate 2"],
      "reason": "brief explanation"
    }
  ]
}

- merge_log must list every merge operation performed
- If no duplicates found, return "merge_log": []
- Return ONLY valid JSON, no other text
"""


BID_FORM_PROMPT = """\
You are a construction estimator reading a bid form document.

Your ONLY task: extract line items that have an EXPLICIT quantity written in the document.
Bid forms often contain a table or list with: item description + quantity + unit.

EXTRACT only items with explicit quantities written in the document, for example:
  "250 Lockers" → qty: 250, unit: EA
  "375 LF Remove Base" → qty: 375, unit: LF
  "1 Drinking Fountain" → qty: 1, unit: EA
  "2,500 SF Flooring" → qty: 2500, unit: SF

DO NOT EXTRACT — return an empty array [] if this is all you find:
- Bid bonds, performance bonds, payment bonds, surety requirements
- Insurance requirements
- MBE/WBE participation goals or reporting
- Walk-through or pre-bid conference attendance
- Project management, supervision, or overhead line items
- Compliance, certification, or documentation requirements
- Any item without an explicit quantity number stated in the document
- Items inferred or assumed from the project name or title

If this document contains ONLY administrative/submission content (instructions, bonding,
MBE/WBE goals, compliance requirements) with no quantity tables, return: []

Return JSON array. For each item with an explicit quantity:
{
  "description": "concise item description",
  "qty": <number>,
  "unit": "EA|SF|LF|SY|CY|LS",
  "trade": "standard trade name",
  "source": "specs:bid_form",
  "method": "from_bid_form: [exact location/text where qty was found]",
  "review": "",
  "material_spec": "any spec details mentioned alongside the item"
}

Return ONLY the JSON array, no other text.
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
        trigger_type = d.get("trigger_type", "explicit") or "explicit"

        # Embed trigger_type as a structured tag at the front of review_reason so it
        # survives serialization without requiring a model schema change.
        if trigger_type != "explicit":
            tag = f"[trigger:{trigger_type}] "
            review = tag + review if review else tag.rstrip()

        if not review and qty is None:
            review = "Qty not stated in specs — verify from drawings or field"

        conf_raw = d.get("confidence")
        if conf_raw is not None:
            try:
                confidence = float(conf_raw)
            except (ValueError, TypeError):
                confidence = 0.75 if qty is not None else 0.45
        else:
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
    extra_text_docs: Optional[List[Tuple[str, str]]] = None,
    scope_boundary: Optional[ScopeBoundary] = None,
) -> List[EstimateItem]:
    """Case 1: No drawings — full takeoff from all project documents in one call.

    Bundles all spec + bid-form PDFs and DOCX text into a single Gemini call so
    the model has full cross-document context (addenda supersede specs, no conflicts).
    Falls back to per-file calls if total size exceeds BUNDLE_SIZE_LIMIT.

    Args:
        classification: Document classification result with raw PDF bytes.
        extra_text_docs: Optional list of (filename, text) for DOCX/text documents.
        scope_boundary: Optional scope boundary from ITB — injected into every extraction
                        prompt so the model avoids generating excluded scope items.
    """
    t0 = time.time()
    client = _get_client()

    # Collect all relevant PDFs (skip environmental surveys)
    all_pdf_files: List[Tuple[str, bytes]] = []
    for cf in classification.files:
        if not cf.categories:
            continue
        # Skip pure environmental surveys (no specs/bid content)
        is_env_only = (
            DocumentCategory.ENVIRONMENTAL_SURVEY in cf.categories
            and DocumentCategory.PROJECT_SPECIFICATIONS not in cf.categories
            and DocumentCategory.BID_FORM not in cf.categories
        )
        if is_env_only:
            logger.info(f"  Skipping pure environmental survey: {cf.filename}")
            continue
        pdf_bytes = classification.raw_pdf_bytes.get(cf.filename)
        if not pdf_bytes:
            continue
        # Include specs and bid-form PDFs — skip files with no relevant categories
        relevant = (
            DocumentCategory.PROJECT_SPECIFICATIONS in cf.categories
            or DocumentCategory.BID_FORM in cf.categories
            or DocumentCategory.CONSTRUCTION_DRAWINGS in cf.categories
        )
        if relevant:
            all_pdf_files.append((cf.filename, pdf_bytes))

    text_docs = extra_text_docs or []

    if not all_pdf_files and not text_docs:
        logger.warning("No specification files found")
        return []

    total_bytes = sum(len(b) for _, b in all_pdf_files)
    logger.info(
        f"  Bundling {len(all_pdf_files)} PDFs ({total_bytes/1024/1024:.1f} MB) "
        f"+ {len(text_docs)} text docs"
    )

    if total_bytes <= BUNDLE_SIZE_LIMIT:
        items = _extract_bundled(client, all_pdf_files, text_docs, t0, scope_boundary)
    else:
        logger.info(f"  Total {total_bytes/1024/1024:.1f} MB > 40 MB — falling back to per-file")
        items = _extract_per_file(client, all_pdf_files, text_docs, t0, scope_boundary)

    logger.info(f"Specs-only extraction: {len(items)} items in {time.time()-t0:.0f}s")
    return items


def _extract_bundled(
    client: genai.Client,
    pdf_files: List[Tuple[str, bytes]],
    text_docs: List[Tuple[str, str]],
    t0: float,
    scope_boundary: Optional[ScopeBoundary] = None,
) -> List[EstimateItem]:
    """Single Gemini call with all documents for full cross-document context."""
    uploaded_files: List[Tuple[str, Any]] = []

    # Upload all PDFs in parallel
    if pdf_files:
        logger.info(f"  Uploading {len(pdf_files)} files in parallel...")
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_upload_pdf, client, b, n): n
                for n, b in pdf_files
            }
            # Preserve original order for document authority labeling
            results: dict = {}
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    uploaded = future.result()
                    if uploaded:
                        results[fname] = uploaded
                    else:
                        logger.error(f"  Upload returned None for {fname}")
                except Exception as e:
                    logger.error(f"  Upload failed for {fname}: {e}")
        # Restore original order
        for fname, _ in pdf_files:
            if fname in results:
                uploaded_files.append((fname, results[fname]))

    if not uploaded_files and not text_docs:
        logger.warning("  No files uploaded successfully and no text docs")
        return []

    # Build contents: scope boundary (if any) + preamble + prompt, then labeled file/text parts
    boundary_block = _scope_boundary_preamble(scope_boundary) if scope_boundary else ""
    prompt = boundary_block + BUNDLED_PREAMBLE + SPECS_ONLY_PROMPT
    contents: List[Any] = [prompt]
    for fname, uploaded in uploaded_files:
        contents.append(f"\n=== DOCUMENT: {fname} ===")
        contents.append(uploaded)
    for fname, text in text_docs:
        contents.append(f"\n=== DOCUMENT: {fname} ===\n{text}")

    logger.info(
        f"  Bundled call: {len(uploaded_files)} PDFs + {len(text_docs)} text docs"
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
            ),
        )
        parsed = _parse_json_response(response.text or "")
        if isinstance(parsed, list):
            items = _items_from_parsed(parsed)
            logger.info(f"  Bundled extraction: {len(items)} items in {time.time()-t0:.0f}s")
            return items
        else:
            logger.error("  Bundled extraction: unexpected response format — falling back to per-file")
            return _extract_per_file(client, [], text_docs, t0, scope_boundary)
    except Exception as e:
        logger.error(f"  Bundled extraction failed: {e} — falling back to per-file")
        return _extract_per_file(client, [], text_docs, t0, scope_boundary)
    finally:
        for _, uploaded in uploaded_files:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass


def _extract_per_file(
    client: genai.Client,
    pdf_files: List[Tuple[str, bytes]],
    text_docs: List[Tuple[str, str]],
    t0: float,
    scope_boundary: Optional[ScopeBoundary] = None,
) -> List[EstimateItem]:
    """Fallback: process each document separately (loses cross-doc context)."""
    all_items: List[EstimateItem] = []
    boundary_block = _scope_boundary_preamble(scope_boundary) if scope_boundary else ""

    for fname, pdf_bytes in pdf_files:
        logger.info(f"  Per-file: {fname} ({len(pdf_bytes)/1024/1024:.1f} MB)")
        uploaded = _upload_pdf(client, pdf_bytes, fname)
        if not uploaded:
            continue
        try:
            prompt = boundary_block + SPECS_ONLY_PROMPT
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt, uploaded],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
                ),
            )
            parsed = _parse_json_response(response.text or "")
            if isinstance(parsed, list):
                items = _items_from_parsed(parsed)
                all_items.extend(items)
                logger.info(f"  {fname}: {len(items)} items")
            else:
                logger.error(f"  {fname}: unexpected format")
        except Exception as e:
            logger.error(f"  {fname}: failed: {e}")
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    for fname, text in text_docs:
        logger.info(f"  Per-file text: {fname} ({len(text)} chars)")
        prompt = boundary_block + SPECS_ONLY_PROMPT + f"\n\n=== DOCUMENT: {fname} ===\n{text}"
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
                ),
            )
            parsed = _parse_json_response(response.text or "")
            if isinstance(parsed, list):
                items = _items_from_parsed(parsed)
                all_items.extend(items)
                logger.info(f"  {fname}: {len(items)} items")
        except Exception as e:
            logger.error(f"  {fname}: text extraction failed: {e}")

    return all_items


# ════════════════════════════════════════════════════════════════════════════════
# ITB scope boundary extraction
# ════════════════════════════════════════════════════════════════════════════════

def _scope_boundary_preamble(boundary: ScopeBoundary) -> str:
    """Format a ScopeBoundary as a text block to prepend to extraction prompts."""
    lines = [
        "═══════════════════════════════════════════════════════════════",
        "SCOPE BOUNDARY — extracted from Invitation to Bid",
        "═══════════════════════════════════════════════════════════════",
    ]
    if boundary.in_scope:
        lines.append("IN SCOPE — extract and price these items:")
        for item in boundary.in_scope:
            lines.append(f"  • {item}")
    if boundary.excluded:
        lines.append("")
        lines.append("EXCLUDED — do NOT extract these as bid line items:")
        lines.append("  If a specification section covers excluded scope, skip it entirely.")
        for item in boundary.excluded:
            lines.append(f"  ✗ {item}")
    if boundary.alternates:
        lines.append("")
        lines.append("ALTERNATES — extract separately, prefix description with 'ALTERNATE N:':")
        for item in boundary.alternates:
            lines.append(f"  • {item}")
    if boundary.notes:
        lines.append("")
        lines.append("NOTES:")
        for note in boundary.notes:
            lines.append(f"  • {note}")
    lines.append("═══════════════════════════════════════════════════════════════")
    lines.append("")
    return "\n".join(lines)


def extract_scope_boundary(
    classification: DocumentClassificationResult,
) -> Optional[ScopeBoundary]:
    """Read ITB/bid instructions files and extract the scope boundary.

    Finds all files classified as INSTRUCTIONS_TO_BIDDER, uploads them together
    in a single Gemini call so addenda can override earlier versions, and returns
    a ScopeBoundary with explicit inclusions, exclusions, alternates, and notes.

    Returns None if no ITB files are found.
    """
    itb_files = []
    addendum_files = []
    for cf in classification.files:
        if not cf.categories:
            continue
        pdf_bytes = classification.raw_pdf_bytes.get(cf.filename)
        if not pdf_bytes:
            continue
        if DocumentCategory.INSTRUCTIONS_TO_BIDDER in cf.categories:
            itb_files.append((cf.filename, pdf_bytes))
        elif DocumentCategory.ADDENDUM in cf.categories:
            # Addenda that are not also ITB — scope change documents that may override exclusions
            addendum_files.append((cf.filename, pdf_bytes))

    if not itb_files and not addendum_files:
        logger.info("  No ITB or addendum files found — skipping scope boundary extraction")
        return None

    # ITB files first (base scope), then addenda (overrides) — order matters for the model
    all_boundary_files = itb_files + addendum_files
    logger.info(
        f"  Extracting scope boundary from {len(all_boundary_files)} file(s) "
        f"({len(itb_files)} ITB, {len(addendum_files)} addenda): "
        + ", ".join(n for n, _ in all_boundary_files)
    )

    client = _get_client()
    uploaded: list = []

    try:
        for fname, pdf_bytes in all_boundary_files:
            f = _upload_pdf(client, pdf_bytes, fname)
            if f:
                uploaded.append((fname, f))

        if not uploaded:
            logger.error("  ITB/addendum upload failed — no scope boundary")
            return None

        contents: list = [ITB_SCOPE_PROMPT]
        for fname, f in uploaded:
            contents.append(f"\n=== DOCUMENT: {fname} ===")
            contents.append(f)

        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=8192),
            ),
        )

        parsed = _parse_json_response(response.text or "")
        if not isinstance(parsed, dict):
            logger.error("  ITB scope boundary: unexpected format")
            return None

        boundary = ScopeBoundary(
            in_scope=parsed.get("in_scope") or [],
            excluded=parsed.get("excluded") or [],
            alternates=parsed.get("alternates") or [],
            notes=parsed.get("notes") or [],
            source=", ".join(n for n, _ in all_boundary_files),
        )

        logger.info(
            f"  Scope boundary: {len(boundary.in_scope)} in-scope, "
            f"{len(boundary.excluded)} excluded, {len(boundary.alternates)} alternates"
        )
        if boundary.excluded:
            for ex in boundary.excluded:
                logger.info(f"    EXCLUDED: {ex}")

        return boundary

    except Exception as e:
        logger.error(f"  Scope boundary extraction failed: {e}")
        return None
    finally:
        for _, f in uploaded:
            try:
                client.files.delete(name=f.name)
            except Exception:
                pass


# ════════════════════════════════════════════════════════════════════════════════
# Bid form extraction
# ════════════════════════════════════════════════════════════════════════════════

def extract_from_bid_forms(
    classification: DocumentClassificationResult,
) -> List[EstimateItem]:
    """Extract quantities from bid form PDFs.

    Targets files classified as BID_FORM but NOT PROJECT_SPECIFICATIONS — those
    are already handled by the specs extractor and would produce duplicates.

    Bid forms contain explicit tables (item name + quantity + unit) that the
    specs extractor misses because it only processes PROJECT_SPECIFICATIONS files.
    """
    t0 = time.time()
    client = _get_client()

    bid_files = []
    for cf in classification.files:
        if not cf.categories:
            continue
        if DocumentCategory.BID_FORM not in cf.categories:
            continue
        # Skip if also classified as specs — specs extractor already covers it
        if DocumentCategory.PROJECT_SPECIFICATIONS in cf.categories:
            logger.info(f"  Bid form {cf.filename} also has specs — skipping (handled by specs extractor)")
            continue
        pdf_bytes = classification.raw_pdf_bytes.get(cf.filename)
        if pdf_bytes:
            bid_files.append((cf.filename, pdf_bytes))

    if not bid_files:
        return []

    all_items: List[EstimateItem] = []

    for fname, pdf_bytes in bid_files:
        logger.info(f"Extracting from bid form: {fname} ({len(pdf_bytes)/1024/1024:.1f} MB)")

        uploaded = _upload_pdf(client, pdf_bytes, fname)
        if not uploaded:
            continue

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[BID_FORM_PROMPT, uploaded],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=16384),
                ),
            )

            parsed = _parse_json_response(response.text or "")
            if isinstance(parsed, list):
                items = _items_from_parsed(parsed)
                all_items.extend(items)
                logger.info(f"  {fname}: {len(items)} items from bid form")
            else:
                logger.error(f"  {fname}: unexpected response format")

        except Exception as e:
            logger.error(f"  {fname}: bid form extraction failed: {e}")
        finally:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass

    elapsed = time.time() - t0
    logger.info(f"Bid form extraction: {len(all_items)} items in {elapsed:.0f}s")
    return all_items


# ════════════════════════════════════════════════════════════════════════════════
# Text scope extraction (DOCX / walkthrough documents)
# ════════════════════════════════════════════════════════════════════════════════

def extract_from_text_scope(
    text: str,
    filename: str = "document",
) -> List[EstimateItem]:
    """Extract scope items from plain text (DOCX walkthrough details, SOW docs).

    Used when a document cannot be uploaded as PDF — e.g. a .docx converted to
    text.  The same SPECS_ONLY_PROMPT is used; the text is injected directly into
    the prompt rather than via the Files API.

    Args:
        text: Full extracted text of the document
        filename: Original filename (used in source fields)

    Returns:
        List of EstimateItem
    """
    if not text.strip():
        return []

    t0 = time.time()
    client = _get_client()
    logger.info(f"Text scope extraction: {filename} ({len(text)} chars)")

    prompt = SPECS_ONLY_PROMPT + f"\n\n=== DOCUMENT: {filename} ===\n{text}"

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[prompt],
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=32768),
                ),
            )
            break
        except Exception as e:
            err_str = str(e)
            if attempt < 2 and ("503" in err_str or "UNAVAILABLE" in err_str or "SSL" in err_str or "EOF" in err_str):
                wait = 15 * (attempt + 1)
                logger.warning(f"  [{filename}] Attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  [{filename}] Text extraction failed: {e}")
                return []

    parsed = _parse_json_response(response.text or "")
    items = _items_from_parsed(parsed) if isinstance(parsed, list) else []

    elapsed = time.time() - t0
    logger.info(f"  {filename}: {len(items)} items in {elapsed:.0f}s")
    return items


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
        if cf.categories and DocumentCategory.ENVIRONMENTAL_SURVEY in cf.categories:
            logger.info(f"  Skipping environmental survey: {cf.filename}")
            continue
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
# Post-processing deduplication
# ════════════════════════════════════════════════════════════════════════════════

def deduplicate_items(items: List[EstimateItem]) -> List[EstimateItem]:
    """Consolidate duplicate items extracted from multiple documents.

    Sends the full item list to Gemini which identifies semantically equivalent
    items across different source files and merges them, keeping the best version
    of each. Logs every merge operation for auditability.

    Safe by design: on any failure returns the original list unchanged.
    """
    if len(items) <= 1:
        return items

    client = _get_client()

    items_data = [
        {
            "trade": item.trade,
            "description": item.item_description,
            "qty": item.qty,
            "unit": item.unit,
            "confidence": item.confidence,
            "source": item.source,
            "method": item.extraction_method,
            "review": item.review_reason,
            "material_spec": item.material_spec,
        }
        for item in items
    ]

    prompt = (
        DEDUP_PROMPT
        + f"\n\nHere are the {len(items)} items to consolidate:\n\n"
        + json.dumps(items_data, indent=2)
    )

    logger.info(f"Deduplicating {len(items)} items...")

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=16384),
            ),
        )

        parsed = _parse_json_response(response.text or "")

        if isinstance(parsed, dict):
            items_raw = parsed.get("items", [])
            merge_log = parsed.get("merge_log", [])

            if merge_log:
                logger.info(f"  Dedup merged {len(merge_log)} group(s):")
                for entry in merge_log:
                    logger.info(f"    KEPT:    {entry.get('kept', '?')}")
                    for dropped in entry.get("merged", []):
                        logger.info(f"    DROPPED: {dropped}")
                    if entry.get("reason"):
                        logger.info(f"    Reason:  {entry['reason']}")
            else:
                logger.info("  Dedup: no duplicates found")

            if isinstance(items_raw, list) and items_raw:
                deduped = _items_from_parsed(items_raw)
                logger.info(f"  Dedup result: {len(items)} → {len(deduped)} items")
                return deduped

        elif isinstance(parsed, list) and parsed:
            deduped = _items_from_parsed(parsed)
            logger.info(f"  Dedup result: {len(items)} → {len(deduped)} items (no merge_log)")
            return deduped

        logger.warning("  Dedup returned unexpected format — using original list")

    except Exception as e:
        logger.error(f"  Dedup failed: {e} — returning original list unchanged")

    return items


# ════════════════════════════════════════════════════════════════════════════════
# Post-processing: collapse conditional items into ALLOWANCE lines
# ════════════════════════════════════════════════════════════════════════════════

def collapse_conditionals(items: List[EstimateItem]) -> List[EstimateItem]:
    """Collapse items tagged [trigger:conditional] or [trigger:inferred] into ALLOWANCE lines.

    Items tagged with trigger_type "conditional" (if required, if damaged, as needed)
    or "inferred" (model-assumed, unstated in documents) are grouped by trade and
    collapsed into a single ALLOWANCE line per trade rather than individual line items.
    This prevents hallucinated scope from inflating the BOM.

    Items tagged "referenced" are dropped entirely — they are NIC / by others.
    Items tagged "explicit" (or with no tag) pass through unchanged.
    """
    def _get_trigger(item: EstimateItem) -> str:
        review = item.review_reason or ""
        if review.startswith("[trigger:conditional]"):
            return "conditional"
        if review.startswith("[trigger:inferred]"):
            return "inferred"
        if review.startswith("[trigger:referenced]"):
            return "referenced"
        return "explicit"

    explicit_items: List[EstimateItem] = []
    conditional_by_trade: Dict[str, List[EstimateItem]] = {}
    inferred_by_trade: Dict[str, List[EstimateItem]] = {}
    dropped_referenced: List[str] = []

    for item in items:
        trigger = _get_trigger(item)
        if trigger == "referenced":
            dropped_referenced.append(item.item_description)
        elif trigger == "conditional":
            conditional_by_trade.setdefault(item.trade, []).append(item)
        elif trigger == "inferred":
            inferred_by_trade.setdefault(item.trade, []).append(item)
        else:
            explicit_items.append(item)

    if dropped_referenced:
        logger.info(
            f"  collapse_conditionals: dropped {len(dropped_referenced)} 'referenced/NIC' items: "
            + ", ".join(dropped_referenced[:5])
            + ("..." if len(dropped_referenced) > 5 else "")
        )

    allowance_items: List[EstimateItem] = []

    for trade, group in conditional_by_trade.items():
        descs = "; ".join(i.item_description for i in group[:6])
        if len(group) > 6:
            descs += f" … +{len(group) - 6} more"
        allowance_items.append(EstimateItem(
            trade=trade,
            item_description=f"ALLOWANCE — Conditional {trade} scope (verify in field)",
            qty=None,
            unit="LS",
            extraction_method="collapsed_conditionals",
            confidence=0.45,
            source=group[0].source if group else "specs",
            schedule_mark="",
            material_spec="",
            spec_section=group[0].spec_section if group else "",
            review_reason=(
                f"ALLOWANCE — {len(group)} conditional item(s) collapsed. "
                f"Originals (all gated on 'if required'/'if damaged'/'as needed'): {descs}. "
                "Verify conditions in field before pricing."
            ),
        ))
        logger.info(
            f"  collapse_conditionals: collapsed {len(group)} conditional {trade} items → 1 ALLOWANCE"
        )

    for trade, group in inferred_by_trade.items():
        descs = "; ".join(i.item_description for i in group[:6])
        if len(group) > 6:
            descs += f" … +{len(group) - 6} more"
        allowance_items.append(EstimateItem(
            trade=trade,
            item_description=f"ALLOWANCE — Inferred {trade} scope (not stated in documents)",
            qty=None,
            unit="LS",
            extraction_method="collapsed_inferred",
            confidence=0.40,
            source=group[0].source if group else "specs",
            schedule_mark="",
            material_spec="",
            spec_section=group[0].spec_section if group else "",
            review_reason=(
                f"ALLOWANCE — {len(group)} inferred item(s) collapsed. "
                f"AI assumed these exist but they are not stated in the documents: {descs}. "
                "Confirm scope before pricing."
            ),
        ))
        logger.info(
            f"  collapse_conditionals: collapsed {len(group)} inferred {trade} items → 1 ALLOWANCE"
        )

    result = explicit_items + allowance_items
    logger.info(
        f"  collapse_conditionals: {len(items)} items → {len(result)} "
        f"({len(explicit_items)} explicit + {len(allowance_items)} allowances, "
        f"{len(dropped_referenced)} NIC dropped)"
    )
    return result


# ════════════════════════════════════════════════════════════════════════════════
# Post-processing: filter items against scope boundary exclusions
# ════════════════════════════════════════════════════════════════════════════════

EXCLUSION_FILTER_PROMPT = """\
You are reviewing a list of extracted construction bid items against a set of explicit
scope EXCLUSIONS from the project's Invitation to Bid.

Your job: for each item in the list, determine whether it falls under one of the
exclusions. If it does, add the tag "[trigger:referenced]" to the start of its
"review" field. Do not modify any other fields. Do not remove items.

EXCLUSIONS:
{exclusions}

Return the COMPLETE item list as a JSON array with the same fields. Only change the
"review" field for items that match an exclusion, and only by prepending
"[trigger:referenced] NIC per ITB: <matched exclusion>. " to the existing review text.

Return ONLY the JSON array, no commentary.
"""

def filter_excluded_items(
    items: List[EstimateItem],
    boundary: Optional[ScopeBoundary],
) -> List[EstimateItem]:
    """Tag items that match ITB exclusions as [trigger:referenced] so collapse_conditionals drops them.

    Sends the full item list + exclusion strings to Gemini for semantic matching.
    Safe by design: on any failure returns the original list unchanged.
    """
    if not boundary or not boundary.excluded or not items:
        return items

    client = _get_client()

    exclusions_text = "\n".join(f"  - {ex}" for ex in boundary.excluded)
    prompt = EXCLUSION_FILTER_PROMPT.format(exclusions=exclusions_text)

    items_data = [
        {
            "trade": item.trade,
            "description": item.item_description,
            "qty": item.qty,
            "unit": item.unit,
            "confidence": item.confidence,
            "source": item.source,
            "method": item.extraction_method,
            "review": item.review_reason,
            "material_spec": item.material_spec,
        }
        for item in items
    ]

    full_prompt = prompt + "\n\nITEMS:\n" + json.dumps(items_data, indent=2)

    logger.info(
        f"  filter_excluded_items: checking {len(items)} items against "
        f"{len(boundary.excluded)} exclusion(s)..."
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[full_prompt],
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=8192),
            ),
        )

        parsed = _parse_json_response(response.text or "")
        if not isinstance(parsed, list) or len(parsed) != len(items):
            logger.warning(
                f"  filter_excluded_items: unexpected response (got {type(parsed).__name__}, "
                f"{len(parsed) if isinstance(parsed, list) else '?'} items) — skipping filter"
            )
            return items

        filtered = _items_from_parsed(parsed)
        tagged = sum(
            1 for item in filtered
            if (item.review_reason or "").startswith("[trigger:referenced]")
        )
        if tagged:
            logger.info(f"  filter_excluded_items: tagged {tagged} item(s) as NIC/referenced")
            for item in filtered:
                if (item.review_reason or "").startswith("[trigger:referenced]"):
                    logger.info(f"    NIC: {item.item_description}")
        else:
            logger.info("  filter_excluded_items: no items matched exclusions")
        return filtered

    except Exception as e:
        logger.error(f"  filter_excluded_items failed: {e} — returning original list")
        return items


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
