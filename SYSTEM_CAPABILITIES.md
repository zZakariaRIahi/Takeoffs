# AI Estimator System — Capabilities & Limitations

## System Overview

The AI Estimator is a multi-agent system that automates construction quantity takeoff and pricing. It consists of two production agents and one planned agent:

| Agent | Status | Purpose |
|-------|--------|---------|
| **Takeoffs Agent** | Production | Extracts scope items, quantities, and materials from construction bid documents |
| **Pricing Agent** | Production | Classifies items by company trades and prices them using RSMeans + retail data |
| **Audit Agent** | Planned | Computer vision-based verification of counts and measurements |

---

## 1. Takeoffs Agent

The Takeoffs Agent processes construction drawing sets through a 3-step pipeline to produce a structured Bill of Materials.

### Step 1 — Document Classification

The agent classifies uploaded PDF files into **8 standard bid document categories**:

| Category | Description |
|----------|-------------|
| Cover Sheet | Title page, table of contents |
| Instructions to Bidder | Bidding procedures and requirements |
| Project Specifications | CSI Divisions 01–49 technical specs |
| Construction Drawings | Actual drawing sheets (plans, details, schedules) |
| General Conditions | Standard contract terms (e.g., AIA A201) |
| Special Conditions | Project-specific amendments to general conditions |
| Bid Form | Proposal forms and pricing sheets |
| Bid Security | Bid bond forms and surety documents |

Classification is performed by uploading the PDF to Google Gemini and using the model to identify each page's category. Only pages identified as construction drawings are rendered as images for downstream processing — specification pages are skipped, saving significant processing time and memory.

### Step 2 — Context Extraction

This step reads all text-based information from the drawing pages in a single pass per discipline. It extracts:

#### Sheet Index & Page Mapping
- Reads the title page to extract the full sheet index
- Maps every sheet to its page number, sheet ID, and title
- Identifies what views are on each page (floor plan, reflected ceiling plan, roof plan, elevation, section, detail, demolition plan, enlarged plan)

#### Discipline Classification
Pages are classified into their construction discipline based on standard sheet ID prefixes:



#### Schedule Extraction
The agent extracts structured data from all schedule tables found on drawing pages, including:

- Door schedules (mark, size, material, type, frame, hardware)
- Window schedules
- Room finish schedules (floor, base, wall, ceiling finishes)
- Fixture schedules (plumbing fixtures, equipment)
- Lighting fixture schedules
- Panel schedules
- Hardware schedules
- Equipment schedules

Each schedule is parsed into structured headers and rows with exact cell values preserved (abbreviations like HM, SC, RB, OHD, STL, GYP/PT are kept as-is).

#### Keynote Extraction
All keynotes, general notes, and numbered notes relevant to construction scope are extracted with their page references. These drive scope items like demolition work, patching, and special conditions.

#### Symbol Legend Extraction
Symbol definitions from legend pages are captured (e.g., "WC-1: Wall-mounted water closet, Zurn Z5615"). These definitions are passed to Step 3 so the plan-reading agent knows what each symbol represents when counting.

#### Plan Names
Each page's plan views are identified and named (e.g., "First Floor Plan", "Reflected Ceiling Plan", "Exterior Elevations"), providing context for the quantification step.

### Step 3 — Per-Discipline Quantification

A dedicated Gemini Pro agent processes each discipline independently, receiving:
- All page images for that discipline (rendered at 300 DPI)
- Pre-extracted schedules, keynotes, and symbol definitions from Step 2

The agent performs:

- **Schedule-based quantities** — Counts rows in schedules to determine quantities for doors, windows, fixtures, and equipment
- **Keynote-based items** — References keynote text and counts keynote callouts on plans
- **Symbol counting on plans** — Attempts to count fixture symbols, device symbols, and equipment on plan pages using a quadrant method (NW, NE, SW, SE)
- **Dimension reading** — Attempts to read dimension strings from drawings and calculate areas/lengths
- **Demolition scope** — Extracts items to be removed from demolition plans
- **Existing item filtering** — Excludes items marked (E), (P), or "EXISTING" from new scope
- **FBO identification** — Flags Furnished-by-Owner items as "Install Only"

Each extracted item includes a source reference (which sheet it came from), extraction method (how the quantity was determined), and confidence score.

### Input Requirements

| Input | Required | Format |
|-------|----------|--------|
| Construction drawings | Yes | Single PDF |
| Project manual / specifications | Yes | Single PDF |
| Pre-extracted schedule data | Optional | CSV, XLSX, or DOCX |

---

## 2. Pricing Agent

The Pricing Agent takes the extracted Bill of Materials and produces unit costs for each item.

### Trade Classification
Items are classified into **23 company-standard trades**:

General Requirements, Site Work, Masonry, Concrete, Metals, Rough Carpentry, Finish Carpentry, Plumbing, Electrical, HVAC and Sheet Metals, Insulation, Doors and Windows, Drywall, Cabinets, Stucco and Siding, Painting, Roofing, Tile & Solid Surfaces, Bath and Accessories, Appliances, Flooring, Fire Sprinklers, Landscaping

### Pricing Method
The agent prices items trade-by-trade using two sources:

- **Labor costs** — Web search on RSMeans construction cost data to find labor rates and man-hour estimates per item
- **Material costs** — Web search on retail supplier websites to find current material pricing

Each priced item returns a unit cost, man-hours estimate, price source citation, and any relevant notes.

---

## 3. Current Limitations

### Input Limitations

| Limitation | Details |
|------------|---------|
| **Single drawing file only** | The system processes one drawings PDF per run. It cannot merge or cross-reference multiple drawing packages. |
| **Single project manual only** | One specification PDF per run. |
| **No addendum support** | Cannot process addendum documents or reconcile changes, additions, or deletions from addenda against the base bid set. |
| **No geotechnical reports** | Does not read or interpret geotechnical data, soil boring logs, or environmental reports. |
| **No multi-attachment handling** | Cannot process supplemental documents such as separate submittals, RFIs, shop drawings, survey data, or owner-furnished information packages. |

These input limitations are planned for future development.

### Accuracy Limitations

The Takeoffs Agent relies on a large language model (Gemini) with vision capabilities for plan reading. While LLMs excel at text understanding and structured data extraction, they have fundamental limitations for visual precision tasks:

#### Symbol Counting — Unreliable

LLM vision models are not designed for precise object counting in construction data. Known issues:

- **Miscounts on dense plans** — Electrical plans (E-101, E-102) with dozens of similar symbols are frequently miscounted. The model may skip symbols, double-count, or confuse similar symbol types.
- **Small or overlapping symbols** — Fixtures in tight spaces (toilet rooms, mechanical rooms) are often missed or merged.
- **Scale sensitivity** — The same symbol at different scales may not be recognized consistently.
- **No pixel-level precision** — LLMs process images as visual patterns, not as geometric data. They cannot reliably distinguish between 14 and 16 instances of a symbol.

The quadrant counting method (dividing pages into NW/NE/SW/SE) improves results for small counts (<20) but remains unreliable for dense plans.

#### Plan Measurements — Unreliable

LLM vision models cannot accurately measure dimensions from construction plans:

- **Cannot measure to scale** — The model cannot reliably interpret graphic scales or use them to measure unmarked dimensions.
- **Dimension string reading errors** — Small text like 3'-6" or 1/4" = 1'-0" is frequently misread.
- **Area calculation errors** — Even when dimensions are correctly read, the model may apply incorrect geometry or miss irregular shapes.
- **No pixel measurement capability** — LLMs have no mechanism to count pixels or calculate true distances on a raster image.

For these reasons, all SF/LF/SY measurement items are flagged for manual takeoff by a human estimator. The system identifies WHAT needs to be measured and WHERE, but does not provide the measurement itself.

### What the AI Gets Right

| Capability | Reliability | Why |
|-----------|-------------|-----|
| File classification | High | Text-based task, well-suited for LLMs |
| Discipline classification | High | Rule-based prefix matching, no AI needed |
| Schedule extraction | High | Structured table reading at 300 DPI |
| Keynote extraction | High | Text reading task |
| Symbol legend reading | High | Text reading task |
| Sheet index / page mapping | High | Structured text on title page |
| Trade classification | High | Text categorization task |
| Item identification (what exists) | High | LLM excels at understanding drawing content |
| Symbol counting | Low | Visual precision task — LLMs are not designed for this |
| Plan measurements | Low | Pixel-level geometric task — LLMs cannot do this |

---

## 4. Planned: Audit Agent (Computer Vision)

To address the accuracy limitations of LLM-based counting and measurement, a dedicated Audit Agent is planned that uses trained computer vision models instead of large language models.

### Design Approach — Three Layers

#### Layer 1: Plan Cropping Model (Image Segmentation)

**Purpose:** Automatically crop and isolate individual plan views from drawing sheets.

Construction drawings often contain multiple views per sheet (floor plan, enlarged details, schedules, title blocks). This layer trains an image segmentation model to identify and crop each plan view, providing clean isolated inputs for the counting and measurement layers.

**Development pipeline:**
- Data collection — Assemble a dataset of construction drawing sheets across multiple project types
- Annotation — Label plan boundaries, title blocks, schedule regions, detail callouts, and revision clouds
- Model training — Train a segmentation model to detect and crop plan regions
- Model validation — Test against held-out drawing sets to verify crop accuracy across disciplines and drawing styles

#### Layer 2: Object Detection Model (Symbol Counting)

**Purpose:** Accurately count construction symbols on plans using trained object detection.

Instead of asking an LLM to visually count symbols, this layer trains a dedicated image segmentation model on construction plan symbols. The model learns to identify specific symbol types (light fixtures, receptacles, plumbing fixtures, mechanical equipment, fire protection devices) and count them precisely.

**Development pipeline:**
- Data collection — Gather annotated construction plans with symbol-level labels
- Annotation — Mark every symbol instance with bounding boxes and type classification
- Model training — Train an object detection / instance segmentation model specialized for construction symbology
- Model validation — Validate counting accuracy against human-verified counts across project types and drawing densities

**Expected improvement:** Pixel-level detection eliminates the hallucination problem — the model either detects a symbol at a specific coordinate or it doesn't. No estimation or pattern-guessing involved.

#### Layer 3: Pixel-Based Measurement (Computer Vision)

**Purpose:** Perform accurate area and length measurements directly from plan images.

This layer uses computer vision techniques to measure dimensions from construction plans by:
- Detecting the graphic scale or reading dimension strings to establish the pixels-per-unit ratio
- Identifying object boundaries through edge detection and segmentation
- Counting pixels within each object or along each path to compute real-world measurements
- Converting pixel counts to construction units (SF, LF) using the established scale ratio

**Expected improvement:** Deterministic geometric computation replaces LLM estimation. Measurements are reproducible and verifiable — the same image always produces the same measurement.

### How the Audit Agent Fits Into the Pipeline

```
┌──────────────────────────────────────────────────────────┐
│               Current Pipeline (LLM-based)                │
│                                                          │
│  Step 1: Classification ──→ Step 2: Context Extraction   │
│                                      │                   │
│                                      ▼                   │
│                              Step 3: Quantification      │
│                              (LLM vision — unreliable    │
│                               for counts & measurements) │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│               Audit Agent (CV-based) — Planned            │
│                                                          │
│  Layer 1: Crop plans from sheets (image segmentation)    │
│                         │                                │
│              ┌──────────┴──────────┐                     │
│              ▼                     ▼                     │
│  Layer 2: Count symbols    Layer 3: Measure dimensions   │
│  (object detection)        (pixel measurement)           │
│              │                     │                     │
│              └──────────┬──────────┘                     │
│                         ▼                                │
│              Verified quantities replace                  │
│              LLM-estimated quantities                     │
└──────────────────────────────────────────────────────────┘
```

The Audit Agent will run after the current pipeline and replace LLM-generated counts and measurements with CV-verified values. Schedule-based quantities and text extraction will continue to use the LLM pipeline, which is already reliable for those tasks.

---

## 5. Role of the Human Estimator

The AI Estimator is an assistant, not a replacement. The system accelerates the estimating process by producing a structured first-pass BOM, but human judgment remains essential.

### What the Estimator Must Verify

| Task | Why |
|------|-----|
| **All EA quantities** | Symbol counts come from LLM vision and can hallucinate. Every count should be spot-checked against the drawings. |
| **All SF/LF measurements** | The system identifies items that need measurement and points to the source sheets, but does not provide the measurement. The estimator must measure from the plans. |
| **Schedule abbreviations** | Small text like HM (hollow metal), SC (sealed concrete), RB (rubber base) can be misread. Cross-check against the original PDF. |
| **Existing vs. new items** | Items prefixed (E) or (P) are existing and should not be in the BOM. Verify none were incorrectly included. |
| **FBO items** | Furnished-by-Owner items should be marked "Install Only" with no material cost. Confirm correct flagging. |
| **Demolition scope** | Verify demo items match the demolition plans and aren't duplicated with new work items. |
| **Missing items** | The AI may miss items that are only shown in details, sections, or general notes. Review the drawings for anything not captured. |
| **Pricing accuracy** | Unit costs from web search may not reflect local market conditions, current pricing, or project-specific factors. Adjust as needed. |

### What the Estimator Can Trust

- **Item identification** — the system reliably finds what's in the documents. If an item appears in a schedule or keynote, it will be in the BOM.
- **Trade classification** — items are correctly assigned to CSI trades.
- **Source traceability** — every item links to a specific sheet and extraction method, making verification straightforward.
- **Schedule data** — structured table data (door types, finish specs, fixture models) is extracted accurately at 300 DPI.
