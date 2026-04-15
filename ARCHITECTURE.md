# Takeoffs Agent — Complete Architecture

## Overview

AI-powered Quantity Takeoff pipeline that extracts scope items from construction bid documents (drawings, specifications, or both), produces an editable Bill of Materials (BOM), and sends it to a pricing agent for material + labor cost estimation.

**Stack:** Python/FastAPI backend, plain HTML/JS frontend, Google Gemini API (Flash + Pro), deployed on Cloud Run.

**URL:** https://takeoffs-670952019485.us-central1.run.app

---

## Infrastructure

```
Cloud Run:
  Memory: 16GB RAM
  CPU: 8 vCPU, gen2 execution environment, CPU boost enabled
  Timeout: 25 minutes
  Instances: min 1, max 3, session affinity

GCS Bucket: takeoffs-uploads-670952019485
  /{session_id}/          — uploaded files (temp, cleaned after pipeline)
  /results/{job_id}.json  — persistent job results (survive deploys)

Gemini API: Google API Key auth
Pricing Agent: https://cost-rag-pricing-670952019485.us-central1.run.app/price
```

---

## Job Queue System

```
3 concurrent pipelines max. 4th+ uploads queue automatically.
Each upload → unique job_id → own state, own results, own URL.
Results persist in GCS across deploys.

Endpoints:
  POST /get-upload-urls        → GCS signed URLs for browser direct upload
  POST /start/{session_id}     → creates job, starts immediately or queues
  GET  /status/{job_id}        → poll specific job progress
  GET  /results/{job_id}       → get specific job results (bookmarkable)
  GET  /jobs                   → list all jobs (active + GCS history)
  POST /submit-pricing         → send items to pricing agent (background)
  GET  /pricing-status         → poll pricing progress
  POST /force-reset            → cancel current state

Frontend isolation:
  Each browser window tracks its OWN job via URL hash (#job=abc123)
  New window → upload form + job history (doesn't attach to others' jobs)
  Job history panel shows all previous jobs, click to view results
```

---

## Pipeline Flow

```
Upload → Step 1 (Classify & Render)
       → Route based on file types:
           Case 1: Specs only    → Specs Extraction → Results
           Case 2: Specs + Dwgs  → Drawings flow (Case 3) [merge TBD]
           Case 3: Drawings only → Step 2 → Step 3 → Results
       → Human Review (edit qty/description)
       → Pricing Agent
       → Priced BOM + CSV Export
```

---

## Step 1: Classify & Render

**File:** `app/agents/document_classifier_agent.py`
**Model:** Gemini 2.5 Pro
**Purpose:** Determine file types, render drawing pages at 300 DPI

```
Input: List[(filename, bytes)] — PDFs, CSV, XLSX, DOCX uploaded by user
  Non-PDF files parsed immediately:
    CSV → pandas → ExtractedTable
    XLSX → pandas per sheet → ExtractedTable
    DOCX → python-docx tables → ExtractedTable

For each PDF, classify FILE TYPE:
  - File < 20MB → Gemini File Search (FileSearchStore + tool)
  - File > 20MB → Gemini Files API (direct upload)
  - Files API fails (400) → vision fallback (20 samples at 100 DPI)
  - File > 1000 pages → split 3 chunks, classify each in parallel

Categories detected:
  construction_drawings, project_specifications, cover_sheet,
  general_conditions, special_conditions, instructions_to_bidder,
  bid_form, bid_security

Flag drawing pages:
  - Pure drawings file (ONLY construction_drawings, no specs mixed in)
    → ALL pages flagged as drawings
  - Mixed file (both construction_drawings AND project_specifications)
    → ONLY model-identified drawing pages flagged (per-page detection)
  - Specs-only file → no drawing pages

Render ALL flagged drawing pages at 300 DPI:
  PyMuPDF → JPEG quality 85
  8 parallel threads (PyMuPDF releases GIL)
  ~8s/page on Cloud Run
  Images stored on PageInfo.image_bytes
  Reused by Steps 2 and 3 — no re-rendering

Output: DocumentClassificationResult
  .files[].pages[].image_bytes  (300 DPI JPEG, drawings only)
  .files[].pages[].has_drawings (True/False)
  .files[].pages[].extracted_text (PyMuPDF text)
  .raw_pdf_bytes {filename: bytes}

Then DETECT what exists:
  has_drawings = any file classified as construction_drawings
  has_specs = any file classified as project_specifications
  → Routes to Case 1, 2, or 3
```

---

## Case 1: Specs Only (no drawings)

**File:** `app/extractors/specs_extractor.py`
**Model:** Gemini 2.5 Pro (thinking=32768)
**Purpose:** Full quantity takeoff from specifications alone

```
Input: DocumentClassificationResult (specs files only)

Upload specs PDF to Gemini Files API
  If > 45MB → split with PyMuPDF + garbage=4

Single Pro call with detailed estimator prompt that teaches:

  QUANTITY EXTRACTION RULES:
  1. EXPLICIT: "Paint walls (approx 35,000 sf)" → qty: 35000, unit: SF
  2. COUNT LOCATIONS: "casework in A300 and B300" → qty: 2, unit: EA
  3. COUNT LISTS: "Rooms A100, A101, A102, B100, B101, B102" → qty: 6
  4. DERIVE FROM CONTEXT: "sinks in classrooms A300 and B300" → qty: 2
  5. PER-ITEM BREAKDOWN: demo + new work = separate line items
  6. AREA CALCULATIONS: rooms × avg size = total SF
  7. LS ONLY for genuinely unquantifiable items (bonds, permits, cleanup)
  8. ALTERNATES as separate items with prefix
  9. NULL only when truly unknowable (needs drawings, "as required")
  10. METHOD field explains HOW quantity was derived

Extracts from ALL document sections:
  - Scope summaries / bid forms (most explicit quantities)
  - CSI spec sections (material specs, products)
  - Demolition sections (separate from new work)
  - Environmental/abatement reports (areas from surveys)
  - General requirements (permits, bonds, temp facilities)
  - Addenda (scope changes)

Output: List[EstimateItem] → goes directly to Results
  Each item: trade, description, qty, unit, spec_section,
  source, method (explains qty derivation), material_spec, review

Skips Steps 2-3 entirely.
```

---

## Case 2: Specs + Drawings (both exist)

**Current behavior:** Treats as Case 3 (drawings only). Specs ignored for now.

**Future:** Will extract from specs first, then merge spec context into drawing
discipline packages for enriched Step 3 plan reading. Code exists in
`specs_extractor.py` (extract_from_specs_with_drawings + merge_specs_into_packages)
but is not yet wired into the pipeline.

---

## Case 3: Drawings Only — also used for Case 2

### Step 2: Context Extraction

**File:** `app/extractors/context_extractor.py`
**Model:** Gemini 2.5 Flash
**Purpose:** Extract ALL text-based content so Step 3 doesn't re-read any text

```
Part A — Sheet Index + Discipline Mapping (1 Flash call on title page):

  Flash reads the sheet index table and extracts per sheet:
    sheet_id, title, discipline, AND page number

  Flash groups disciplines correctly from the actual title page:
    Architectural (includes AD- demolition sheets)
    Civil, Structural
    Plumbing (includes PD- demolition)
    Mechanical (includes V- ventilation + VD- demo)
    Electrical (includes ED- demolition)
    Fire Protection, Abatement (ASB- + LBP-), Landscape

  Page numbers assigned by Flash from sheet index order.
  Disciplines from Flash, not prefix guessing in code.
  Code groups: discipline_pages = {discipline: [page_numbers]}

Part B — Per-Discipline Extraction (parallel Flash calls):
  10 workers, retry on 503/SSL with 15s/30s backoff

  For each discipline group:
    Send all pages (300 DPI images from Step 1)
    Flash returns MARKDOWN (not JSON — avoids 3'-0" escaping issues):

      # CONTEXT
      Renovation of Building 1A. Existing items marked (E)/(P)...

      # PAGE INFO
      - Page 14 (A102): Floor Plan, Toilet Plan [has_plans] [has_schedules]

      # SYMBOLS
      - WC-1: Wall-mounted water closet [fixture]

      # KEYNOTES
      - 1 (page 7): REMOVE EXISTING DOOR AND FRAME

      # SCHEDULE: DOOR SCHEDULE (door) [page 14]
      | DOOR NO. | WIDTH | HEIGHT | MATERIAL |
      |----------|-------|--------|----------|
      | 101 | 3'-0" | 7'-0" | HM |

  Markdown parsed in Python:
    _parse_markdown_response() → splits on # section headers
    _parse_markdown_table() → splits on | pipes
    No JSON escaping issues with construction values

Output:
  sheets: List[SheetInfo]
  tables: List[ExtractedTable]
  packages: Dict[discipline, DisciplinePackage]
    .schedules — tables with headers + full row data
    .keynotes — [{key, text, page}]
    .symbols — [{symbol, description, category}]
    .page_info — [{page, sheet_id, plans, has_schedules, has_plans}]
    .context — discipline scope summary
```

### Step 3: Plan Reading & Quantification

**File:** `app/extractors/trade_extractor.py`
**Model:** Gemini 2.5 Pro (thinking_budget=32768)
**Purpose:** Read plans visually — count symbols, read dimensions, validate quantities

```
Input: classification, sheets, tables, packages (from Step 2)

Context-only disciplines skipped:
  Title/Index, Cover, General, Unknown → page refs only, not extracted

For each active discipline (10 workers, retry 503/SSL):

  Receives from Step 2 (text, NOT re-read from images):
    - Schedules with full row data
    - Keynotes with text
    - Symbol definitions from legends
    - Page info with plan names
    - Context summary

  Pro receives 300 DPI page images + focused prompt:

  SCHEDULE ITEMS:
    qty = number of matching rows in schedule
    method: "from_schedule: Door Schedule, 27 rows"
    source: "schedule:MARK"

  COUNT SYMBOLS (EA items):
    Quadrant method: NW, NE, SW, SE → total
    method: "counted: WC-1 on P-112, NW:2 NE:1 SW:2 SE:1 = 6"
    source: "plan_count:SHEET_ID"

  READ DIMENSIONS (SF/LF items):
    Read dimension strings, show math
    method: "measured: Room 101 = 15'-0" × 12'-6" = 187.5 SF"
    If unreadable → qty=null, review="Manual takeoff — measure from: A-107"
    source: "plan_measurement:SHEET_ID"

  KEYNOTE ITEMS:
    method: "from_keynote: K-5 on sheet P-112"
    source: "keynote:SHEET_ID"

  DEMOLITION:
    Count items to remove on demo plans
    source: "plan_count:SHEET_ID"

  Rules:
    (E)/(P) = existing → skip
    FBO = "(FBO — Install Only)"

Output: List[EstimateItem]
  trade, description, qty, unit, mark, source, method,
  confidence (0.85/0.65/0.45), review, material_spec
```

---

## All Cases Converge: Results & Human Review

```
Serialization:
  Each EstimateItem → {trade, description, qty, unit, mark,
    confidence, source, method, review}
  Saved to: _jobs[job_id]["items"] (memory) + GCS (persistent)

Frontend (static/index.html):
  Editable table:
    Description (editable), Qty (editable), Unit (editable)
    Mark, Method, Source, Confidence, Review (read-only)
  Manual takeoff items highlighted red with source page reference
  Add/remove rows
  Job history panel — click any previous job to view
  URL hash: #job=abc123 (bookmarkable, shareable)
  "Submit for Pricing" button
```

---

## Pricing

```
Background thread sends to pricing agent:
  POST https://cost-rag-pricing-670952019485.us-central1.run.app/price

  Payload per item:
    {trade, description, qty (default 1), unit, mark, source}

  Response per item:
    material_unit_cost, labor_unit_cost,
    material_total, labor_total,
    man_hours, man_hour_rate, line_total,
    price_source, citations[], notes

  Frontend displays:
    Mat/Unit | Mat Total | Labor/Unit | Labor Total |
    Man HRs | Rate | Total | Notes

  Export CSV
```

---

## Files

| File | Step | Purpose |
|------|------|---------|
| `main.py` | — | Job queue (3 concurrent), case routing, endpoints, pricing |
| `app/agents/document_classifier_agent.py` | 1 | File classification + 300 DPI rendering |
| `app/extractors/specs_extractor.py` | 1.5 | Specs extraction (Case 1 + future Case 2) |
| `app/extractors/context_extractor.py` | 2 | Sheet index + schedules + keynotes + symbols (markdown) |
| `app/extractors/trade_extractor.py` | 3 | Per-discipline plan reading & quantification |
| `app/core/estimate_models.py` | — | Data models (EstimateItem, SheetInfo, ExtractedTable) |
| `app/core/document_classification.py` | — | Classification models (PageInfo, ClassifiedFile) |
| `static/index.html` | — | Frontend (upload, progress, edit, job history, pricing, CSV) |

---

## Models & API Calls

| Step | Model | Thinking | Calls | Purpose |
|------|-------|----------|-------|---------|
| 1 Classification | Gemini 2.5 Pro | default | 1/file | File type detection |
| 1 Vision fallback | Gemini 2.5 Pro | default | 1/file (if needed) | Fallback for large files |
| 1.5 Specs (Case 1) | Gemini 2.5 Pro | 32768 | 1/specs file | Full takeoff from specs |
| 2A Sheet index | Gemini 2.5 Flash | no | 1 (title page) | Sheet index + discipline + page mapping |
| 2B Context | Gemini 2.5 Flash | no | 1/discipline | Schedules, keynotes, symbols (markdown) |
| 3 Plan reading | Gemini 2.5 Pro | 32768 | 1/discipline | Counting, measuring, quantification |
| Pricing | External API | — | 1 total | Material + labor cost lookup |

---

## Disciplines (from Flash, not hardcoded)

Extracted by Flash from the actual sheet index on the title page.
Demo sheets grouped under parent discipline.

| Discipline | Sheet Prefixes | Includes |
|------------|---------------|----------|
| Architectural | A-, AD- | Plans, schedules, demo, finishes, doors |
| Civil | C- | Site plans, utilities, grading |
| Structural | S- | Foundation, framing, details |
| Plumbing | P-, PD- | Piping, fixtures, demo |
| Mechanical | V-, VD-, M- | Ventilation, ductwork, equipment, demo |
| Electrical | E-, ED- | Power, lighting, controls, demo |
| Fire Protection | FP- | Sprinkler plans |
| Abatement | ASB-, LBP- | Asbestos, lead paint |
| Landscape | L- | Landscape plans |

---

## Key Design Decisions

| Decision | Reasoning |
|----------|-----------|
| 3 cases based on file types | Specs-only bids are real. Combined files need per-page detection |
| Specs prompt teaches estimator thinking | Count rooms, derive from context, split demo/new, explain method |
| Mixed file → per-page detection | "File = drawings → all pages" fails for combined bid packets |
| Markdown output in Step 2 | Construction values (3'-0", 1-3/4") break JSON escaping |
| Discipline from Flash | Sheet index has the mapping. No prefix guessing in code |
| Step 2 = text, Step 3 = vision | Flash cheap for OCR. Pro expensive — don't waste on schedules |
| 3 concurrent + queue | Multiple estimators share the app simultaneously |
| Results in GCS | Persist across deploys, bookmarkable per job |
| 300 DPI once in Step 1 | Single render pass. Steps 2 and 3 reuse same images |
| Retry on 503/SSL | Gemini demand spikes. 15s/30s backoff prevents lost disciplines |
| Per-browser job isolation | URL hash prevents window A seeing window B's running job |

---

## Deploy

```bash
gcloud run deploy takeoffs \
  --source . --region us-central1 \
  --memory 16Gi --cpu 8 --timeout 1500 \
  --allow-unauthenticated --min-instances 1 --max-instances 3 \
  --session-affinity --cpu-boost --execution-environment gen2 \
  --set-env-vars "GOOGLE_API_KEY=...,OPENAI_API_KEY=..." \
  --project gen-lang-client-0144509263
```
