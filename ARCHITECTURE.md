# QTO Pipeline — Architecture Reference

## Overview

AI-powered **Quantity Takeoff (QTO) Pipeline** that extracts scope items from construction bid documents (drawings + specifications) using Google Gemini APIs.

**Stack:** Streamlit UI, Python 3.11, Gemini 2.5 Pro/Flash, PyMuPDF, img2table + Tesseract OCR, Pydantic

---

## Project Structure

```
Takeoffs/
├── streamlit_app.py                     # Main UI + background pipeline runner
├── Dockerfile                           # Cloud Run deployment
├── requirements.txt                     # Python deps
├── packages.txt                         # System deps (tesseract)
└── app/
    ├── config/
    │   ├── settings.py                  # Pydantic BaseSettings (API keys, feature flags)
    │   └── trades.py                    # 23 CSI MasterFormat trades
    ├── core/
    │   ├── document_classification.py   # Data models: PageInfo, ClassifiedFile, DocumentClassificationResult
    │   └── estimate_models.py           # Data models: SheetInfo, ExtractedTable, EstimateItem
    ├── utils/
    │   └── genai_client.py              # get_genai_client() helper
    ├── agents/
    │   └── document_classifier_agent.py # Step 1: Classify + render drawing pages
    └── extractors/
        ├── sheet_indexer.py             # Step 2a: Parse sheet index from title page
        ├── table_extractor.py           # Step 2a: Detect & extract schedules (img2table + Gemini Flash)
        ├── drawing_reader.py            # Step 2d: Full drawing extraction (Gemini Pro + Files API)
        └── vision_quantifier.py         # Step 3: Count/measure items on plan pages (Gemini Pro vision)
```

---

## 4-Step Pipeline

```
[Upload PDFs]
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1: Document Classification                        │
│  document_classifier_agent.py                           │
│                                                         │
│  1. Ingest files (extract text only, no image rendering)│
│  2. Upload raw PDFs to Gemini File Search → classify    │
│     into 8 categories + identify drawing page numbers   │
│  3. Fallback chain: File Search/Files API → Vision       │
│  4. Render images ONLY for drawing pages (200 DPI)      │
│                                                         │
│  Output: DocumentClassificationResult                   │
│    - files[].pages[].has_drawings, extracted_text        │
│    - files[].pages[].image_bytes (drawings only)        │
│    - raw_pdf_bytes (for downstream uploads)             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2a: Sheet Index + Table Extraction                │
│  sheet_indexer.py + table_extractor.py                  │
│                                                         │
│  Sheet Indexer:                                         │
│  1. Filter to DRAWINGS + SPECS files only               │
│  2. Send title page image to Gemini Flash → parse       │
│     sheet index (sheet_id, title, discipline)           │
│  3. Match pages to index entries (text search + order)  │
│  4. Build SheetInfo per drawing page                    │
│                                                         │
│  Table Extractor:                                       │
│  1. Render pages at 100 DPI → img2table detects tables  │
│  2. Crop tables at 200 DPI → send to Gemini Flash       │
│  3. LLM returns structured JSON (headers + rows)        │
│  4. Convert to ExtractedTable → ExtractedScheduleRow    │
│                                                         │
│  Output: List[SheetInfo], List[ExtractedTable]          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 2d: Drawing Reader                                │
│  drawing_reader.py                                      │
│                                                         │
│  1. Upload full PDFs to Gemini Files API                │
│  2. Build context: page index + pre-extracted schedules │
│  3. Send to Gemini Pro with 4-pass extraction prompt:   │
│     Pass 1: Schedule-based items (link to plans)        │
│     Pass 2: Plan-driven items (symbols, keynotes)       │
│     Pass 3: General notes/details (text-stated scope)   │
│     Pass 4: Cross-reference (link schedules ↔ plans)    │
│  4. Parse response → EstimateItem objects               │
│     - qty may be null if needs_counting/measurement     │
│                                                         │
│  Output: List[EstimateItem]                             │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 3: Vision Quantifier                              │
│  vision_quantifier.py                                   │
│                                                         │
│  For items with needs_counting or needs_measurement:    │
│  1. Group items by (trade, page)                        │
│  2. Render plan pages at 300 DPI                        │
│  3. Send trade-focused prompt + page image to Gemini    │
│     Pro vision (up to 5 concurrent calls)               │
│  4. Model counts symbols / reads dimensions             │
│  5. Merge qty results back into EstimateItem list       │
│                                                         │
│  Output: List[EstimateItem] (qty filled where possible) │
└─────────────────────────────────────────────────────────┘
    │
    ▼
[Results: DataFrame with Trade, Description, Qty, Unit, Mark, Confidence]
```

---

## Key Data Models

### PageInfo (per page)
```
page_number (0-based in file), global_page_number, extracted_text,
has_drawings (bool), image_bytes (JPEG, only if drawing), categories
```

### ClassifiedFile (per uploaded file)
```
filename, categories (Set of 8 types), pages (List[PageInfo]),
has_visual_content, visual_pages, text_pages
```

### DocumentClassificationResult (Step 1 output)
```
files (List[ClassifiedFile]), raw_pdf_bytes (Dict[filename → bytes])
Query helpers: get_files_by_category(), get_pages_by_category(), summary()
```

### SheetInfo (per drawing sheet)
```
sheet_id ("A9.00"), title, discipline ("Architectural"),
global_page_number, source_file, extracted_text, image_bytes,
tables (List[ExtractedTable])
```

### ExtractedTable (per schedule found)
```
page_number, sheet_id, schedule_type ("door"|"window"|"finish"|...),
headers (List[str]), rows (List[Dict]), confidence
```

### EstimateItem (final output — one per scope item)
```
trade, item_description, qty (nullable), unit ("EA"|"LF"|"SF"|...),
extraction_method, confidence, source_page, sheet_id,
material_spec, schedule_mark,
needs_counting (bool), needs_measurement (bool),
counting_target, counting_source_pages
```

---

## 8 Document Categories
1. `cover_sheet` — title page, TOC
2. `instructions_to_bidder` — bidding procedures
3. `project_specifications` — CSI Divisions 01-49
4. `construction_drawings` — actual drawing sheets
5. `general_conditions` — AIA A201
6. `special_conditions` — project-specific amendments
7. `bid_form` — proposal forms, pricing sheets
8. `bid_security` — bid bond forms

---

## 23 CSI Trades
General Requirements, Site Work, Masonry, Concrete, Metals,
Rough Carpentry, Finish Carpentry, Plumbing, Electrical,
HVAC and Sheet Metals, Insulation, Doors and Windows, Drywall,
Cabinets, Stucco and Siding, Painting, Roofing,
Tile & Solid Surfaces, Bath and Accessories, Appliances,
Flooring, Fire Sprinklers, Landscaping

---

## External API Calls

| Step | Model | API | Purpose |
|------|-------|-----|---------|
| 1 | gemini-2.5-pro | File Search | Classify documents (upload PDF → classify) |
| 1 (fallback) | gemini-2.5-pro | Vision | Classify via sampled page images |
| 2a | gemini-2.5-flash | Vision | Parse sheet index from title page |
| 2a | gemini-2.5-flash | Vision | Extract table data from crops |
| 2d | gemini-2.5-pro | Files API | Read full drawings (upload PDF → extract scope) |
| 3 | gemini-2.5-pro | Vision | Count/measure items on plan page images |

**Tesseract OCR**: Used by img2table for table bounding box detection (Step 2a)

---

## DPI Settings
| Context | DPI | Purpose |
|---------|-----|---------|
| Drawing page images (Step 1) | 200 | Stored in PageInfo.image_bytes |
| Table detection (Step 2a) | 100 | img2table bbox detection (low memory) |
| Table crops (Step 2a) | 200 | Sent to Gemini Flash |
| Vision quantification (Step 3) | 300 | Sent to Gemini Pro (high quality for counting) |

---

## Streamlit UI Architecture

- **Background thread** runs the pipeline (Streamlit reruns can't interrupt it)
- **threading.Lock** prevents duplicate concurrent runs
- **Module-level `_pipeline_state` dict** survives Streamlit reruns
- **3-second polling** (`time.sleep(3) + st.rerun()`) updates progress
- Button disabled while running, Reset button on error

---

## Deployment

### Google Cloud Run (primary)
```bash
gcloud run deploy takeoffs \
  --source . \
  --region us-central1 \
  --memory 8Gi --cpu 4 \
  --timeout 1500 \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_API_KEY=...,OPENAI_API_KEY=..."
```

### Streamlit Cloud (fallback)
- `packages.txt` for Tesseract
- `st.secrets` for API keys
- Limited to 1GB RAM (may OOM on large documents)

---

## Error Handling & Fallbacks

| Component | Failure | Fallback |
|-----------|---------|----------|
| Gemini File Search | Upload/indexing fails | Vision classification (sampled pages) |
| Vision classification | API error | Keyword heuristics on filename + text |
| Sheet index parsing | No index found | Detect sheet IDs from page text |
| Table extraction | img2table/Tesseract fails | Continue without tables for that page |
| Drawing reader | Gemini API error | Retry up to 3x with exponential backoff |
| Vision quantifier | Timeout (200s) | Retry once, then leave qty as null |
| Vision quantifier | Low confidence | Leave qty as null (manual review) |
