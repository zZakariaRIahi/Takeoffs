# QTO Pipeline — Project Context

## What this is
AI-powered Quantity Takeoff pipeline that extracts scope items from construction bid documents (drawings + specifications) using Google Gemini APIs.

## Deployment
- **Cloud Run**: https://takeoffs-670952019485.us-central1.run.app
- **Project**: gen-lang-client-0144509263 (Estimator)
- **Config**: 16GB RAM, 8 CPU, 25min timeout, min-instances 1, session-affinity, gen2, cpu-boost
- **Frontend**: FastAPI + plain HTML (NOT Streamlit — it was removed due to rerun/websocket issues)

## Architecture (see ARCHITECTURE.md for full details)
- `main.py` — FastAPI backend, runs pipeline in background thread, polls via /status
- `static/index.html` — plain HTML/JS frontend, polls every 3s
- `app/agents/document_classifier_agent.py` — Step 1: classify PDFs via Gemini File Search + Files API
- `app/extractors/table_extractor.py` — Step 2a: img2table bbox detection (no OCR) → Gemini Flash filter → 300 DPI crop → Gemini Flash extraction
- `app/extractors/drawing_reader.py` — Step 2d: full PDF upload to Gemini Pro for scope extraction
- `app/extractors/vision_quantifier.py` — Step 3: count/measure items on plan pages via Gemini Pro vision

## Key decisions made
- **Streamlit removed** — caused reruns, websocket drops, lost results. Replaced with FastAPI + polling.
- **img2table WITHOUT Tesseract OCR** — used only for table bbox detection (line detection), <1s/page. Content extraction done by Gemini Flash at 300 DPI.
- **Image-heavy PDFs use Files API** — Gemini File Search hangs on PDFs with low text density (known bug). Files API is reliable.
- **Drawing page images rendered only after classification** — saves memory by not rendering 230 spec pages.
- **Background thread for pipeline** — FastAPI doesn't interfere with execution. Frontend polls /status.
- **Results saved to /tmp** — survives websocket drops within same instance.

## Known accuracy issues (from ChatGPT review of BOM output)
- Door schedule misread: HM (hollow metal) read as "Wood", OHD read as "Aluminum"
- Finish schedule misread: SC (sealed concrete) read as VCT, RB (rubber base) misread
- Root cause: small text in schedule cells needs 300 DPI crops (being fixed with 3-phase approach)
- Electrical counts blank: dense E101/E102 plans need quadrant counting + symbol legend
- Existing items: (P)/(E) prefix means existing, must NOT be in BOM as new items
- FBO items: must be labeled "Install only", material = "Furnished by Owner"
- Missing envelope items: roof, siding, gutters, trim only on elevations/details

## Deploy command
```bash
gcloud run deploy takeoffs \
  --source . --region us-central1 \
  --memory 16Gi --cpu 8 --timeout 1500 \
  --allow-unauthenticated --min-instances 1 --max-instances 3 \
  --session-affinity --cpu-boost --execution-environment gen2 \
  --set-env-vars "GOOGLE_API_KEY=<YOUR_GOOGLE_API_KEY>,OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>" \
  --project gen-lang-client-0144509263
```

## User preferences
- Wants fast, direct responses
- Prefers testing changes before deploying
- Deploy only when asked
- Don't add timeouts to Gemini indexing without asking
- Check logs with: `gcloud run services logs read takeoffs --region us-central1 --project gen-lang-client-0144509263 --limit N`
