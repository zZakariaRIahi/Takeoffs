# AI Estimator Assistant — User Documentation

## What Is This?

An AI-powered Quantity Takeoff (QTO) tool that reads construction bid documents — drawings and project manuals — and automatically extracts a Bill of Materials (BOM) with scope items, quantities, units, and trade classifications. It is designed to accelerate the estimating process by giving human estimators a structured starting point instead of a blank spreadsheet.

---

## Current Capabilities & Limitations

### What It Can Do

- Accept **one drawings PDF** and **one project manual PDF** as input (plus optional CSV/XLSX schedule data)
- Classify pages into drawings vs. specifications automatically
- Extract schedules (door, window, finish, fixture, equipment, lighting, panel, hardware)
- Identify scope items organized by **23 CSI trades**
- Count symbols on plan sheets (fixtures, devices, equipment)
- Flag items that require manual measurement (SF, LF, SY areas and lengths)
- Provide source traceability — every item links back to its source sheet and extraction method
- Submit extracted items for automated pricing via an integrated pricing agent
- Export results to CSV

### What It Cannot Do (Yet)

| Limitation | Status |
|------------|--------|
| **Addenda** — Cannot process addendum documents or reconcile changes from addenda | Planned for future |
| **Geotechnical reports** — Does not read or interpret geotech data | Planned for future |
| **Multiple attachments** — Only handles one drawings file + one project manual per run | Planned for future |
| **Supplemental documents** — Cannot process separate submittals, RFIs, or shop drawings | Planned for future |

> **In short:** Upload one drawings PDF and one project manual. That's it for now. Multi-file support and addendum handling are on the roadmap.

---

## Accuracy & Trust Model

### Where the AI Is Reliable

- **Schedule extraction** — The AI reads door schedules, window schedules, finish schedules, fixture schedules, and equipment schedules. These are structured tables and the AI performs well here, especially at 300 DPI. Schedule-sourced items are the **most trustworthy** output.
- **Keynote and note extraction** — Text-based callouts and keynotes on drawings are reliably captured.
- **Trade classification** — Assigning items to the correct CSI trade is generally accurate.

### Where Human Review Is Required

> **Counts and measurements are performed by an AI vision model that can hallucinate. All quantities must be double-checked by a human estimator.**

| Extraction Type | Risk Level | What to Check |
|----------------|------------|---------------|
| **Symbol counting** (EA items on plans) | **Medium-High** — AI may miscount dense plans, miss symbols, or double-count | Verify every EA count against the drawings |
| **Area/length measurements** (SF, LF items) | **Flagged for manual takeoff** — AI does not measure these, it only identifies them | Estimator must measure from plans |
| **Small text in schedules** | **Medium** — Abbreviations like HM, OHD, SC, RB can be misread | Cross-check schedule marks against the original PDF |
| **Existing vs. new items** | **Medium** — Items prefixed with (E) or (P) should be excluded from new scope | Verify no existing items snuck into the BOM |
| **Furnished-by-Owner (FBO)** | **Medium** — FBO items should be "Install only" with no material cost | Confirm FBO items are flagged correctly |
| **Electrical fixture counts** | **High** — Dense electrical plans (E-101, E-102) are the hardest for AI to count accurately | Always recount electrical fixtures manually |

### Confidence Scores

Every extracted item includes a confidence score (0.0–1.0). Items below 0.7 should be treated as suggestions that need verification. Items above 0.9 from schedule sources are generally reliable.

---

## How to Use It — Step by Step

### 1. Upload Documents

1. Open the application in your browser
2. Drag and drop (or click to select) your files:
   - **Required:** One drawings PDF (the construction drawing set)
   - **Required:** One project manual PDF (specifications)
   - **Optional:** CSV or XLSX files with pre-extracted schedule data


### 2. Run the Pipeline

1. Click **"Run Pipeline"**
2. The system processes your documents through 4 automated steps:

| Step | What Happens | Typical Time |
|------|-------------|--------------|
| **Step 1: Classification** 
| **Step 2a: Schedule Extraction** | AI scans drawing pages for tables/schedules, then extracts their content |
| **Step 2b: Sheet Mapping** | AI builds a map of which drawing sheets are relevant to each trade 
| **Step 3: Trade Extraction** | AI extracts scope items per trade using plan images + schedule data 

3. A progress indicator shows which step is running. The page polls for updates every 3 seconds.
4. **Total pipeline time:** typically 15–20 minutes depending on document size.

### 3. Review the Results

Once the pipeline completes, you'll see an editable table with all extracted items:

| Column | Description |
|--------|-------------|
| **Trade** | CSI trade category (e.g., "Doors and Windows", "Electrical") |
| **Description** | Scope item description |
| **Qty** | Extracted quantity (blank if manual measurement required) |
| **Unit** | Unit of measure (EA, SF, LF, LS, etc.) |
| **Mark** | Schedule mark or tag reference |
| **Source** | Where the item was found (e.g., "schedule:Door Schedule", "plan_count:E-121") |
| **Review** | Flag if item needs manual attention |

**What to do:**
- **Review every item** — scan for obviously wrong descriptions or misclassified trades
- **Check all EA quantities** — these came from AI counting and may be wrong
- **Fill in blank quantities** — items marked for manual measurement need you to measure from the plans
- **Delete duplicates** — the AI occasionally extracts the same item from multiple sources
- **Add missing items** — use the "Add Row" button for anything the AI missed
- **Edit freely** — every cell in the table is editable

### 4. Submit for Pricing

1. After review, click **"Submit for Pricing"**
2. The system sends your BOM to an automated pricing agent
3. Wait for pricing results (progress indicator shown)
4. Review priced items — each gets a unit cost, man-hours estimate, and price source

### 5. Export

Click **"Export CSV"** to download the final priced BOM as a spreadsheet for use in your estimate.

---

## Understanding the Output

### Source Types

The `source` field tells you where each item was extracted from:

| Source Pattern | Meaning | Reliability |
|---------------|---------|-------------|
| `schedule:<name>` | From a schedule table (door, window, finish, etc.) | High |
| `plan_count:<sheet_id>` | AI counted symbols on a plan sheet | Medium — verify count |
| `plan_measurement:<sheet_id>` | Item identified on plan, needs manual measurement | N/A — qty blank |
| `keynote:<sheet_id>` | From a keynote callout on drawings | High |
| `detail:<sheet_id>` | From a detail or section view | Medium |
| `note` | From a general note on drawings or specs | High |
| `elevation:<sheet_id>` | From building elevations | Medium |

### Item Flags

- **Blank quantity** — means the item requires manual measurement from plans (area, length, or volume)
- **"Review" column populated** — the AI flagged this item for human attention with a reason
- **Low confidence** (< 0.7) — treat as a suggestion, not a fact

---

## Tips for Best Results

1. **Clean PDFs work best** — scanned documents with low resolution produce worse results. If possible, use the digital/vector PDF from the architect, not a scanned copy.

2. **Check schedules first** — schedule-sourced items are the most reliable. Start your review there and use them as anchors.

3. **Electrical is the weakest trade** — dense electrical plans with many similar symbols are the hardest for AI. Budget extra review time for electrical counts.

4. **Use the mark column** — marks like "D-101", "WC-1", "FCU-3" link items back to schedules. Use these to cross-reference against the original drawings.

5. **Don't trust quantities blindly** — the AI is a first-pass assistant, not a replacement for estimator judgment. It gives you a structured starting point and catches items you might miss, but every number should be verified.

6. **Pre-extracted schedules help** — if you already have schedule data in CSV/XLSX format, uploading it alongside the PDFs improves accuracy since the AI doesn't need to OCR those tables.

---

## Architecture Overview (For Technical Users)

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (Frontend)                     │
│  Upload files → Poll status → Edit BOM → Export CSV      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (polling every 3s)
┌──────────────────────▼──────────────────────────────────┐
│                 FastAPI Backend (main.py)                 │
│  /get-upload-urls → /start → /status → /results          │
│  Pipeline runs in background thread                      │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
   ┌─────────┐  ┌────────────┐  ┌──────────────┐
   │ Step 1   │  │ Step 2a/2b │  │   Step 3     │
   │ Classify │→ │ Schedules  │→ │ Per-Trade    │
   │ Pages    │  │ + Sheet Map│  │ Extraction   │
   └─────────┘  └────────────┘  └──────────────┘
        │              │                  │
        └──────────────┼──────────────────┘
                       ▼
              Google Gemini APIs
         (Pro for complex tasks,
          Flash for table detection)
```

**Tech Stack:** Python, FastAPI, Google Gemini AI (2.5-pro, 2.5-flash, 2.0-flash), PyMuPDF, Google Cloud Run

**Deployment:** Google Cloud Run with 16GB RAM, 8 CPU, 25-minute timeout

---

## FAQ

**Q: How long does a typical run take?**
A: 8–15 minutes for a standard drawing set (40–80 sheets). Larger sets may take up to 20 minutes.

**Q: What if the pipeline fails or times out?**
A: Results are saved to disk as they're generated. Refresh the page — if partial results exist, they'll be shown. You can also click "Force Reset" and try again.

**Q: Can I run multiple projects at once?**
A: Currently one pipeline run at a time per session. Wait for one to complete before starting another.

**Q: What file formats are supported?**
A: PDFs for drawings and specs. CSV, XLSX, and DOCX for supplemental schedule data.

**Q: Are my documents stored permanently?**
A: No. Uploaded files are deleted from cloud storage after the pipeline completes. Results persist only in the current session.

**Q: What trades does it cover?**
A: 23 CSI-based trades: General Requirements, Site Work, Masonry, Concrete, Metals, Rough Carpentry, Finish Carpentry, Plumbing, Electrical, HVAC and Sheet Metals, Insulation, Doors and Windows, Drywall, Cabinets, Stucco and Siding, Painting, Roofing, Tile & Solid Surfaces, Bath and Accessories, Appliances, Flooring, Fire Sprinklers, and Landscaping.
