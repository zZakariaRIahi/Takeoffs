"""Step 1 — Document Ingestion + Classification.

Pipeline:
  1. ingest_files()                — extract ZIPs, read PDFs (PyMuPDF), render page images
  2. classify_file_with_gemini()   — upload PDF to Gemini File Search store,
                                     ask gemini-2.5-pro to classify into 8 categories.
                                     Gemini handles chunking/indexing internally — no token limits.
     2b. _classify_with_vision()   — fallback if File Search / Files API fails:
                                     sends sampled page images to Gemini vision instead.
  3. Post-classification           — set has_drawings flag per page based on model's
                                     drawing page identification (not all pages in file)
  4. Return DocumentClassificationResult ready for Steps 2-4
"""
from __future__ import annotations

import io
import json
import logging
import re
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple

import fitz  # PyMuPDF
from google.genai import types

from app.config.settings import settings
from app.core.document_classification import (
    ClassifiedFile,
    DocumentCategory,
    DocumentClassificationResult,
    PageInfo,
)
from app.utils.genai_client import get_genai_client

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════════

DRAWING_DPI = 200       # render DPI for all pages (used by vision in Step 3c)

MAX_PAGES_FOR_SINGLE_UPLOAD = 300  # above this → split PDF into 3 chunks
NUM_CHUNKS = 3                     # number of equal PDF chunks for large files

MAX_VISION_PAGES = 20   # max pages to sample for vision fallback classification
MAX_FILE_SEARCH_BYTES = 20 * 1024 * 1024  # 20 MB — skip File Search for larger files
MAX_CLASSIFY_WORKERS = 4  # max concurrent classification calls to avoid rate limits

CLASSIFICATION_PROMPT = """You are a construction bid document classifier.

Analyze the uploaded document and provide TWO things:
1. Which document categories it belongs to
2. Which specific pages (if any) are construction drawing sheets

═══ CATEGORIES (assign one or more) ═══

1. cover_sheet — project title page, table of contents, summary
2. instructions_to_bidder — bidding procedures, submission requirements
3. project_specifications — technical specs (CSI Divisions 01-49), materials, methods
4. construction_drawings — the document contains actual construction drawing sheets
5. general_conditions — AIA A201, standard contract conditions
6. special_conditions — project-specific amendments to general conditions
7. bid_form — proposal forms, bid schedules, pricing sheets
8. bid_security — bid bond forms, surety requirements

═══ DRAWING PAGE IDENTIFICATION ═══

Construction drawing sheets are pages with:
- Sheet numbers in title blocks (A1.0, S-100, M-001, E-101, P-500, L1.0, etc.)
- Primarily graphical content: plans, elevations, sections, details, schedules on drawing sheets
- Standard drawing title blocks along the bottom or right edge
- Minimal body text (just callouts, notes, legends — NOT paragraphs of specifications)

NOT drawing sheets: text-heavy spec pages, bid forms, conditions, general requirements,
tables of contents — even if they contain small embedded figures or diagrams.

═══ RULES ═══

- A document can belong to MULTIPLE categories.
- Search through the ENTIRE document, not just the beginning.
- Do NOT return categories that have zero evidence.
- Only include "construction_drawings" in categories if you identify actual drawing pages.
- For drawing_pages: return page numbers (1-based) of ALL pages that are actual drawing sheets.
  If the ENTIRE file is a drawing set, return "all" instead of listing every page number.
  If there are no drawing pages, return an empty array [].

Return ONLY a JSON object (no markdown fences, no extra text):
{"categories": ["project_specifications", "general_conditions"], "drawing_pages": []}
{"categories": ["construction_drawings"], "drawing_pages": "all"}
{"categories": ["project_specifications", "construction_drawings"], "drawing_pages": [195, 196, 197]}
"""


# ════════════════════════════════════════════════════════════════════════════════
# 1. FILE INGESTION
# ════════════════════════════════════════════════════════════════════════════════

def ingest_files(
    file_contents: List[Tuple[str, bytes]],
) -> List[ClassifiedFile]:
    """Read uploaded files → extract text + render images per page.

    Handles: PDFs (via PyMuPDF), ZIPs (extract then recurse), images, text files.
    Files are ingested in parallel (PyMuPDF releases the GIL during rendering).
    Global page numbers are assigned sequentially after all files are processed.
    """
    # Flatten ZIPs first
    flat_files: List[Tuple[str, bytes]] = []
    for filename, content in file_contents:
        ext = Path(filename).suffix.lower()
        if ext == ".zip":
            extracted = _extract_zip(content, filename)
            flat_files.extend(extracted)
        else:
            flat_files.append((filename, content))

    def _ingest_one(filename: str, content: bytes) -> Optional[ClassifiedFile]:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            pages, _ = _process_pdf(content, filename, 0)
        elif ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            pages = [PageInfo(page_number=0, global_page_number=0,
                              has_drawings=True, image_bytes=content)]
        elif ext in (".txt", ".csv", ".rtf"):
            text = content.decode("utf-8", errors="replace")
            pages = [PageInfo(page_number=0, global_page_number=0,
                              extracted_text=text)]
        else:
            logger.warning(f"Skipping unsupported file: {filename}")
            return None
        logger.info(f"Ingested {filename}: {len(pages)} pages")
        return ClassifiedFile(filename=filename, pages=pages)

    # Ingest all files in parallel
    ordered_results: List[Optional[ClassifiedFile]] = [None] * len(flat_files)
    with ThreadPoolExecutor(max_workers=len(flat_files)) as executor:
        futures = {
            executor.submit(_ingest_one, fn, data): i
            for i, (fn, data) in enumerate(flat_files)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                ordered_results[idx] = future.result()
            except Exception as e:
                fn = flat_files[idx][0]
                logger.error(f"Ingestion failed for {fn}: {e}")

    # Assign global page numbers sequentially (preserves file order)
    results: List[ClassifiedFile] = []
    global_page_counter = 0
    for cf in ordered_results:
        if cf is None:
            continue
        for p in cf.pages:
            p.global_page_number = global_page_counter
            global_page_counter += 1
        results.append(cf)

    return results


# ════════════════════════════════════════════════════════════════════════════════
# 2. GEMINI FILE SEARCH CLASSIFICATION (8 categories)
# ════════════════════════════════════════════════════════════════════════════════

def classify_file_with_gemini(
    cf: ClassifiedFile,
    file_bytes: bytes,
) -> Tuple[Set[DocumentCategory], List[int]]:
    """Classify a file using Gemini File Search.

    Returns (categories, drawing_page_numbers) where drawing_page_numbers
    is a list of 1-based page numbers within this file that are drawing sheets.

    • ≤ 300 pages → single upload to File Search store
    • > 300 pages → split PDF into 3 equal chunks, classify each, union results
    """
    total = len(cf.pages)
    file_size = len(file_bytes)
    is_pdf = cf.filename.lower().endswith(".pdf")

    drawing_pages: List[int] = []

    # Large files → skip File Search, use Files API
    if file_size > MAX_FILE_SEARCH_BYTES:
        logger.info(
            f"Classifying {cf.filename} ({total} pages, {file_size/1024/1024:.1f} MB) "
            f"with Files API (too large for File Search)"
        )
        cats, drawing_pages = _classify_single_pdf_with_files_api(
            file_bytes, cf.filename, total_pages=total
        )
    # Image-heavy PDFs (drawings) → use Files API (File Search hangs on these)
    elif is_pdf and _is_image_heavy_pdf(cf):
        cats, drawing_pages = _classify_single_pdf_with_files_api(
            file_bytes, cf.filename, total_pages=total
        )
    elif is_pdf and total > MAX_PAGES_FOR_SINGLE_UPLOAD:
        logger.info(f"Classifying {cf.filename} ({total} pages) with Gemini File Search…")
        # Split PDF into 3 equal sub-PDFs, classify each chunk separately
        chunk_size = total // NUM_CHUNKS
        remainder = total % NUM_CHUNKS
        chunk_pdfs = _split_pdf(file_bytes, NUM_CHUNKS)
        logger.info(f"  Split into {len(chunk_pdfs)} chunks for classification")

        cats: Set[DocumentCategory] = set()
        page_start = 0
        for i, chunk_bytes in enumerate(chunk_pdfs):
            chunk_pages = chunk_size + (1 if i < remainder else 0)
            label = f"{cf.filename} [chunk {i+1}/{len(chunk_pdfs)}]"
            chunk_cats, chunk_drawing = _classify_single_pdf_with_gemini(
                chunk_bytes, label, total_pages=chunk_pages
            )
            cats |= chunk_cats

            # Offset chunk-relative drawing pages to file-level page numbers
            for dp in chunk_drawing:
                drawing_pages.append(page_start + dp)

            # Tag each page in this chunk with its categories
            page_end = page_start + chunk_pages
            for p in cf.pages[page_start:page_end]:
                p.categories = chunk_cats
            logger.info(
                f"  Chunk {i+1} (pages {page_start+1}–{page_end}): "
                f"{sorted(c.value for c in chunk_cats)}, "
                f"{len(chunk_drawing)} drawing pages"
            )
            page_start = page_end
    else:
        logger.info(f"Classifying {cf.filename} ({total} pages) with Gemini File Search…")
        cats, drawing_pages = _classify_single_pdf_with_gemini(
            file_bytes, cf.filename, total_pages=total
        )

    if not cats:
        cats, drawing_pages = _classify_with_vision(cf, cf.filename, raw_pdf_bytes=file_bytes)

    cf.categories = cats
    drawing_pages = sorted(set(drawing_pages))
    logger.info(
        f"  → {cf.filename}: {sorted(c.value for c in cats)}, "
        f"{len(drawing_pages)} drawing pages"
    )
    return cats, drawing_pages


def _split_pdf(pdf_bytes: bytes, num_chunks: int) -> List[bytes]:
    """Split a PDF into num_chunks equal sub-PDFs using PyMuPDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = len(doc)
    chunk_size = total // num_chunks
    remainder = total % num_chunks

    chunks: List[bytes] = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        # Create sub-PDF
        sub_doc = fitz.open()
        sub_doc.insert_pdf(doc, from_page=start, to_page=end - 1)
        chunks.append(sub_doc.tobytes())
        sub_doc.close()
        logger.info(f"    Chunk {i+1}: pages {start+1}–{end} ({end - start} pages)")
        start = end

    doc.close()
    return chunks


def _classify_single_pdf_with_gemini(
    file_bytes: bytes,
    label: str,
    total_pages: int = 0,
) -> Tuple[Set[DocumentCategory], List[int]]:
    """Upload a single PDF to Gemini File Search, classify, clean up.

    Returns (categories, drawing_page_numbers) where drawing_page_numbers
    is a list of 1-based page numbers identified as construction drawings.

    1. Create File Search store
    2. Upload PDF → Gemini indexes internally
    3. generate_content with file_search tool
    4. Parse categories + drawing pages
    5. Delete store
    """
    client = get_genai_client()
    model = settings.CLASSIFICATION_MODEL
    store_name = None

    try:
        # 1. Create store
        store = client.file_search_stores.create(
            config={"display_name": f"classify-{label[:50]}"}
        )
        store_name = store.name

        # 2. Upload to store
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        operation = client.file_search_stores.upload_to_file_search_store(
            file=tmp_path,
            file_search_store_name=store_name,
            config={"display_name": label},
        )

        logger.info(f"  Uploading & indexing {label}…")
        while not operation.done:
            time.sleep(3)
            operation = client.operations.get(operation)
        logger.info(f"  Indexing complete for {label}")

        Path(tmp_path).unlink(missing_ok=True)

        # 3. Classify
        response = client.models.generate_content(
            model=model,
            contents=f"Classify this document ({total_pages} pages): {label}\n\n{CLASSIFICATION_PROMPT}",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store_name]
                        )
                    )
                ],
                temperature=0,
            ),
        )

        raw = response.text or "{}"
        logger.info(f"  Gemini raw response for {label}: {raw[:300]}")

        return _parse_classification_response(raw, total_pages)

    except Exception as e:
        logger.error(f"Gemini classification failed for {label}: {e}")
        return set(), []
    finally:
        if store_name:
            try:
                client.file_search_stores.delete(name=store_name, config={"force": True})
            except Exception as e:
                logger.warning(f"  Failed to delete store {store_name}: {e}")


def _classify_single_pdf_with_files_api(
    file_bytes: bytes,
    label: str,
    total_pages: int = 0,
) -> Tuple[Set[DocumentCategory], List[int]]:
    """Upload PDF via Files API and classify with generateContent.

    Unlike File Search (which hangs on image-heavy PDFs), the Files API
    reliably processes any PDF. The model sees the full document content
    including images, diagrams, and drawings.

    Limit: 50 MB per PDF (Files API cap).
    """
    client = get_genai_client()
    model = settings.CLASSIFICATION_MODEL

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        logger.info(f"  Uploading {label} via Files API ({len(file_bytes)/1024/1024:.1f} MB)…")
        uploaded = client.files.upload(file=tmp_path)

        # Poll until ACTIVE
        max_wait = 120
        waited = 0
        while uploaded.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(3)
            waited += 3
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"Files API upload failed: state={uploaded.state.name}")

        logger.info(f"  Files API upload ACTIVE for {label}")

        # Classify — pass the uploaded file directly to generateContent
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded.uri,
                            mime_type="application/pdf",
                        ),
                        types.Part.from_text(
                            text=f"Classify this document ({total_pages} pages): {label}\n\n{CLASSIFICATION_PROMPT}"
                        ),
                    ]
                )
            ],
            config=types.GenerateContentConfig(temperature=0),
        )

        raw = response.text or "{}"
        logger.info(f"  Gemini raw response for {label}: {raw[:300]}")

        return _parse_classification_response(raw, total_pages)

    except Exception as e:
        logger.error(f"Files API classification failed for {label}: {e}")
        return set(), []
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Clean up uploaded file
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def _is_image_heavy_pdf(cf: ClassifiedFile) -> bool:
    """Check if a PDF is mostly images (low text density).

    Image-heavy PDFs (like construction drawings) cause File Search
    to hang indefinitely. These should use Files API instead.
    """
    if not cf.pages:
        return False
    total_text = sum(len(p.extracted_text.strip()) for p in cf.pages)
    avg_chars_per_page = total_text / len(cf.pages)
    # Text-heavy PDFs (specs) typically have 1000+ chars/page
    # Drawing PDFs typically have < 200 chars/page
    is_heavy = avg_chars_per_page < 300
    if is_heavy:
        logger.info(
            f"  {cf.filename}: image-heavy PDF ({avg_chars_per_page:.0f} chars/page avg) "
            f"→ using Files API instead of File Search"
        )
    return is_heavy


def _classify_with_vision(
    cf: ClassifiedFile,
    label: str,
    raw_pdf_bytes: Optional[bytes] = None,
) -> Tuple[Set[DocumentCategory], List[int]]:
    """Fallback: classify by sending sampled page images to Gemini vision.

    Used when File Search upload fails (e.g., large files hitting 503/400).
    Samples up to MAX_VISION_PAGES evenly across the document.
    Renders images on-the-fly for sampled pages if not already rendered.
    Returns (categories, drawing_page_numbers).
    """
    # Render images on-the-fly for sampled pages if needed
    pages_with_images = [p for p in cf.pages if p.image_bytes]
    if not pages_with_images and raw_pdf_bytes and cf.filename.lower().endswith(".pdf"):
        # Pick sample indices first, then render only those
        total = len(cf.pages)
        n_sample = min(total, MAX_VISION_PAGES)
        sample_indices = [int(i * total / n_sample) for i in range(n_sample)]
        zoom = DRAWING_DPI / 72
        doc = fitz.open(stream=raw_pdf_bytes, filetype="pdf")
        for idx in sample_indices:
            if idx < len(doc) and idx < len(cf.pages):
                pix = doc[idx].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                cf.pages[idx].image_bytes = pix.tobytes("jpeg", 85)
        doc.close()
        pages_with_images = [p for p in cf.pages if p.image_bytes]

    if not pages_with_images:
        return set(), []

    total = len(pages_with_images)
    if total <= MAX_VISION_PAGES:
        sample = pages_with_images
    else:
        indices = [int(i * total / MAX_VISION_PAGES) for i in range(MAX_VISION_PAGES)]
        sample = [pages_with_images[i] for i in indices]

    logger.info(f"  Vision fallback for {label}: sending {len(sample)}/{total} page images")

    client = get_genai_client()
    model = settings.CLASSIFICATION_MODEL

    # Tell the model which page numbers it's seeing so it can report drawing pages
    page_num_info = ", ".join(str(p.page_number + 1) for p in sample)
    contents: list = [
        f"Classify this document ({total} pages total): {label}\n"
        f"I'm showing you {len(sample)} of {total} pages. "
        f"The page numbers shown are: {page_num_info}\n\n"
        f"IMPORTANT: For drawing_pages, identify which of the shown pages are "
        f"construction drawing sheets. If ALL shown pages are drawings, return \"all\" "
        f"(meaning the entire file is a drawing set). Otherwise list the specific "
        f"page numbers (1-based) that are drawings.\n\n"
        f"{CLASSIFICATION_PROMPT}"
    ]
    for p in sample:
        contents.append(f"--- Page {p.page_number + 1} ---")
        contents.append(
            types.Part.from_bytes(data=p.image_bytes, mime_type="image/jpeg")
        )

    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0),
        )
        raw = response.text or "{}"
        logger.info(f"  Vision fallback response for {label}: {raw[:300]}")
        return _parse_classification_response(raw, total)
    except Exception as e:
        logger.error(f"  Vision fallback failed for {label}: {e}")
        return set(), []


def classify_documents(
    file_contents: List[Tuple[str, bytes]],
) -> DocumentClassificationResult:
    """Full Step 1 entry point: ingest → Gemini classify → set page flags."""
    # Keep raw bytes for uploading to Gemini
    raw_bytes_map = {name: data for name, data in file_contents}

    files = ingest_files(file_contents)

    # Classify files in parallel (each file's Gemini calls are independent)
    def _classify_one(
        cf: ClassifiedFile,
    ) -> Tuple[str, Set[DocumentCategory], List[int]]:
        file_bytes = raw_bytes_map.get(cf.filename, b"")
        cats, drawing_pages = classify_file_with_gemini(cf, file_bytes)
        return cf.filename, cats, drawing_pages

    file_results: dict[str, Tuple[Set[DocumentCategory], List[int]]] = {}
    with ThreadPoolExecutor(max_workers=MAX_CLASSIFY_WORKERS) as executor:
        futures = {executor.submit(_classify_one, cf): cf for cf in files}
        for future in as_completed(futures):
            cf = futures[future]
            try:
                _, cats, drawing_pages = future.result()
                file_results[cf.filename] = (cats, drawing_pages)
            except Exception as e:
                logger.error(f"  Classification failed for {cf.filename}: {e}")
                file_results[cf.filename] = (set(), [])

    # Apply drawing flags using model-identified page numbers
    for cf in files:
        cats, drawing_pages = file_results.get(cf.filename, (set(), []))

        if drawing_pages:
            # Convert 1-based drawing page numbers to a set for fast lookup
            drawing_set = set(drawing_pages)
            flagged = 0
            for p in cf.pages:
                # p.page_number is 0-based within file, drawing_pages are 1-based
                if (p.page_number + 1) in drawing_set:
                    p.has_drawings = True
                    flagged += 1
            cf.has_visual_content = flagged > 0
            logger.info(
                f"  {cf.filename}: {flagged}/{len(cf.pages)} pages flagged as drawings"
            )
        else:
            logger.info(f"  {cf.filename}: 0 drawing pages")

    # Render images only for drawing pages (not all 270 pages)
    render_drawing_images(files, raw_bytes_map)

    result = DocumentClassificationResult(
        files=files,
        raw_pdf_bytes=raw_bytes_map,
    )
    logger.info(result.summary())
    return result


# ════════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _process_pdf(
    content: bytes,
    filename: str,
    global_offset: int,
) -> Tuple[List[PageInfo], int]:
    """Extract text per page. Images are NOT rendered here to save memory.

    The raw PDFs are sent directly to Gemini File Search for classification.
    After classification, render_drawing_images() renders images only for
    pages flagged as drawings (~40 pages instead of all 270).
    """
    doc = fitz.open(stream=content, filetype="pdf")
    pages: List[PageInfo] = []

    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        pages.append(PageInfo(
            page_number=i,
            global_page_number=global_offset + i,
            extracted_text=text,
        ))

    doc.close()
    return pages, global_offset + len(pages)


def render_drawing_images(
    files: List[ClassifiedFile],
    raw_pdf_bytes: dict,
) -> None:
    """Render images ONLY for pages flagged as drawings (in-place).

    Called after classification so we skip the 230 spec pages entirely.
    Uses parallel threads for speed (PyMuPDF releases GIL during rendering).
    """
    zoom = DRAWING_DPI / 72
    for cf in files:
        if not cf.filename.lower().endswith(".pdf"):
            continue
        pdf_bytes = raw_pdf_bytes.get(cf.filename)
        if not pdf_bytes:
            continue

        drawing_pages = [p for p in cf.pages if p.has_drawings]
        if not drawing_pages:
            continue

        logger.info(f"Rendering {len(drawing_pages)} drawing page images for {cf.filename} (parallel)")

        def _render_one(page_info):
            # Each thread opens its own doc (PyMuPDF is not thread-safe)
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            if page_info.page_number < len(doc):
                pix = doc[page_info.page_number].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                page_info.image_bytes = pix.tobytes("jpeg", 85)
            doc.close()

        with ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(_render_one, drawing_pages)

        logger.info(f"  Rendered {len(drawing_pages)} pages")


def _parse_classification_response(
    raw: str,
    total_pages: int,
) -> Tuple[Set[DocumentCategory], List[int]]:
    """Parse Gemini response into categories + drawing page numbers.

    Returns (categories, drawing_page_numbers) where drawing_page_numbers
    is a list of 1-based page numbers.

    Handles:
    - New format: {"categories": [...], "drawing_pages": [...] | "all"}
    - Legacy format: ["cover_sheet", "project_specifications"]
    - Markdown-fenced JSON
    """
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        raw = raw.strip()

    parsed = None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find JSON object first, then array
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                pass
        if parsed is None:
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

    if parsed is None:
        logger.warning(f"Could not parse classification response: {raw[:300]}")
        return set(), []

    # Extract categories
    if isinstance(parsed, dict):
        cat_list = parsed.get("categories", [])
    elif isinstance(parsed, list):
        # Legacy format: just a list of categories
        cat_list = parsed
    else:
        logger.warning(f"Unexpected classification format: {type(parsed)}")
        return set(), []

    cats: Set[DocumentCategory] = set()
    for item in cat_list:
        item_str = str(item).strip().lower()
        try:
            cats.add(DocumentCategory(item_str))
        except ValueError:
            logger.warning(f"Unknown category: {item_str}")

    # Extract drawing pages
    drawing_pages: List[int] = []
    if isinstance(parsed, dict):
        dp = parsed.get("drawing_pages", [])
        if dp == "all" or dp == "ALL":
            drawing_pages = list(range(1, total_pages + 1))
        elif isinstance(dp, list):
            for p in dp:
                if isinstance(p, int) and 1 <= p <= total_pages:
                    drawing_pages.append(p)
                elif isinstance(p, (int, float)):
                    pn = int(p)
                    if 1 <= pn <= total_pages:
                        drawing_pages.append(pn)

    return cats, sorted(set(drawing_pages))



def _extract_zip(content: bytes, filename: str) -> List[Tuple[str, bytes]]:
    """Extract supported files from a ZIP archive."""
    supported = set(settings.SUPPORTED_FILE_TYPES)
    results: List[Tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name
                if name.startswith(".") or name.startswith("__"):
                    continue
                ext = Path(name).suffix.lower().lstrip(".")
                if ext == "zip" or ext not in supported:
                    continue
                results.append((name, zf.read(info.filename)))
    except Exception as e:
        logger.error(f"Failed to extract ZIP {filename}: {e}")
    return results
