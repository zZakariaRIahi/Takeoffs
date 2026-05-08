"""Walkthrough Photo Analyzer — Extract scope items from site walkthrough photos.

Analyzes JPG/PNG site photos to identify existing conditions that generate
additional scope items (patching, demolition, remediation, etc.) not visible
on drawings.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, List, Tuple

from google import genai
from google.genai import types as genai_types

from app.core.estimate_models import EstimateItem

logger = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
MAX_PHOTOS_PER_BATCH = 20  # Flash handles up to 20 images per call reliably

_CONF_MAP = {"high": 0.85, "medium": 0.65, "low": 0.45}


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


WALKTHROUGH_PROMPT = """\
You are a SENIOR CONSTRUCTION ESTIMATOR reviewing site walkthrough photos taken
before renovation work begins. These photos document EXISTING CONDITIONS.

Your job:
1. Identify the space type in each photo (hallway, classroom, toilet room, etc.)
2. Detect existing conditions that add scope: damage, moisture, failing finishes,
   fixtures to remove/replace, accessibility issues, utility concerns
3. Generate scope items based on what you actually see

For each photo, extract scope items that a contractor must price but that
drawings alone would not capture — field conditions that only a site visit reveals.

Focus on:
  Surfaces      — peeling paint, staining, water damage, cracks → patching/prep
  Moisture      — water damage, efflorescence, staining → remediation, waterproofing
  Demolition    — existing fixtures, casework, finishes to remove
  Ceilings      — damaged tiles, staining, exposed structure, missing sections
  Floors        — worn finish, transitions, damage, subfloor issues
  Utilities     — exposed conduit, damaged piping, junction boxes, accessibility
  Accessibility — ADA compliance issues visible in the space

RULES:
- Only extract items you can actually SEE in the photos — no assumptions
- qty: set to null when condition is visible but not measurable from a photo
- confidence: always "low" for photo-derived items (no precise measurements possible)
- source: must be "walkthrough_photo:[filename]"
- review: always describe what to measure from plan to get actual qty
- These items SUPPLEMENT drawing-based items — they capture what drawings miss

Return a JSON array. Each item:
{
  "trade": "Painting",
  "description": "Patch and prime water-damaged ceiling — moisture staining visible",
  "qty": null,
  "unit": "SF",
  "source": "walkthrough_photo:2880.jpg",
  "method": "Observed: significant water staining on ceiling tiles, ~15-20% of ceiling area affected in NW hallway section",
  "confidence": "low",
  "material_spec": "",
  "review": "Measure affected ceiling area from floor plan — confirm extent of moisture damage before pricing"
}

Return ONLY the JSON array. No markdown fences, no commentary.
"""


def analyze_walkthrough_photos(
    photos: List[Tuple[str, bytes]],
) -> List[EstimateItem]:
    """Analyze walkthrough photos and return condition-based scope items.

    Args:
        photos: List of (filename, image_bytes) tuples

    Returns:
        List of EstimateItem with source="walkthrough_photo:..."
    """
    if not photos:
        return []

    t0 = time.time()
    client = _get_client()
    logger.info(f"Walkthrough analysis: {len(photos)} photos")

    all_items: List[EstimateItem] = []

    for batch_start in range(0, len(photos), MAX_PHOTOS_PER_BATCH):
        batch = photos[batch_start: batch_start + MAX_PHOTOS_PER_BATCH]
        batch_num = batch_start // MAX_PHOTOS_PER_BATCH + 1
        items = _analyze_batch(client, batch)
        all_items.extend(items)
        logger.info(f"  Batch {batch_num}: {len(items)} items from {len(batch)} photos")

    elapsed = time.time() - t0
    logger.info(f"Walkthrough analysis done: {len(all_items)} items in {elapsed:.1f}s")
    return all_items


def _analyze_batch(
    client: genai.Client,
    photos: List[Tuple[str, bytes]],
) -> List[EstimateItem]:
    """Send one batch of photos to Flash and parse results."""
    contents: List[Any] = [WALKTHROUGH_PROMPT]

    for filename, img_bytes in photos:
        contents.append(f"\n--- Photo: {filename} ---")
        mime = "image/jpeg"
        if filename.lower().endswith(".png"):
            mime = "image/png"
        elif filename.lower().endswith(".webp"):
            mime = "image/webp"
        contents.append(genai_types.Part.from_bytes(data=img_bytes, mime_type=mime))

    response = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(temperature=0.1),
            )
            break
        except Exception as e:
            err_str = str(e)
            if attempt < 2 and ("503" in err_str or "UNAVAILABLE" in err_str or "SSL" in err_str or "EOF" in err_str):
                wait = 15 * (attempt + 1)
                logger.warning(f"  Photo batch attempt {attempt+1} failed, retrying in {wait}s...")
                time.sleep(wait)
            else:
                logger.error(f"  Photo batch failed: {e}")
                return []

    if not response:
        return []

    raw = (response.text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                logger.error(f"  Photo batch JSON parse failed: {raw[:200]}")
                return []
        else:
            logger.error(f"  No JSON array in photo response: {raw[:200]}")
            return []

    if not isinstance(parsed, list):
        return []

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

        unit = d.get("unit", "LS")
        conf_str = d.get("confidence", "low")
        confidence = _CONF_MAP.get(conf_str, 0.45)
        needs_measurement = qty is None and unit in ("SF", "LF", "SY", "CY")

        items.append(EstimateItem(
            trade=d.get("trade", "General Requirements"),
            item_description=desc,
            qty=qty,
            unit=unit,
            extraction_method=d.get("method", ""),
            confidence=confidence,
            source=d.get("source", "walkthrough_photo"),
            material_spec=d.get("material_spec", ""),
            needs_measurement=needs_measurement,
            needs_counting=False,
            review_reason=d.get("review", ""),
            notes=d.get("method", ""),
        ))

    return items
