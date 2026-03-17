"""Google Generative AI client initialization."""
import os
from google import genai
from app.config.settings import settings


def get_genai_client():
    """Get initialized Gemini client."""
    api_key = settings.GOOGLE_API_KEY or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return genai.Client(api_key=api_key)
