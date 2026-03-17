"""Application configuration settings.

v4.2 Architecture:
- Company focus: GOVERNMENT WORK (prevailing wage projects)
- Unified labor rate: $101.85/hr for all trades (no variations)
- QTO Agent: Extracts project info + performs quantity takeoff with Gemini
- Audit Agent: Validates quantities using OpenAI Vision (skips GR items)
- HITL QTO Review: Human reviews quantities AND prices GR items manually
- Aggregator: Material prices from web search + labor from production rates
- HITL Pricing Review: Human reviews and approves final pricing
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ══════════════════════════════════════════════════════════════════════════
    # APPLICATION
    # ══════════════════════════════════════════════════════════════════════════
    APP_NAME: str = "AI Construction Bid Estimator"
    VERSION: str = "6.0.0"  # v6.0: 9-Agent takeoff architecture (context-safe, WP-exact)
    DEBUG: bool = False

    # ══════════════════════════════════════════════════════════════════════════
    # API KEYS
    # ══════════════════════════════════════════════════════════════════════════
    GOOGLE_API_KEY: str = ""  # For Gemini (QTO extraction)
    OPENAI_API_KEY: str = ""  # For Vision audit + web search pricing

    # ══════════════════════════════════════════════════════════════════════════
    # CORS
    # ══════════════════════════════════════════════════════════════════════════
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3003", "http://localhost:5173"]

    # ══════════════════════════════════════════════════════════════════════════
    # FILE PROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    SUPPORTED_FILE_TYPES: List[str] = [
        # Construction documents (primary)
        "pdf",
        # Images (for individual drawing sheets)
        "jpg", "jpeg", "png", "gif", "webp", "bmp",
        # Supplementary documents
        "docx", "doc", "txt", "rtf",
        # Spreadsheets (specs, schedules)
        "xlsx", "xls", "csv",
        # Archives (will be extracted)
        "zip",
    ]
    MAX_FILE_SIZE_MB: int = 1000  # Large construction PDFs supported

    # ══════════════════════════════════════════════════════════════════════════
    # UNIFIED LABOR RATE
    # ══════════════════════════════════════════════════════════════════════════
    # All trades use the same rate - $101.85/hr (company standard)
    # This is an all-in rate including burden, benefits, taxes
    UNIFIED_LABOR_RATE: float = 101.85

    # Legacy settings (kept for compatibility but not used)
    DEFAULT_LABOR_RATE: float = 101.85  # Same as unified rate
    DEFAULT_WASTE_FACTOR: float = 1.10  # 10% waste factor for materials

    # ══════════════════════════════════════════════════════════════════════════
    # QTO (QUANTITY TAKEOFF) SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    # QTO is the primary focus - accurate quantity extraction from drawings
    ENABLE_QTO_VALIDATION: bool = True  # Validate quantities (negative, unrealistic)
    QTO_CONFIDENCE_THRESHOLD: float = 0.5  # Min confidence to proceed without warning

    # Multi-stage QTO Pipeline (v5.0)
    # Replaces monolithic QTO with 4-stage pipeline:
    # 1. Scope Discovery - identifies work packages from DrawingIndex
    # 2. Work Package Extraction - extracts each WP in parallel
    # 3. Coverage QA - validates completeness
    # 4. Trade Aggregation - merges to trade-organized output
    ENABLE_MULTISTAGE_QTO: bool = False  # Feature flag for new pipeline

    # 9-Agent Takeoff Pipeline (v6.0)
    # Context-safe, WP-exact architecture with:
    # - Subtask decomposition for controlled extraction
    # - Evidence-before-quantity rule enforcement
    # - Deterministic quantification (code-first, LLM for ambiguity)
    # - Iterative gap detection and targeted re-runs
    # - Cross-trade handshake support
    ENABLE_NINE_AGENT_PIPELINE: bool = False  # Feature flag for 9-agent architecture

    # New 4-Step Pipeline (v7.0)
    # Replaces 9-agent with 4 clear sequential steps, OpenAI-only
    # Step 1: Document Classification (gpt-4o)
    # Step 2: Scope Extraction (gpt-4o, 2 calls)
    # Step 3: Work Package Extraction (gpt-4o, multimodal)
    # Step 4: Quantity Takeoff per WP (gpt-4o, multimodal)
    ENABLE_NEW_PIPELINE: bool = True  # Feature flag for new 4-step pipeline
    CLASSIFICATION_MODEL: str = "gemini-2.5-pro"  # Model for document classification (Step 1)

    # ══════════════════════════════════════════════════════════════════════════
    # AUDIT AGENT SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    ENABLE_AUDIT_AGENT: bool = True  # Enable vision-based audit
    ENABLE_VISION_AUDIT: bool = True  # Use OpenAI Vision for page verification
    AUDIT_MAX_PARALLEL_PAGES: int = 5  # Max concurrent vision API calls
    AUDIT_VISION_MODEL: str = "gpt-4o"  # OpenAI model for vision audit

    # ══════════════════════════════════════════════════════════════════════════
    # MATERIAL PRICING SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    # Material prices are searched from retail sites (Home Depot, Lowe's, Menards)
    ENABLE_WEB_PRICING: bool = True  # Search web for material prices
    MATERIAL_MARKUP: float = 1.10  # 10% contractor markup on materials
    RATE_LIMIT_DELAY: float = 0.1  # Seconds between pricing requests

    # ══════════════════════════════════════════════════════════════════════════
    # COST VALIDATION
    # ══════════════════════════════════════════════════════════════════════════
    ENABLE_COST_VALIDATION: bool = True  # Validate cost proportions
    COST_CONFIDENCE_THRESHOLD: float = 0.5  # Min confidence for validation
    MIN_COST_PER_SF: float = 25.0  # Minimum reasonable cost per SF
    MAX_COST_PER_SF: float = 2000.0  # Maximum reasonable cost per SF
    MIN_PROJECT_COST: float = 1000.0  # Minimum project cost

    # ══════════════════════════════════════════════════════════════════════════
    # CONTINGENCY SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    ENABLE_AUTO_CONTINGENCY: bool = True  # Calculate contingency automatically
    MIN_CONTINGENCY_PERCENT: float = 5.0  # Minimum contingency %
    MAX_CONTINGENCY_PERCENT: float = 35.0  # Maximum contingency %
    DEFAULT_DESIGN_CONTINGENCY: float = 10.0  # Default design contingency %
    DEFAULT_CONSTRUCTION_CONTINGENCY: float = 5.0  # Default construction contingency %

    # ══════════════════════════════════════════════════════════════════════════
    # AI MODELS
    # ══════════════════════════════════════════════════════════════════════════
    # Gemini models (for QTO extraction - handles large documents well)
    QTO_MODEL: str = "gemini-2.5-pro"  # Best for detailed quantity takeoff

    # OpenAI models
    OPENAI_PRICING_MODEL: str = "gpt-4o-search-preview"  # Web search for prices
    OPENAI_VISION_MODEL: str = "gpt-4o"  # Vision audit

    # Legacy (not used in v4.2 architecture - QTO handles extraction)
    EXTRACTOR_MODEL: str = "gemini-2.0-flash"

    # ══════════════════════════════════════════════════════════════════════════
    # PARALLEL PROCESSING
    # ══════════════════════════════════════════════════════════════════════════
    MAX_PARALLEL_TRADES: int = 4  # Max trades to price simultaneously
    MAX_RETRIES: int = 3  # Max retries for API failures
    RETRY_DELAY: float = 1.0  # Seconds between retries

    # ══════════════════════════════════════════════════════════════════════════
    # CONTEXT SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    MAX_PROJECT_REPORT_LENGTH: int = 10000  # Max chars for project context
    MAX_SPEC_CONTEXT_LENGTH: int = 5000  # Max chars for spec context per item

    # ══════════════════════════════════════════════════════════════════════════
    # JOB MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════════
    JOB_EXPIRY_HOURS: int = 2  # Jobs expire after 2 hours (keeps memory clean)

    # ══════════════════════════════════════════════════════════════════════════
    # DATABASE (Postgres for job history)
    # ══════════════════════════════════════════════════════════════════════════
    DATABASE_URL: Optional[str] = None  # Postgres connection URL

    # ══════════════════════════════════════════════════════════════════════════
    # HUMAN-IN-THE-LOOP (HITL) SETTINGS
    # ══════════════════════════════════════════════════════════════════════════
    ENABLE_HITL_QTO_REVIEW: bool = True  # Pause for human QTO review
    ENABLE_HITL_PRICING_REVIEW: bool = True  # Pause for human pricing review
    HITL_TIMEOUT_MINUTES: int = 60  # Timeout for human review (0 = no timeout)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
