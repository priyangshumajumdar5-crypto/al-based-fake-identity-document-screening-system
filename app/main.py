"""
FastAPI application entrypoint.

Registers all API routers and initialises the database on startup.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import ocr_routes, screening_routes, validation_routes
from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.database.database import initialize_database

# Initialise logging first
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Application factory
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-Based Indian Passport Screening System — Phase 5 Integration\n\n"
        "**India → Bangladesh Border Checkpoint Demo**\n\n"
        "This is a **PROTOTYPE** using real test passport data.\n"
        "NOT connected to any government database. NOT an official system.\n\n"
        "Implements:\n"
        "- **Module 1**: OCR — EasyOCR field extraction + MRZ parsing\n"
        "- **Module 2**: Document Validation — rules + SQLite prototype database\n"
        "- **Module 3**: Tampering Detection — ELA + Metadata + EfficientNet-B0 (Phase 5 integration)\n\n"
        "Module 4 (Face Verification) is NOT implemented.\n"
        "Deepfake Detection is NOT implemented."
    ),
    version="1.0.0-phase5",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Startup event
# ─────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup() -> None:
    """Initialise the SQLite database on application startup."""
    logger.info("Starting %s (env=%s)", settings.APP_NAME, settings.APP_ENV)
    try:
        initialize_database()
        logger.info("Database ready.")
    except Exception as exc:
        logger.error("Database initialisation failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(ocr_routes.router)
app.include_router(validation_routes.router)
app.include_router(screening_routes.router)   # Phase 5 — unified screening


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": "1.0.0-phase5",
        "status": "running",
        "disclaimer": (
            "PHASE 5 PROTOTYPE — TEST DATA — "
            "NOT AN OFFICIAL GOVERNMENT SYSTEM"
        ),
        "docs": "/docs",
        "screening_endpoint": "POST /api/v1/screening/analyze",
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
