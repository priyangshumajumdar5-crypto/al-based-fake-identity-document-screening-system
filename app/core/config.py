"""
Core application configuration.
Reads values from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised project configuration loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────
    APP_NAME: str = "AI Document Screening System"
    APP_ENV: str = "development"
    DEBUG: bool = True

    # ── API ───────────────────────────────────────────────────
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000

    # ── Database ──────────────────────────────────────────────
    DATABASE_PATH: str = "data/database/passport_screening.db"

    # ── OCR ───────────────────────────────────────────────────
    OCR_LANGUAGES: str = "en"  # comma-separated, e.g. "en,hi"
    OCR_GPU: bool = False
    OCR_MAX_IMAGE_SIZE: int = 4096  # px on longest edge

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    # ── Streamlit ─────────────────────────────────────────────
    STREAMLIT_API_BASE_URL: str = "http://127.0.0.1:8000"

    # ── Face Verification (Module 4) ──────────────────────────
    FACE_MODEL_NAME: str = "ArcFace"
    FACE_DETECTOR_BACKEND: str = "retinaface"
    FACE_DISTANCE_METRIC: str = "cosine"
    FACE_STRICT_MATCH_DISTANCE: float = 0.50     # Distance barrier: d <= 0.50 for MATCH
    FACE_STRICT_MATCH_SIMILARITY: float = 70.0   # Similarity barrier: sim >= 70% for MATCH
    FACE_REJECTION_DISTANCE: float = 0.68        # Distance barrier: d > 0.68 for NO MATCH (MISMATCH)
    FACE_VERIFICATION_THRESHOLD: Optional[float] = None
    FACE_BORDERLINE_MARGIN: float = 0.05
    FACE_AGING_THRESHOLD_YEARS: float = 5.0

    # ── Validation thresholds ─────────────────────────────────
    PASSPORT_NUMBER_MIN_LEN: int = 7
    PASSPORT_NUMBER_MAX_LEN: int = 9
    # Regex pattern for Indian passport numbers  (configurable)
    PASSPORT_NUMBER_PATTERN: str = r"^[A-Z][0-9]{7}$"
    # Required passport fields for Module 2 validation
    REQUIRED_FIELDS: List[str] = [
        "name",
        "passport_number",
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
        "country_code",
        "passport_type",
    ]

    @property
    def ocr_language_list(self) -> List[str]:
        """Return OCR languages as a list."""
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(",") if lang.strip()]

    @property
    def database_path_absolute(self) -> Path:
        """Return absolute path to the SQLite database file."""
        p = Path(self.DATABASE_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# Singleton — import this everywhere
settings = Settings()
