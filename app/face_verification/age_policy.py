"""
Module 4 — Age-Related Context Policy Engine.

Evaluates face verification results alongside passport age / issue-date context
to produce structured findings and manual review recommendations.

Implements all 6 required policy cases:
  - Case A: High similarity (MATCH)
  - Case B: Low similarity + old issue date (MISMATCH_POSSIBLE_AGING)
  - Case C: Low similarity + recent issue date (MISMATCH_RECENT_DOCUMENT)
  - Case D: Low similarity + issue date unavailable (MISMATCH_UNKNOWN_AGE)
  - Case E: Borderline similarity (BORDERLINE_SIMILARITY)
  - Case F: Face verification unavailable (VERIFICATION_UNAVAILABLE)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Policy Category Constants
POLICY_CASE_A_HIGH_SIMILARITY = "MATCH_HIGH_SIMILARITY"
POLICY_CASE_B_POSSIBLE_AGING = "MISMATCH_POSSIBLE_AGING"
POLICY_CASE_C_RECENT_DOCUMENT = "MISMATCH_RECENT_DOCUMENT"
POLICY_CASE_D_UNKNOWN_AGE = "MISMATCH_UNKNOWN_AGE"
POLICY_CASE_E_BORDERLINE = "BORDERLINE_SIMILARITY"
POLICY_CASE_F_UNAVAILABLE = "VERIFICATION_UNAVAILABLE"
POLICY_CASE_DB_NOT_FOUND = "DATABASE_RECORD_NOT_FOUND"
POLICY_CASE_DB_PORTRAIT_UNAVAILABLE = "DATABASE_PORTRAIT_UNAVAILABLE"
POLICY_CASE_FACE_DETECTION_FAILED = "FACE_DETECTION_FAILED"

# Case D Exact Required Text
CASE_D_EXPLANATION = (
    "Face similarity is below the configured threshold. The passport issue date "
    "could not be verified, so the time elapsed since issuance is unknown. "
    "Age-related facial changes may be one possible factor contributing to the "
    "low similarity score, but this cannot be confirmed. Manual verification is required."
)


@dataclass
class AgePolicyConfig:
    """Configurable threshold parameters for Age Context Policy."""
    aging_threshold_years: float = 5.0
    low_similarity_threshold: Optional[float] = None
    borderline_margin: float = 0.05


def apply_age_context_policy(
    verification_result: Dict[str, Any],
    resolved_issue_date: Optional[date],
    elapsed_years: Optional[float],
    issue_date_source: str,
    config: Optional[AgePolicyConfig] = None,
) -> Dict[str, Any]:
    """
    Apply age-related context policy to face verification findings.

    Args:
        verification_result: Result dict from verifier.verify_faces().
        resolved_issue_date: Resolved datetime.date object (or None).
        elapsed_years: Elapsed time in years (or None).
        issue_date_source: 'DATABASE', 'OCR', or 'UNAVAILABLE'.
        config: Optional AgePolicyConfig override.

    Returns:
        Structured policy evaluation dict containing:
          - policy_case: Case identifier (Case A through Case F or DB missing)
          - age_context_status: Human-readable policy tag
          - manual_review_recommended: bool
          - explanation: Detailed human-readable explanation
          - age_warning_flag: bool
          - aging_threshold_years: float
    """
    cfg = config or AgePolicyConfig(
        aging_threshold_years=getattr(settings, "FACE_AGING_THRESHOLD_YEARS", 5.0),
        borderline_margin=getattr(settings, "FACE_BORDERLINE_MARGIN", 0.05),
    )

    verif_status = verification_result.get("verification_status", "UNAVAILABLE")
    raw_dist = verification_result.get("raw_distance")
    thresh = verification_result.get("threshold")
    derived_sim = verification_result.get("derived_similarity_score")
    err_msg = verification_result.get("error_message")

    result: Dict[str, Any] = {
        "policy_case": POLICY_CASE_F_UNAVAILABLE,
        "age_context_status": "VERIFICATION_UNAVAILABLE",
        "manual_review_recommended": True,
        "explanation": "",
        "age_warning_flag": False,
        "aging_threshold_years": cfg.aging_threshold_years,
        "elapsed_years_evaluated": elapsed_years,
    }

    # ── Database Lookup & Specific Error Cases ──────────────────────────────────
    if verif_status == "DATABASE_RECORD_NOT_FOUND":
        result["policy_case"] = POLICY_CASE_DB_NOT_FOUND
        result["age_context_status"] = "DATABASE_RECORD_NOT_FOUND"
        result["explanation"] = err_msg or "No matching passport record found in database. Face verification skipped."
        return result

    if verif_status == "DATABASE_PORTRAIT_UNAVAILABLE":
        result["policy_case"] = POLICY_CASE_DB_PORTRAIT_UNAVAILABLE
        result["age_context_status"] = "DATABASE_PORTRAIT_UNAVAILABLE"
        result["explanation"] = err_msg or "Database record exists but reference portrait is unavailable. Face verification skipped."
        return result

    if verif_status == "FACE_DETECTION_FAILED":
        result["policy_case"] = POLICY_CASE_FACE_DETECTION_FAILED
        result["age_context_status"] = "FACE_DETECTION_FAILED"
        result["explanation"] = err_msg or "Face detection failed on reference portrait or current photo."
        return result

    # ── Case F: Face verification unavailable ─────────────────────────────────
    if verif_status in ("UNAVAILABLE", "NO_FACE_DETECTED", "MULTIPLE_FACES_DETECTED", "INVALID_IMAGE", "NOT_VERIFIED") or raw_dist is None:
        result["policy_case"] = POLICY_CASE_F_UNAVAILABLE
        result["age_context_status"] = "VERIFICATION_UNAVAILABLE"
        result["manual_review_recommended"] = True
        result["explanation"] = (
            f"Face verification could not be completed ({verif_status}). "
            f"Reason: {err_msg or 'Image missing or usable face not detected.'} "
            "Manual verification required."
        )
        logger.info("Age Policy Case F applied: Verification unavailable.")
        return result

    # ── Case A: High similarity (MATCH) ───────────────────────────────────────
    if verif_status == "MATCH":
        result["policy_case"] = POLICY_CASE_A_HIGH_SIMILARITY
        result["age_context_status"] = "HIGH_SIMILARITY_MATCH"
        result["manual_review_recommended"] = False
        result["age_warning_flag"] = False

        exp = (
            f"Face verification PASSED. High similarity confirmed (derived score: {derived_sim}%, "
            f"distance: {raw_dist} <= strict threshold 0.50)."
        )
        if elapsed_years is not None:
            exp += f" Document age: {elapsed_years:.1f} years."
        result["explanation"] = exp
        logger.info("Age Policy Case A applied: High similarity match.")
        return result

    # ── Case E: Inconclusive / Borderline similarity ───────────────────────────
    if verif_status in ("INCONCLUSIVE", "BORDERLINE"):
        result["policy_case"] = POLICY_CASE_E_BORDERLINE
        result["age_context_status"] = "INCONCLUSIVE_BIOMETRIC_MATCH"
        result["manual_review_recommended"] = True
        result["age_warning_flag"] = False
        result["explanation"] = (
            f"Face verification is INCONCLUSIVE (derived score: {derived_sim}%, distance: {raw_dist}, "
            f"strict match threshold: <= 0.50 distance & >= 70.0% similarity). "
            "The biometric match score is insufficient to declare a verified identity. "
            "Manual review by an authorized officer is required."
        )
        logger.info("Age Policy Case E applied: Inconclusive biometric match.")
        return result

    # ── Handle Low Similarity Cases (B, C, D) ──────────────────────────────────
    # If face similarity is low (MISMATCH / distance > threshold):

    # Case D: Issue date unavailable or invalid
    if issue_date_source == "UNAVAILABLE" or resolved_issue_date is None or elapsed_years is None:
        result["policy_case"] = POLICY_CASE_D_UNKNOWN_AGE
        result["age_context_status"] = "LOW_SIMILARITY_UNKNOWN_DOCUMENT_AGE"
        result["manual_review_recommended"] = True
        result["age_warning_flag"] = True
        result["explanation"] = CASE_D_EXPLANATION
        logger.info("Age Policy Case D applied: Low similarity with unknown document age.")
        return result

    # Case B: Old issue date (elapsed_years >= aging_threshold_years)
    if elapsed_years >= cfg.aging_threshold_years:
        result["policy_case"] = POLICY_CASE_B_POSSIBLE_AGING
        result["age_context_status"] = "LOW_SIMILARITY_OLD_DOCUMENT_POSSIBLE_AGING"
        result["manual_review_recommended"] = True
        result["age_warning_flag"] = True
        result["explanation"] = (
            f"Face similarity is below the threshold (derived score: {derived_sim}%, distance: {raw_dist} > threshold: {thresh}). "
            f"The passport was issued {elapsed_years:.1f} years ago (>= {cfg.aging_threshold_years:.1f} year threshold). "
            "Natural facial changes over time may be one possible factor contributing to the similarity difference. "
            "This cannot be confirmed automatically. Manual verification is required."
        )
        logger.info("Age Policy Case B applied: Low similarity with old document age (%.1f yrs).", elapsed_years)
        return result

    # Case C: Recent issue date (elapsed_years < aging_threshold_years)
    result["policy_case"] = POLICY_CASE_C_RECENT_DOCUMENT
    result["age_context_status"] = "LOW_SIMILARITY_RECENT_DOCUMENT"
    result["manual_review_recommended"] = True
    result["age_warning_flag"] = False
    result["explanation"] = (
        f"Face similarity is below the threshold (derived score: {derived_sim}%, distance: {raw_dist} > threshold: {thresh}). "
        f"The passport is relatively recent (issued {elapsed_years:.1f} years ago, < {cfg.aging_threshold_years:.1f} year threshold). "
        "Age-related facial changes are unlikely to be the primary cause of mismatch. Manual verification is required."
    )
    logger.info("Age Policy Case C applied: Low similarity with recent document age (%.1f yrs).", elapsed_years)
    return result
