"""
Advisory Risk Assessment — Module 3, Phase 5.

Produces a transparent, configurable risk level for officer prioritization.

PURPOSE: Help officers decide which documents need closer inspection.
         NOT a fraud determination. NOT a replace for human judgment.

Signal sources (used only when available):
  1. Module 2 database validation findings  (confirmed mismatches)
  2. ELA metrics                            (supplementary forensic signal)
  3. Metadata observations                  (supplementary forensic signal)
  4. EfficientNet-B0 model score            (supplementary — NOT a probability)

Weights and thresholds are configurable via RiskConfig.

DISCLAIMER: Experimental prototype screening score only.
            A human officer retains full responsibility for the final decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Risk levels ───────────────────────────────────────────────────────────────
RISK_LOW              = "LOW"
RISK_MEDIUM           = "MEDIUM"
RISK_HIGH             = "HIGH"
RISK_INSUFFICIENT     = "INSUFFICIENT_DATA"

_LEVEL_ORDER = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Configurable weights and thresholds for the risk assessment."""

    # ── Database validation signals ───────────────────────────────────────────
    # Points added per confirmed database field mismatch
    pts_per_db_mismatch: float = 30.0
    # Points for a database record not found
    pts_db_not_found: float = 20.0
    # Points for database record in EXPIRED / INVALID / FLAGGED status
    pts_db_bad_status: float = 40.0

    # ── ELA signals ───────────────────────────────────────────────────────────
    # Mean ELA threshold above which a signal is raised
    ela_mean_threshold: float = 8.0
    pts_ela_elevated: float = 10.0

    # ── Metadata signals ─────────────────────────────────────────────────────
    # Points per metadata observation (e.g. editing software detected, GPS present)
    pts_per_metadata_obs: float = 5.0
    # Maximum metadata contribution (caps observation stacking)
    pts_metadata_cap: float = 15.0

    # ── EfficientNet-B0 model signal ─────────────────────────────────────────
    # Only applied when inference succeeded
    # Tampered score threshold above which signal is raised
    model_tampered_threshold: float = 0.60
    pts_model_tampered: float = 30.0

    # ── Module 4 Face Verification signals ────────────────────────────────────
    pts_face_mismatch: float = 35.0
    pts_face_borderline: float = 15.0
    pts_face_unknown_age: float = 25.0
    pts_face_possible_aging: float = 20.0

    # ── Risk thresholds ───────────────────────────────────────────────────────
    threshold_high:   float = 50.0
    threshold_medium: float = 20.0


# ── Result structure ──────────────────────────────────────────────────────────

@dataclass
class RiskAssessmentResult:
    risk_level:              str
    risk_score:              float
    db_validation_concerns:  List[str] = field(default_factory=list)
    tampering_indicators:    List[str] = field(default_factory=list)
    unavailable_signals:     List[str] = field(default_factory=list)
    risk_reasons:            List[str] = field(default_factory=list)
    signals_used:            List[str] = field(default_factory=list)
    signals_missing:         List[str] = field(default_factory=list)
    manual_review_recommended: bool = True
    disclaimer: str = (
        "ADVISORY PROTOTYPE SCREENING SCORE ONLY. "
        "Not a statistically calibrated fraud probability. "
        "A human officer retains full responsibility for the final decision."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level":                  self.risk_level,
            "risk_score":                  round(self.risk_score, 2),
            "db_validation_concerns":      self.db_validation_concerns,
            "tampering_indicators":        self.tampering_indicators,
            "unavailable_signals":         self.unavailable_signals,
            "risk_reasons":                self.risk_reasons,
            "signals_used":                self.signals_used,
            "signals_missing":             self.signals_missing,
            "manual_review_recommended":   self.manual_review_recommended,
            "disclaimer":                  self.disclaimer,
        }


# ── Main function ─────────────────────────────────────────────────────────────

def assess_risk(
    validation_result:        Optional[Dict[str, Any]] = None,
    ela_result:               Optional[Dict[str, Any]] = None,
    metadata_result:          Optional[Dict[str, Any]] = None,
    inference_result:         Optional[Dict[str, Any]] = None,
    face_verification_result: Optional[Dict[str, Any]] = None,
    config:                   Optional[RiskConfig]     = None,
) -> RiskAssessmentResult:
    """
    Compute an advisory risk assessment from available module outputs.

    Distinguishes Database Validation Concerns (Module 2) from Image Tampering / Biometric
    Indicators (Module 3 & Module 4). Missing EXIF metadata does NOT increase tampering score.

    Args:
        validation_result:        Module 2 validate_passport() result dict.
        ela_result:               Module 3 Phase 1 run_ela() result dict.
        metadata_result:          Module 3 Phase 1 run_metadata_analysis() result dict.
        inference_result:         Module 3 Phase 3 predict() result dict.
        face_verification_result: Module 4 run_face_verification_pipeline() result dict.
        config:                   Optional RiskConfig (uses defaults if None).

    Returns:
        RiskAssessmentResult with separated DB concerns, tampering/biometric indicators, and risk score.
    """
    cfg = config or RiskConfig()

    score:                  float     = 0.0
    db_concerns:            List[str] = []
    tampering_indicators:   List[str] = []
    unavailable_signals:    List[str] = []
    reasons:                List[str] = []
    signals_used:           List[str] = []
    signals_missing:        List[str] = []

    # ── 1. Module 2: Database Validation (Database Inconsistency Signal) ─────
    if validation_result is not None:
        overall_status = validation_result.get("overall_status", "")
        db_match       = validation_result.get("db_match", False)
        comparisons    = validation_result.get("field_comparisons", [])

        if not db_match or overall_status == "NOT_FOUND":
            pts = cfg.pts_db_not_found
            score += pts
            msg = f"+{pts:.0f}pts: Passport number not found in prototype database (Database Concern)."
            db_concerns.append(msg)
            reasons.append(msg)
        else:
            # Count confirmed mismatches
            mismatches = [
                c.get("field_name", "field")
                for c in comparisons
                if isinstance(c, dict) and c.get("status") == "MISMATCH"
            ]
            if mismatches:
                pts = cfg.pts_per_db_mismatch * len(mismatches)
                score += pts
                msg = (
                    f"+{pts:.0f}pts: {len(mismatches)} confirmed database field mismatch(es): "
                    f"{', '.join(mismatches)} (Database Concern)."
                )
                db_concerns.append(msg)
                reasons.append(msg)

        signals_used.append("database_validation")
    else:
        signals_missing.append("database_validation")
        unavailable_signals.append("Database Validation (Module 2)")

    # ── 2. Module 3a: ELA Analysis (Supplementary Forensic Signal) ───────────
    if ela_result is not None and ela_result.get("success"):
        mean_ela = ela_result.get("mean_ela")
        if mean_ela is not None and mean_ela > cfg.ela_mean_threshold:
            pts = cfg.pts_ela_elevated
            score += pts
            msg = (
                f"+{pts:.0f}pts: Elevated mean ELA ({mean_ela:.2f} > threshold {cfg.ela_mean_threshold}). "
                "Informational ELA signal only — does NOT prove image tampering."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
        signals_used.append("ela_analysis")
    else:
        signals_missing.append("ela_analysis")
        unavailable_signals.append("ELA Analysis (Module 3a)")

    # ── 3. Module 3b: Metadata Analysis (Informational Signal) ────────────────
    if metadata_result is not None and metadata_result.get("success"):
        observations = metadata_result.get("observations", [])
        # Only add risk points for explicit photo editing software tags!
        # Missing EXIF data alone MUST NOT add risk points or be labeled as tampering.
        editing_obs = [
            ob for ob in observations
            if "Software tag" in ob or "photo-editing" in ob
        ]
        if editing_obs:
            pts = min(cfg.pts_per_metadata_obs * len(editing_obs), cfg.pts_metadata_cap)
            score += pts
            msg = f"+{pts:.0f}pts: Photo-editing software tag detected in metadata."
            tampering_indicators.append(msg)
            reasons.append(msg)

        signals_used.append("metadata_analysis")
    else:
        signals_missing.append("metadata_analysis")
        unavailable_signals.append("Metadata Analysis (Module 3b)")

    # ── 4. Module 3c: EfficientNet-B0 Model Inference ───────────────────────
    if inference_result is not None and inference_result.get("success"):
        score_tampered = inference_result.get("score_tampered") or 0.0
        band = inference_result.get("band") or (
            "TAMPERED" if score_tampered >= cfg.model_tampered_threshold else "GENUINE"
        )
        if band == "TAMPERED":
            pts = cfg.pts_model_tampered
            score += pts
            msg = (
                f"+{pts:.0f}pts: Classifier flags image regions as TAMPERED "
                f"(score {score_tampered:.3f}). "
                "Uncalibrated model output — supplementary signal only."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
        elif band == "UNCERTAIN":
            pts = 15.0
            score += pts
            msg = (
                f"+{pts:.0f}pts: Classifier output is UNCERTAIN (score {score_tampered:.3f}). "
                "Model could not establish a genuine classification — manual inspection required."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
        signals_used.append("model_inference")
    else:
        signals_missing.append("efficientnet_b0_inference")
        msg_unavail = "EfficientNet-B0 Classifier (Module 3c — Checkpoint Unavailable or Failed)"
        unavailable_signals.append(msg_unavail)

    # ── 5. Module 4: Face Verification Signal ────────────────────────────────
    if face_verification_result is not None:
        fv_status = face_verification_result.get("face_verification_status") or face_verification_result.get("verification_status")
        policy_case = face_verification_result.get("policy_case")
        raw_dist = face_verification_result.get("raw_distance")
        thresh = face_verification_result.get("model_threshold")
        derived_sim = face_verification_result.get("derived_similarity_score")

        if policy_case == "MATCH_HIGH_SIMILARITY":
            signals_used.append("face_verification")
        elif policy_case == "MISMATCH_RECENT_DOCUMENT":
            pts = cfg.pts_face_mismatch
            score += pts
            msg = (
                f"+{pts:.0f}pts: Face verification MISMATCH on recent document "
                f"(derived score {derived_sim}%, distance {raw_dist} > threshold {thresh}). "
                "Biometric mismatch indicator."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
            signals_used.append("face_verification")
        elif policy_case == "MISMATCH_POSSIBLE_AGING":
            pts = cfg.pts_face_possible_aging
            score += pts
            msg = (
                f"+{pts:.0f}pts: Low face similarity on old document. "
                "Natural facial aging over time may be a contributing factor. Manual inspection required."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
            signals_used.append("face_verification")
        elif policy_case == "MISMATCH_UNKNOWN_AGE":
            pts = cfg.pts_face_unknown_age
            score += pts
            msg = (
                f"+{pts:.0f}pts: Low face similarity and passport issue date unverified. "
                "Age context unknown — manual verification required."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
            signals_used.append("face_verification")
        elif policy_case == "BORDERLINE_SIMILARITY":
            pts = cfg.pts_face_borderline
            score += pts
            msg = (
                f"+{pts:.0f}pts: Face verification is BORDERLINE (derived score {derived_sim}%). "
                "Inconclusive biometric verification — officer review required."
            )
            tampering_indicators.append(msg)
            reasons.append(msg)
            signals_used.append("face_verification")
        elif fv_status == "ok" or fv_status == "MATCH":
            signals_used.append("face_verification")
        else:
            signals_missing.append("face_verification")
            unavailable_signals.append("Face Verification (Module 4 — Face Not Detected or Unreadable)")
    else:
        signals_missing.append("face_verification")
        unavailable_signals.append("Face Verification (Module 4 — Current Photo Not Provided)")

    # ── Insufficient data guard ───────────────────────────────────────────────
    if not signals_used:
        return RiskAssessmentResult(
            risk_level=RISK_INSUFFICIENT,
            risk_score=0.0,
            db_validation_concerns=[],
            tampering_indicators=[],
            unavailable_signals=unavailable_signals,
            risk_reasons=["No module outputs available — cannot compute risk score."],
            signals_used=[],
            signals_missing=signals_missing,
            manual_review_recommended=True,
        )

    # ── Determine risk level ──────────────────────────────────────────────────
    if score >= cfg.threshold_high:
        level = RISK_HIGH
    elif score >= cfg.threshold_medium:
        level = RISK_MEDIUM
    else:
        level = RISK_LOW

    # Manual review recommended if database mismatches, tampering indicators, or high risk score
    manual_review = bool(db_concerns or tampering_indicators or level in (RISK_HIGH, RISK_MEDIUM, RISK_INSUFFICIENT))

    return RiskAssessmentResult(
        risk_level=level,
        risk_score=score,
        db_validation_concerns=db_concerns,
        tampering_indicators=tampering_indicators,
        unavailable_signals=unavailable_signals,
        risk_reasons=reasons if reasons else ["No risk signals detected."],
        signals_used=signals_used,
        signals_missing=signals_missing,
        manual_review_recommended=manual_review,
    )
