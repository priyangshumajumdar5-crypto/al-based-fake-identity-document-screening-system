"""
Unit tests for Module 4 — Primary Portrait Extraction and Face Verification.

Tests multi-factor candidate analysis, ghost watermark rejection, single face policy enforcement,
and age-context policy handling.
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure deepface module exists in sys.modules for patching if not installed
if "deepface" not in sys.modules:
    sys.modules["deepface"] = MagicMock()

from app.face_verification.verifier import (
    VERIFY_STATUS_BORDERLINE,
    VERIFY_STATUS_INVALID_IMAGE,
    VERIFY_STATUS_MATCH,
    VERIFY_STATUS_MISMATCH,
    VERIFY_STATUS_MULTIPLE_FACES,
    VERIFY_STATUS_NO_FACE,
    analyze_and_rank_face_candidates,
    calculate_derived_similarity,
    validate_image_input,
    verify_faces,
)


class TestFaceVerificationEngine(unittest.TestCase):
    """Test suite for Module 4 face verification logic."""

    def test_validate_image_input_invalid(self):
        """Test image input validation for None and missing files."""
        is_valid, msg, _ = validate_image_input(None, "Test Image")
        self.assertFalse(is_valid)
        self.assertIn("Input is None", msg)

        is_valid, msg, _ = validate_image_input("non_existent_file.jpg", "Test Image")
        self.assertFalse(is_valid)
        self.assertIn("File not found", msg)

    def test_calculate_derived_similarity(self):
        """Test linear derived similarity mapping."""
        # Distance 0 -> 100%
        self.assertEqual(calculate_derived_similarity(0.0, 0.68), 100.0)
        # Distance = threshold -> 50%
        self.assertEqual(calculate_derived_similarity(0.68, 0.68), 50.0)
        # Distance 2*threshold -> 0%
        self.assertEqual(calculate_derived_similarity(1.36, 0.68), 0.0)
        # Distance > 2*threshold -> 0%
        self.assertEqual(calculate_derived_similarity(2.0, 0.68), 0.0)

    @patch("app.face_verification.verifier.validate_image_input", return_value=(True, None, "dummy.jpg"))
    @patch("app.face_verification.verifier.analyze_and_rank_face_candidates")
    def test_passport_with_ghost_watermark_successful_verification(self, mock_analyze, mock_val):
        """
        Test Case: Passport with primary portrait + blue ghost watermark.
        System isolates the primary portrait, ignores ghost watermark, and completes face verification.
        """
        mock_analyze.side_effect = [
            (
                "tmp_cropped_primary.jpg", 2, True, "tmp_cropped_primary.jpg", None,
                [
                    {"candidate_index": 1, "status": "SELECTED_PRIMARY", "relative_area_pct": 22.5, "center_x_ratio": 0.25, "composite_score": 0.92},
                    {"candidate_index": 2, "status": "REJECTED_GHOST", "relative_area_pct": 5.1, "center_x_ratio": 0.78, "composite_score": 0.28},
                ]
            ),
            (
                "current_live.jpg", 1, False, None, None,
                [{"candidate_index": 1, "status": "SELECTED_PRIMARY", "relative_area_pct": 35.0, "center_x_ratio": 0.5, "composite_score": 0.88}]
            ),
        ]

        with patch("deepface.DeepFace.verify") as mock_df_verify:
            mock_df_verify.return_value = {
                "verified": True,
                "distance": 0.32,
                "threshold": 0.68,
            }

            res = verify_faces("dummy_passport.jpg", "dummy_live.jpg")

            self.assertEqual(res["verification_status"], VERIFY_STATUS_MATCH)
            self.assertTrue(res["is_match"])
            self.assertEqual(res["raw_distance"], 0.32)
            self.assertEqual(res["passport_face_count"], 2)
            self.assertTrue(res["primary_portrait_extracted"])
            self.assertEqual(len(res["passport_candidate_info"]), 2)
            self.assertEqual(res["passport_candidate_info"][0]["status"], "SELECTED_PRIMARY")

    @patch("app.face_verification.verifier.validate_image_input", return_value=(True, None, "dummy.jpg"))
    @patch("app.face_verification.verifier.analyze_and_rank_face_candidates")
    def test_passport_ambiguous_multiple_comparable_faces(self, mock_analyze, mock_val):
        """
        Test Case: Passport image with two genuinely comparable primary portraits.
        System fails dominance criteria and safely returns MULTIPLE_FACES_DETECTED for manual review.
        """
        mock_analyze.side_effect = [
            (
                "dummy_passport.jpg", 2, False, None,
                "Primary passport portrait could not be isolated reliably. Manual verification required.",
                [
                    {"candidate_index": 1, "status": "AMBIGUOUS_MULTIPLE", "relative_area_pct": 20.0, "center_x_ratio": 0.3, "composite_score": 0.85},
                    {"candidate_index": 2, "status": "AMBIGUOUS_MULTIPLE", "relative_area_pct": 19.5, "center_x_ratio": 0.45, "composite_score": 0.82},
                ]
            ),
            (
                "dummy_live.jpg", 1, False, None, None,
                [{"candidate_index": 1, "status": "SELECTED_PRIMARY", "relative_area_pct": 30.0, "center_x_ratio": 0.5, "composite_score": 0.88}]
            ),
        ]

        res = verify_faces("dummy_passport.jpg", "dummy_live.jpg")

        self.assertEqual(res["verification_status"], VERIFY_STATUS_MULTIPLE_FACES)
        self.assertFalse(res["primary_portrait_extracted"])
        self.assertIn("could not be isolated reliably", res["error_message"])

    @patch("app.face_verification.verifier.validate_image_input", return_value=(True, None, "dummy.jpg"))
    @patch("app.face_verification.verifier.analyze_and_rank_face_candidates")
    def test_current_photo_multiple_genuine_faces_safeguard(self, mock_analyze, mock_val):
        """
        Test Case: Current face photo (Image B) contains multiple genuine faces.
        Strict single-face safeguard prevents multi-person live photo from passing.
        """
        mock_analyze.side_effect = [
            (
                "dummy_passport.jpg", 1, False, None, None,
                [{"candidate_index": 1, "status": "SELECTED_PRIMARY", "relative_area_pct": 25.0, "center_x_ratio": 0.25, "composite_score": 0.90}]
            ),
            (
                "dummy_live.jpg", 2, False, None,
                "Multiple faces (2) detected in Current Face Photo (Image B). Exactly 1 face is required. Manual verification required.",
                [
                    {"candidate_index": 1, "status": "AMBIGUOUS_MULTIPLE", "relative_area_pct": 25.0, "center_x_ratio": 0.4, "composite_score": 0.85},
                    {"candidate_index": 2, "status": "AMBIGUOUS_MULTIPLE", "relative_area_pct": 24.0, "center_x_ratio": 0.7, "composite_score": 0.83},
                ]
            ),
        ]

        res = verify_faces("dummy_passport.jpg", "dummy_live.jpg")

        self.assertEqual(res["verification_status"], VERIFY_STATUS_MULTIPLE_FACES)
        self.assertIn("Multiple faces (2) detected in Current Face Photo", res["error_message"])

    @patch("app.face_verification.verifier.validate_image_input", return_value=(True, None, "dummy.jpg"))
    @patch("app.face_verification.verifier.analyze_and_rank_face_candidates")
    def test_no_face_detected(self, mock_analyze, mock_val):
        """Test Case: No face detected in passport photo."""
        mock_analyze.side_effect = [
            ("dummy_passport.jpg", 0, False, None, "No face detected in image.", []),
            ("dummy_live.jpg", 1, False, None, None, []),
        ]

        res = verify_faces("dummy_passport.jpg", "dummy_live.jpg")

        self.assertEqual(res["verification_status"], VERIFY_STATUS_NO_FACE)
        self.assertIn("No face detected in Passport Photo", res["error_message"])


if __name__ == "__main__":
    unittest.main()

