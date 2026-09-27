"""
Module 4 — Face Detection, Face Verification, and Age-Related Context Handling.

This package provides components for:
  - Database lookup for passport records by passport number
  - Issue date resolution and elapsed time calculation
  - Pairwise face detection and verification using DeepFace with ArcFace
  - Age-context policy evaluation (6 policy cases)
  - Unified pipeline service wrapper
"""

from app.face_verification.service import run_face_verification_pipeline

__all__ = ["run_face_verification_pipeline"]
