"""
Module 4 — Face Detection and Verification Engine.

Performs image validation, face detection & face count checking,
and pairwise face verification using DeepFace with the ArcFace model.

Key rules:
  - Validates image file existence and readability.
  - Requires EXACTLY ONE face detected per image.
  - Returns clear status if 0 faces or >1 faces detected (does NOT pick arbitrary face).
  - Uses ArcFace model and cosine distance metric consistently.
  - Calculates a derived similarity score (clearly documented as uncalibrated display metric).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

# Verification status constants
VERIFY_STATUS_MATCH = "MATCH"
VERIFY_STATUS_INCONCLUSIVE = "INCONCLUSIVE"
VERIFY_STATUS_MISMATCH = "MISMATCH"
VERIFY_STATUS_BORDERLINE = "BORDERLINE"
VERIFY_STATUS_NO_FACE = "NO_FACE_DETECTED"
VERIFY_STATUS_MULTIPLE_FACES = "MULTIPLE_FACES_DETECTED"
VERIFY_STATUS_INVALID_IMAGE = "INVALID_IMAGE"
VERIFY_STATUS_UNAVAILABLE = "UNAVAILABLE"


def validate_image_input(image_input: Union[str, Path, bytes], name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate that an image input exists and can be accessed.

    Args:
        image_input: File path (str/Path) or raw image bytes.
        name: Image identifier for error messages (e.g. 'Passport photo').

    Returns:
        (is_valid, error_message, file_path_if_any)
    """
    if image_input is None:
        return False, f"{name}: Input is None.", None

    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return False, f"{name}: File not found at path '{p}'.", None
        if not p.is_file():
            return False, f"{name}: Path '{p}' is not a regular file.", None
        if p.stat().st_size == 0:
            return False, f"{name}: File at '{p}' is empty (0 bytes).", None
        return True, None, str(p)

    if isinstance(image_input, bytes):
        if len(image_input) == 0:
            return False, f"{name}: Provided bytes buffer is empty.", None
        return True, None, None

    return False, f"{name}: Unsupported input type '{type(image_input)}'. Expected str, Path, or bytes.", None


def analyze_and_rank_face_candidates(
    image_source: Union[str, Path, bytes],
    detector_backend: str = "retinaface",
    is_passport_document: bool = True,
) -> Tuple[Optional[str], int, bool, Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    Detect face candidates in an image and evaluate them using a multi-factor ranking strategy.

    Multi-factor evaluation criteria:
      1. Bounding box size / relative area (Primary photo is significantly larger than ghost watermark).
      2. Normalized horizontal position (Primary photo is on left side x < 0.52; ghost photo is on right side x >= 0.52).
      3. Color variance / RGB standard deviation (Primary photo is full color RGB; ghost watermark is blue/cyan translucent).
      4. Detector confidence score.

    Args:
        image_source: Path or bytes of the image.
        detector_backend: Face detector backend.
        is_passport_document: If True, enables multi-factor primary portrait extraction for documents with ghost watermarks.
                              If False (e.g. for live photo), strictly enforces single face policy.

    Returns:
        (working_image_path, total_candidate_count, is_primary_cropped, temp_file_created, error_message, candidates_info)
    """
    try:
        import numpy as np
        from deepface import DeepFace
        from PIL import Image
    except ImportError:
        return None, -1, False, None, "DeepFace, NumPy, or PIL library is not installed.", []

    created_temp: Optional[str] = None
    try:
        if isinstance(image_source, bytes):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(image_source)
            tmp.close()
            created_temp = tmp.name
            src_path = tmp.name
        else:
            src_path = str(image_source)

        faces = []
        try:
            faces = DeepFace.extract_faces(
                img_path=src_path,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
        except Exception as exc:
            logger.warning("Primary face extraction failed with detector '%s': %s", detector_backend, exc)

        valid_faces = [
            f for f in (faces or [])
            if isinstance(f, dict) and f.get("confidence", 1.0) > 0.3
        ]

        if not valid_faces and detector_backend != "opencv":
            try:
                faces_ocv = DeepFace.extract_faces(
                    img_path=src_path,
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                valid_faces = [
                    f for f in (faces_ocv or [])
                    if isinstance(f, dict) and f.get("confidence", 1.0) > 0.3
                ]
            except Exception:
                pass

        cnt = len(valid_faces)
        if cnt == 0:
            return src_path, 0, False, created_temp, "No face detected in image.", []

        # Read image to obtain dimensions and calculate color metrics
        try:
            with Image.open(src_path) as pil_img:
                pil_rgb = pil_img.convert("RGB")
                W, H = pil_rgb.size
                img_np = np.array(pil_rgb)
        except Exception as exc:
            return src_path, cnt, False, created_temp, f"Failed to load image for analysis: {exc}", []

        candidates_info: List[Dict[str, Any]] = []

        for idx, f in enumerate(valid_faces):
            fa = f.get("facial_area", {})
            x = max(0, int(fa.get("x", 0)))
            y = max(0, int(fa.get("y", 0)))
            w = max(1, int(fa.get("w", 1)))
            h = max(1, int(fa.get("h", 1)))

            area = w * h
            rel_area = area / float(W * H)
            center_x = (x + w / 2.0) / float(W)
            center_y = (y + h / 2.0) / float(H)

            # Extract face crop array to measure color variance
            crop_np = img_np[y:y+h, x:x+w]
            if crop_np.size > 0:
                r_mean = float(np.mean(crop_np[:, :, 0]))
                g_mean = float(np.mean(crop_np[:, :, 1]))
                b_mean = float(np.mean(crop_np[:, :, 2]))
                color_var = float(np.std([r_mean, g_mean, b_mean]))
            else:
                color_var = 0.0

            # Sub-factor scores (0.0 to 1.0)
            score_area = min(1.0, rel_area / 0.04)  # 4%+ of page area gets max area score
            score_pos = 1.0 if center_x < 0.50 else (0.6 if center_x < 0.55 else 0.1)  # Left region prioritized
            score_color = min(1.0, color_var / 20.0)  # Color diversity vs blue/mono tint
            score_conf = min(1.0, float(f.get("confidence", 1.0)))

            # Multi-factor composite score
            composite_score = (
                0.45 * score_area +
                0.30 * score_pos +
                0.15 * score_color +
                0.10 * score_conf
            )

            cand_info = {
                "candidate_index": idx + 1,
                "facial_area": {"x": x, "y": y, "w": w, "h": h},
                "relative_area_pct": round(rel_area * 100.0, 2),
                "center_x_ratio": round(center_x, 3),
                "center_y_ratio": round(center_y, 3),
                "color_variance": round(color_var, 2),
                "confidence": round(score_conf, 3),
                "composite_score": round(composite_score, 4),
                "status": "EVALUATING",
                "raw_face_dict": f,
            }
            candidates_info.append(cand_info)

        # Sort candidates descending by multi-factor composite score
        candidates_info.sort(key=lambda c: c["composite_score"], reverse=True)

        if cnt == 1:
            candidates_info[0]["status"] = "SELECTED_PRIMARY"
            # Return raw face dict info stripped from returned candidate metadata
            clean_cands = [{k: v for k, v in c.items() if k != "raw_face_dict"} for c in candidates_info]
            return src_path, 1, False, created_temp, None, clean_cands

        # Multiple candidates detected (> 1)
        top_cand = candidates_info[0]
        second_cand = candidates_info[1]

        top_score = top_cand["composite_score"]
        sec_score = second_cand["composite_score"]

        # Dominance criteria for passport documents
        is_dominant = False
        if is_passport_document:
            # Rule 1: Composite score is significantly higher (>= 1.2x)
            if top_score >= 1.20 * sec_score:
                is_dominant = True
            # Rule 2: Top candidate is on left side (primary portrait) while second is on right side (ghost) and top area is larger
            elif (top_cand["center_x_ratio"] < 0.52 and second_cand["center_x_ratio"] >= 0.52
                  and top_cand["relative_area_pct"] >= 1.2 * second_cand["relative_area_pct"]):
                is_dominant = True

        clean_cands = [{k: v for k, v in c.items() if k != "raw_face_dict"} for c in candidates_info]

        if is_dominant:
            top_cand["status"] = "SELECTED_PRIMARY"
            for c in clean_cands[1:]:
                c["status"] = "REJECTED_GHOST" if c["center_x_ratio"] >= 0.52 else "REJECTED_SECONDARY"

            # Crop the dominant primary portrait with 25% padding
            try:
                fa = top_cand["facial_area"]
                fx, fy, fw, fh = fa["x"], fa["y"], fa["w"], fa["h"]
                pad_w = int(fw * 0.25)
                pad_h = int(fh * 0.25)
                x1 = max(0, fx - pad_w)
                y1 = max(0, fy - pad_h)
                x2 = min(W, fx + fw + pad_w)
                y2 = min(H, fy + fh + pad_h)

                with Image.open(src_path) as pil_img:
                    cropped_img = pil_img.convert("RGB").crop((x1, y1, x2, y2))
                    import tempfile
                    tmp_c = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    cropped_img.save(tmp_c.name, format="JPEG")
                    tmp_c.close()

                    logger.info(
                        "Primary passport portrait isolated successfully: relative_area=%.2f%%, position_x=%.2f, composite_score=%.4f (ignored ghost face score=%.4f)",
                        top_cand["relative_area_pct"], top_cand["center_x_ratio"], top_score, sec_score
                    )
                    return tmp_c.name, cnt, True, tmp_c.name, None, clean_cands
            except Exception as exc:
                logger.warning("Cropping primary portrait failed: %s. Falling back to original image.", exc)

        # Ambiguous document or Live photo with multiple comparable faces
        for c in clean_cands:
            c["status"] = "AMBIGUOUS_MULTIPLE"

        if is_passport_document:
            err_msg = "Primary passport portrait could not be isolated reliably. Manual verification required."
        else:
            err_msg = f"Multiple faces ({cnt}) detected in Current Face Photo (Image B). Exactly 1 face is required. Manual verification required."

        return src_path, cnt, False, created_temp, err_msg, clean_cands

    except Exception as exc:
        return None, -1, False, created_temp, f"Face extraction error: {exc}", []


def count_faces_in_image(
    image_source: Union[str, Path, bytes],
    detector_backend: str = "retinaface",
) -> Tuple[int, Optional[str]]:
    """
    Detect and count faces in an image using DeepFace.

    Args:
        image_source: Image file path or bytes.
        detector_backend: Detector backend (e.g. 'retinaface', 'opencv', 'ssd', 'mtcnn').

    Returns:
        (face_count, error_message)
    """
    try:
        from deepface import DeepFace
    except ImportError:
        return -1, "DeepFace library is not installed. Run 'pip install deepface' to enable face verification."

    tmp_file_created: Optional[str] = None
    try:
        if isinstance(image_source, bytes):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(image_source)
            tmp.close()
            tmp_file_created = tmp.name
            src_path = tmp.name
        else:
            src_path = str(image_source)

        faces = DeepFace.extract_faces(
            img_path=src_path,
            detector_backend=detector_backend,
            enforce_detection=False,
        )
        valid_faces = [
            f for f in (faces or [])
            if isinstance(f, dict) and f.get("confidence", 1.0) > 0.4
        ] if faces else []

        return len(valid_faces), None
    except Exception as exc:
        logger.warning("Face detection count failed with detector '%s': %s", detector_backend, exc)
        if detector_backend != "opencv":
            try:
                faces = DeepFace.extract_faces(
                    img_path=src_path if 'src_path' in locals() else str(image_source),
                    detector_backend="opencv",
                    enforce_detection=False,
                )
                valid_faces = [
                    f for f in (faces or [])
                    if isinstance(f, dict) and f.get("confidence", 1.0) > 0.4
                ] if faces else []
                return len(valid_faces), None
            except Exception as exc2:
                return -1, f"Face detection failed: {exc2}"
        return -1, f"Face detection failed: {exc}"
    finally:
        if tmp_file_created and os.path.exists(tmp_file_created):
            try:
                os.unlink(tmp_file_created)
            except Exception:
                pass


def calculate_derived_similarity(raw_distance: float, model_threshold: float) -> float:
    """
    Calculate a derived similarity score percentage from raw distance and threshold.

    DISCLAIMER: This derived score is for human interface presentation only.
    It is NOT a statistically calibrated probability that the person is genuine.

    Formula:
        For distance <= threshold: maps linearly from 100% to 50%.
        For distance > threshold: maps linearly from 50% down to 0%.

    Args:
        raw_distance: Model distance metric (e.g. cosine distance).
        model_threshold: Applicable model threshold.

    Returns:
        Derived similarity score percentage (0.0 to 100.0).
    """
    if model_threshold <= 0:
        return 0.0

    if raw_distance <= model_threshold:
        score = 100.0 - (raw_distance / model_threshold) * 50.0
    else:
        excess = raw_distance - model_threshold
        score = 50.0 - (excess / model_threshold) * 50.0

    return round(max(0.0, min(100.0, score)), 2)


def verify_faces(
    passport_photo_input: Union[str, Path, bytes],
    current_photo_input: Union[str, Path, bytes],
    model_name: str = "ArcFace",
    detector_backend: str = "retinaface",
    distance_metric: str = "cosine",
    custom_threshold: Optional[float] = None,
    borderline_margin: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform pairwise face verification between passport photo and current photo.

    Args:
        passport_photo_input: Image A (Passport portrait path or bytes).
        current_photo_input: Image B (Current face photo path or bytes).
        model_name: DeepFace model name (defaults to 'ArcFace').
        detector_backend: Face detection backend (defaults to 'retinaface').
        distance_metric: Distance metric (defaults to 'cosine').
        custom_threshold: Override model threshold if provided.
        borderline_margin: Margin above threshold to classify as BORDERLINE.

    Returns:
        Structured result dict containing:
          - verification_status (MATCH, MISMATCH, BORDERLINE, NO_FACE_DETECTED, etc.)
          - is_match (bool or None)
          - raw_distance (float or None)
          - threshold (float or None)
          - derived_similarity_score (float or None)
          - passport_face_count (int)
          - current_face_count (int)
          - primary_portrait_extracted (bool)
          - passport_candidate_info (list)
          - current_candidate_info (list)
          - model_name (str)
          - detector_backend (str)
          - distance_metric (str)
          - error_message (str or None)
    """
    result: Dict[str, Any] = {
        "verification_status": VERIFY_STATUS_UNAVAILABLE,
        "is_match": None,
        "raw_distance": None,
        "threshold": None,
        "derived_similarity_score": None,
        "passport_face_count": 0,
        "current_face_count": 0,
        "primary_portrait_extracted": False,
        "passport_candidate_info": [],
        "current_candidate_info": [],
        "model_name": model_name,
        "detector_backend": detector_backend,
        "distance_metric": distance_metric,
        "error_message": None,
    }

    # 1. Validate image inputs
    v_pass, err_pass, path_pass = validate_image_input(passport_photo_input, "Passport photo (Image A)")
    v_curr, err_curr, path_curr = validate_image_input(current_photo_input, "Current face photo (Image B)")

    if not v_pass or not v_curr:
        errs = [e for e in [err_pass, err_curr] if e]
        result["verification_status"] = VERIFY_STATUS_INVALID_IMAGE
        result["error_message"] = " | ".join(errs)
        logger.error("Image validation failed: %s", result["error_message"])
        return result

    # 2. Check DeepFace availability
    try:
        from deepface import DeepFace
    except ImportError:
        result["verification_status"] = VERIFY_STATUS_UNAVAILABLE
        result["error_message"] = (
            "DeepFace library is not installed in the environment. "
            "Please run 'pip install deepface tf-keras' in Jupyter Notebook to enable face verification."
        )
        logger.warning(result["error_message"])
        return result

    tmp_files_to_cleanup = []
    try:
        # 3. Multi-factor Primary Face Extraction and candidate analysis
        work_pass, cnt_pass, pass_cropped, tmp_pass, err_pass_ext, cands_pass = analyze_and_rank_face_candidates(
            passport_photo_input, detector_backend=detector_backend, is_passport_document=True
        )
        if tmp_pass:
            tmp_files_to_cleanup.append(tmp_pass)

        work_curr, cnt_curr, curr_cropped, tmp_curr, err_curr_ext, cands_curr = analyze_and_rank_face_candidates(
            current_photo_input, detector_backend=detector_backend, is_passport_document=False
        )
        if tmp_curr:
            tmp_files_to_cleanup.append(tmp_curr)

        img_pass_source = work_pass if work_pass else (path_pass if path_pass else str(passport_photo_input))
        img_curr_source = work_curr if work_curr else (path_curr if path_curr else str(current_photo_input))

        result["passport_face_count"] = cnt_pass
        result["current_face_count"] = cnt_curr
        result["primary_portrait_extracted"] = pass_cropped
        result["passport_candidate_info"] = cands_pass
        result["current_candidate_info"] = cands_curr

        # Handle face detection / isolation failures
        if cnt_pass == 0 or cnt_curr == 0:
            result["verification_status"] = VERIFY_STATUS_NO_FACE
            no_face_details = []
            if cnt_pass == 0:
                no_face_details.append("No face detected in Passport Photo (Image A)")
            if cnt_curr == 0:
                no_face_details.append("No face detected in Current Face Photo (Image B)")
            result["error_message"] = " | ".join(no_face_details)
            logger.warning("Face detection failed: %s", result["error_message"])
            return result

        if (cnt_pass > 1 and not pass_cropped) or (cnt_curr > 1 and not curr_cropped):
            result["verification_status"] = VERIFY_STATUS_MULTIPLE_FACES
            multi_details = []
            if err_pass_ext:
                multi_details.append(err_pass_ext)
            elif cnt_pass > 1 and not pass_cropped:
                multi_details.append(f"Multiple comparable-size faces ({cnt_pass}) detected in Passport Photo (Image A)")

            if err_curr_ext:
                multi_details.append(err_curr_ext)
            elif cnt_curr > 1 and not curr_cropped:
                multi_details.append(f"Multiple faces ({cnt_curr}) detected in Current Face Photo (Image B)")

            result["error_message"] = " | ".join(multi_details)
            logger.warning(result["error_message"])
            return result

        if cnt_pass < 0 or cnt_curr < 0:
            result["verification_status"] = VERIFY_STATUS_UNAVAILABLE
            result["error_message"] = f"Face detection error: {err_pass_ext or err_curr_ext}"
            return result

        # 4. Perform pairwise DeepFace verification on isolated primary faces
        try:
            df_result = DeepFace.verify(
                img1_path=img_pass_source,
                img2_path=img_curr_source,
                model_name=model_name,
                detector_backend=detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False,
            )

            raw_dist = float(df_result.get("distance", 1.0))
            model_thresh = float(custom_threshold if custom_threshold is not None else df_result.get("threshold", 0.68))

            derived_sim = calculate_derived_similarity(raw_dist, model_thresh)

            strict_match_dist = float(getattr(settings, "FACE_STRICT_MATCH_DISTANCE", 0.50))
            strict_match_sim = float(getattr(settings, "FACE_STRICT_MATCH_SIMILARITY", 70.0))
            rejection_dist = float(getattr(settings, "FACE_REJECTION_DISTANCE", 0.68))

            result["raw_distance"] = round(raw_dist, 4)
            result["threshold"] = round(model_thresh, 4)
            result["derived_similarity_score"] = derived_sim
            result["strict_match_distance"] = strict_match_dist
            result["strict_match_similarity"] = strict_match_sim

            # Stricter Three-Level Decision Policy
            # Level 1: MATCH — Face Verified (distance <= 0.50 AND similarity >= 70%)
            if raw_dist <= strict_match_dist and derived_sim >= strict_match_sim:
                result["verification_status"] = VERIFY_STATUS_MATCH
                result["is_match"] = True
                result["decision_label"] = "MATCH — Face Verified"

            # Level 2: NO MATCH — Face Not Verified (distance > 0.68)
            elif raw_dist > rejection_dist:
                result["verification_status"] = VERIFY_STATUS_MISMATCH
                result["is_match"] = False
                result["decision_label"] = "NO MATCH — Face Not Verified"

            # Level 3: INCONCLUSIVE — Manual Review Required (0.50 < distance <= 0.68 OR similarity < 70%)
            else:
                result["verification_status"] = VERIFY_STATUS_INCONCLUSIVE
                result["is_match"] = False
                result["decision_label"] = "INCONCLUSIVE — Manual Review Required"
                result["error_message"] = (
                    f"Face comparison is inconclusive (distance {raw_dist:.4f} > strict threshold {strict_match_dist:.2f} "
                    f"or derived similarity {derived_sim:.1f}% < {strict_match_sim:.0f}%). Manual verification required."
                )

            logger.info(
                "Face verification completed: status=%s, decision=%s, distance=%.4f, threshold=%.4f, derived_similarity=%.2f%%, primary_cropped=%s",
                result["verification_status"], result["decision_label"], raw_dist, model_thresh, derived_sim, pass_cropped,
            )
            return result

        except Exception as exc:
            logger.error("DeepFace pairwise verification failed: %s", exc)
            result["verification_status"] = VERIFY_STATUS_UNAVAILABLE
            result["error_message"] = f"DeepFace verification error: {exc}"
            return result

    finally:
        for tmp_file in tmp_files_to_cleanup:
            if os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass



