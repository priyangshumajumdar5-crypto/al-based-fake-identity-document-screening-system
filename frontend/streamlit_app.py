"""
AI Passport Screening System — Streamlit Dashboard
Module 1 (OCR) + Module 2 (Database Validation)

India → Bangladesh Border Checkpoint — Phase 1 Prototype

IMPORTANT: This is a prototype using real test passport data.
NOT connected to any government database.
NOT an official immigration or border control system.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from PIL import Image

# ── Path setup for direct Python import fallback ──────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://127.0.0.1:8000")
API_TIMEOUT = 120  # seconds (OCR model load can take time on first call)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Passport Screening",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Premium dark theme with status color coding
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ──────────────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Header ──────────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white; padding: 24px 32px; border-radius: 12px;
        margin-bottom: 24px; border-left: 5px solid #f0a500;
    }
    .main-header h1 { font-size: 1.7rem; font-weight: 700; margin: 0; letter-spacing: 1px; }
    .main-header p  { font-size: 0.85rem; margin: 4px 0 0 0; opacity: 0.75; }

    /* ── Disclaimer ─────────────────────────────────────────────────────── */
    .disclaimer {
        background: rgba(240, 165, 0, 0.12); border: 1px solid #f0a500;
        border-radius: 8px; padding: 10px 16px; font-size: 0.78rem; color: #b8860b;
        margin-bottom: 16px;
    }

    /* ── Section headers ────────────────────────────────────────────────── */
    .section-header {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        color: #e0e0e0; padding: 10px 18px; border-radius: 8px;
        font-size: 0.9rem; font-weight: 600; letter-spacing: 0.5px;
        border-left: 4px solid #f0a500; margin: 20px 0 12px 0;
    }

    /* ── Field table ────────────────────────────────────────────────────── */
    .field-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 14px; border-bottom: 1px solid #2a2a3e;
        font-size: 0.87rem;
    }
    .field-label { color: #888; font-weight: 500; min-width: 170px; }
    .field-value { color: #e8e8e8; font-family: 'Roboto Mono', monospace; font-weight: 500; }
    .field-conf  { color: #4caf50; font-size: 0.77rem; min-width: 60px; text-align: right; }
    .field-conf.low  { color: #ff6b35; }
    .field-conf.med  { color: #f0a500; }

    /* ── MRZ display ────────────────────────────────────────────────────── */
    .mrz-box {
        background: #0d1117; border: 1px solid #333; border-radius: 8px;
        padding: 14px 16px; font-family: 'Roboto Mono', monospace;
        font-size: 0.92rem; color: #58d68d; letter-spacing: 2px;
        margin: 8px 0;
    }

    /* ── Status badges ──────────────────────────────────────────────────── */
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.5px;
    }
    .badge-pass    { background: #1b5e20; color: #a5d6a7; }
    .badge-fail    { background: #b71c1c; color: #ef9a9a; }
    .badge-review  { background: #e65100; color: #ffcc80; }
    .badge-na      { background: #37474f; color: #90a4ae; }
    .badge-match   { background: #1b5e20; color: #a5d6a7; }
    .badge-mismatch{ background: #b71c1c; color: #ef9a9a; }
    .badge-not-verified { background: #37474f; color: #90a4ae; }

    /* ── Overall verdict ────────────────────────────────────────────────── */
    .verdict-verified   { background: linear-gradient(135deg,#1b5e20,#2e7d32); color:#c8e6c9; }
    .verdict-needs-review { background: linear-gradient(135deg,#e65100,#ef6c00); color:#ffe0b2; }
    .verdict-not-found  { background: linear-gradient(135deg,#b71c1c,#c62828); color:#ffcdd2; }
    .verdict-pass   { background: linear-gradient(135deg,#1b5e20,#2e7d32); color:#c8e6c9; }
    .verdict-fail   { background: linear-gradient(135deg,#b71c1c,#c62828); color:#ffcdd2; }
    .verdict-review { background: linear-gradient(135deg,#e65100,#ef6c00); color:#ffe0b2; }
    .verdict-box {
        padding: 18px 24px; border-radius: 12px; text-align: center;
        font-size: 1.2rem; font-weight: 700; letter-spacing: 1px;
        margin: 16px 0;
    }

    /* ── Comparison table ───────────────────────────────────────────────── */
    .comp-row {
        display: flex; align-items: center; padding: 10px 16px;
        border-bottom: 1px solid #2a2a3e; font-size: 0.85rem;
    }
    .comp-row-matched     { border-left: 3px solid #4caf50; }
    .comp-row-mismatch    { border-left: 3px solid #f44336; background: rgba(244,67,54,0.06); }
    .comp-row-not-verified{ border-left: 3px solid #607d8b; }
    .comp-field  { min-width: 150px; font-weight: 600; color: #b0b0b0; }
    .comp-ocr    { flex: 1; color: #e0e0e0; font-family: 'Roboto Mono', monospace; }
    .comp-db     { flex: 1; color: #a0a0a0; font-family: 'Roboto Mono', monospace; }
    .comp-status { min-width: 120px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def badge(status: str) -> str:
    cls_map = {
        "PASS":         "badge-pass",
        "FAIL":         "badge-fail",
        "REVIEW":       "badge-review",
        "NOT_AVAILABLE":"badge-na",
        "MATCH":        "badge-match",
        "MATCHED":      "badge-match",
        "MISMATCH":     "badge-mismatch",
        "NOT_FOUND":    "badge-review",
        "NOT_VERIFIED": "badge-not-verified",
        "VERIFIED":     "badge-pass",
        "NEEDS_REVIEW": "badge-review",
    }
    sym_map = {
        "PASS":"✓","FAIL":"✗","REVIEW":"⚠","MATCH":"✓","MATCHED":"✓",
        "MISMATCH":"✗","NOT_VERIFIED":"–","VERIFIED":"✓","NEEDS_REVIEW":"⚠",
    }
    cls = cls_map.get(status.upper(), "badge-na")
    symbol = sym_map.get(status.upper(), "–")
    return f'<span class="badge {cls}">{symbol} {status}</span>'


def conf_label(conf: float) -> str:
    pct = int(conf * 100)
    cls = "low" if conf < 0.5 else "med" if conf < 0.75 else ""
    return f'<span class="field-conf {cls}">{pct}%</span>'


def field_row(label: str, value: str, conf: Optional[float] = None) -> str:
    conf_html = conf_label(conf) if conf is not None else ""
    disp = value if value else '<span style="color:#555;font-style:italic">Not detected</span>'
    return (
        f'<div class="field-row">'
        f'<span class="field-label">{label}</span>'
        f'<span class="field-value">{disp}</span>'
        f'{conf_html}'
        f'</div>'
    )


def call_validate_passport_api(img_bytes: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Call POST /api/v1/validate-passport and return JSON result or None on error."""
    try:
        resp = requests.post(
            f"{API_BASE_URL}/api/v1/validate-passport",
            files={"file": (filename, io.BytesIO(img_bytes), "image/png")},
            timeout=API_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. The OCR model may still be loading. Please try again.")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"API error: {exc.response.status_code} — {exc.response.text[:300]}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error calling API: {exc}")
        return None


def run_local_validate(img_bytes: bytes) -> Optional[Dict[str, Any]]:
    """
    Run OCR + validate_passport locally (without FastAPI) as a fallback.
    Used when the API server is not running.
    """
    try:
        from app.database.database import initialize_database
        from app.ocr.field_extractor import extract_all_fields
        from app.utils.image_utils import load_image_from_bytes as lib_load
        from app.validation.validator import validate_passport as _validate_passport

        initialize_database()
        img = lib_load(img_bytes)
        raw_fields = extract_all_fields(img, use_template=True)

        # Build OCR dict
        ocr_dict = {}
        for key, val in raw_fields.items():
            if isinstance(val, dict):
                ocr_dict[key] = {
                    "value": val.get("value", ""),
                    "confidence": val.get("confidence", 0.0),
                }
            else:
                ocr_dict[key] = {"value": str(val) if val else "", "confidence": 0.0}

        val_result = _validate_passport(raw_fields)

        return {
            "ocr_result": ocr_dict,
            "validation_result": val_result.model_dump(),
            "_source": "local",
        }
    except Exception as exc:
        st.error(f"Local processing error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1> PASSPORT SCREENING SYSTEM </h1>
    <p>INDIA BORDER CHECKPOINT PROTOTYPE &nbsp;|&nbsp; OCR + DATABASE VALIDATION</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>PROTOTYPE — TEST DATA</strong> &nbsp;|&nbsp;
    NOT an official government system &nbsp;|&nbsp;
    Final decisions remain with the authorised officer.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")

    # Check API health
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=3)
        api_ok = health.status_code == 200
    except Exception:
        api_ok = False

    if api_ok:
        st.success("✅ Server Online")
    else:
        st.warning("⚠️ Server Offline\n\nFalling back to local processing.")

    st.markdown("---")

    # ── Sample Test Images Download ───────────────────────────────────────────
    import zipfile as _zipfile
    _sample_dir = Path(__file__).parent / "sample_downloads"
    _sample_files = [f for f in _sample_dir.iterdir() if f.is_file()] if _sample_dir.is_dir() else []
    if _sample_files:
        st.markdown("### 🖼️ Sample Test Images")
        st.caption("Try the system with sample passport images")
        _zip_buf = io.BytesIO()
        with _zipfile.ZipFile(_zip_buf, "w", _zipfile.ZIP_DEFLATED) as _zf:
            for _fp in _sample_files:
                _zf.write(_fp, arcname=_fp.name)
        st.download_button(
            label="📥 Download Sample Images",
            data=_zip_buf.getvalue(),
            file_name="sample_images.zip",
            mime="application/zip",
            use_container_width=True,
        )
    else:
        st.markdown("### 🖼️ Sample Test Images")
        st.caption("No sample images available yet.")

    st.markdown("---")
    st.markdown("### 📋 Module Status")

    st.markdown("""
| Module | Status |
|--------|--------|
| Module 1 — OCR Extraction | ✅ Active |
| Module 2 — Database Validation | ✅ Active |
| Module 3 — Tampering Detection | ✅ Active (ELA / Meta / Model) |
| Module 4 — Face Verification | ✅ Active (DeepFace / ArcFace) |
""")

    st.markdown("---")
    st.markdown("### 📁 Database")
    st.markdown("Synthetic border checkpoint database (SQLite)")

# ─────────────────────────────────────────────────────────────────────────────
# Main — Upload
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">📤 STEP 1 — UPLOAD PASSPORT & CURRENT FACE IMAGES</div>', unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 1])

with col_upload:
    uploaded = st.file_uploader(
        "1. Passport Document Image (Required — Modules 1, 2, 3, 4)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Upload a passport image for OCR extraction, database verification, tampering analysis, and portrait extraction.",
    )

    uploaded_face = st.file_uploader(
        "2. Current Face Photo (Optional — Module 4 Face Verification)",
        type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        help="Upload a live/current photo of the person to compare against the passport photo.",
    )

    run_btn = st.button(
        "🔍 Run AI Screening",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )

with col_preview:
    if uploaded:
        c_p1, c_p2 = st.columns(2) if uploaded_face else (st.container(), None)
        with c_p1:
            try:
                st.image(Image.open(io.BytesIO(uploaded.getvalue())), caption="Passport Document", use_container_width=True)
            except Exception as exc:
                st.error(f"Cannot preview passport image: {exc}")
        if uploaded_face and c_p2 is not None:
            with c_p2:
                try:
                    st.image(Image.open(io.BytesIO(uploaded_face.getvalue())), caption="Current Face Photo", use_container_width=True)
                except Exception as exc:
                    st.error(f"Cannot preview current face photo: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# Processing
# ─────────────────────────────────────────────────────────────────────────────

if run_btn and uploaded:
    img_bytes = uploaded.getvalue()
    filename = uploaded.name or "passport.png"
    face_bytes = uploaded_face.getvalue() if uploaded_face else None
    face_filename = uploaded_face.name if uploaded_face else "current_face.png"

    with st.spinner("🔄 Processing — full screening pipeline (OCR + Validation + Tampering + Face Verification + Risk)..."):
        # Try Phase 5 unified screening endpoint first
        screening_report: Optional[Dict[str, Any]] = None
        use_phase5 = False

        if api_ok:
            try:
                files_dict = {"file": (filename, io.BytesIO(img_bytes), "image/jpeg")}
                if face_bytes:
                    files_dict["current_face_file"] = (face_filename, io.BytesIO(face_bytes), "image/jpeg")

                resp = requests.post(
                    f"{API_BASE_URL}/api/v1/screening/analyze",
                    files=files_dict,
                    timeout=API_TIMEOUT,
                )
                if resp.status_code == 200:
                    screening_report = resp.json()
                    use_phase5 = True
            except Exception:
                pass

        # Fallback: local run_screening
        if screening_report is None:
            try:
                from app.services.screening_service import run_screening
                screening_report = run_screening(
                    image_input=img_bytes,
                    current_face_image_input=face_bytes,
                    filename=filename,
                )
                use_phase5 = True
            except Exception as exc:
                st.warning(f"Phase 5 screening failed, falling back to Module 1+2 only: {exc}")

        # Further fallback: old Module 1+2 only
        if not use_phase5:
            if api_ok:
                result = call_validate_passport_api(img_bytes, filename)
                if result is None:
                    st.info("API unreachable. Switching to local processing...")
                    result = run_local_validate(img_bytes)
            else:
                result = run_local_validate(img_bytes)

            if result is None:
                st.error("❌ Processing failed. Please check the image and try again.")
                st.stop()

            # Wrap legacy result to look like screening_report
            screening_report = {
                "ocr":        {"status": "ok",   "result": result.get("ocr_result", {})},
                "validation": {"status": "ok",   "result": result.get("validation_result", {})},
                "ela":        {"status": "unavailable", "result": None},
                "metadata":   {"status": "unavailable", "result": None},
                "model_inference": {"status": "unavailable", "result": None, "checkpoint_available": False},
                "risk_assessment": {"status": "unavailable", "result": None},
                "module_statuses": {},
                "warnings": [],
                "errors": [],
            }

    # Extract module results for rendering
    val_data = (screening_report.get("validation", {}).get("result") or {})
    ocr_data = (screening_report.get("ocr", {}).get("result") or {})

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 2 — VALIDATION RESULT (shown FIRST for officer priority)
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">✅ MODULE 2 — DATABASE VALIDATION RESULT</div>', unsafe_allow_html=True)

    overall_status = val_data.get("overall_status", "NOT_FOUND")
    db_match = val_data.get("db_match", False)
    fuzzy_matched = val_data.get("fuzzy_matched", False)
    message = val_data.get("message", "")

    # ── Overall verdict banner ─────────────────────────────────────────────
    verdict_cls = {
        "VERIFIED":     "verdict-verified",
        "NEEDS_REVIEW": "verdict-needs-review",
        "NOT_FOUND":    "verdict-not-found",
    }.get(overall_status, "verdict-not-found")

    status_icon = {
        "VERIFIED": "✅",
        "NEEDS_REVIEW": "⚠️",
        "NOT_FOUND": "❌",
    }.get(overall_status, "❌")

    st.markdown(
        f'<div class="verdict-box {verdict_cls}">'
        f'{status_icon} DATABASE STATUS: {overall_status}'
        f'</div>',
        unsafe_allow_html=True,
    )

    if message:
        st.info(f"💬 {message}")

    if fuzzy_matched:
        st.warning("⚠️ Passport number was fuzzy-matched (not exact). Verify the number manually.")

    if overall_status == "NOT_FOUND":
        # ── NOT FOUND — show clear message, no field table ─────────────────
        st.error(
            "🚫 **No matching record in database** — Full manual verification required.\n\n"
            "The passport number extracted by OCR does not match any record in the database. "
            "The officer should verify all fields manually."
        )
    else:
        # ── Summary metrics ────────────────────────────────────────────────
        summary = val_data.get("summary", {})
        matched_count = summary.get("matched_count", 0)
        mismatch_count = summary.get("mismatch_count", 0)
        not_verified_count = summary.get("not_verified_count", 0)

        col_m, col_mm, col_nv = st.columns(3)
        col_m.metric("✅ Matched", matched_count)
        col_mm.metric("❌ Mismatch", mismatch_count)
        col_nv.metric("❓ Not Verified", not_verified_count)

        # ── Side-by-side photos ────────────────────────────────────────────
        photo_path = val_data.get("passport_photo_path")
        if photo_path:
            st.markdown('<div class="section-header">📸 PHOTO COMPARISON</div>', unsafe_allow_html=True)
            col_uploaded, col_db = st.columns(2)
            with col_uploaded:
                st.markdown("**Uploaded Document**")
                try:
                    upl_img = Image.open(io.BytesIO(img_bytes))
                    st.image(upl_img, use_container_width=True)
                except Exception:
                    st.warning("Could not display uploaded image.")

            with col_db:
                st.markdown("**Database Reference Photo**")
                ref_path = Path(ROOT) / photo_path
                if ref_path.exists():
                    try:
                        ref_img = Image.open(ref_path)
                        st.image(ref_img, use_container_width=True)
                    except Exception:
                        st.warning(f"Could not display reference photo: {ref_path}")
                else:
                    st.info(f"Reference photo not found at: {photo_path}")

        # ── Field comparison table ─────────────────────────────────────────
        st.markdown('<div class="section-header">📋 FIELD-BY-FIELD COMPARISON</div>', unsafe_allow_html=True)

        comparisons = val_data.get("field_comparisons", [])
        if comparisons:
            # Table header
            header_html = (
                '<div class="comp-row" style="background:#1a1a2e;border-bottom:2px solid #f0a500;">'
                '<span class="comp-field" style="color:#f0a500;">Field</span>'
                '<span class="comp-ocr" style="color:#f0a500;">OCR Value</span>'
                '<span class="comp-db" style="color:#f0a500;">Database Value</span>'
                '<span class="comp-status" style="color:#f0a500;">Status</span>'
                '</div>'
            )

            rows_html = ""
            for comp in comparisons:
                field_name = comp.get("field_name", "")
                ocr_val = comp.get("ocr_value", "")
                db_val = comp.get("db_value", "")
                status = comp.get("status", "NOT_VERIFIED")

                row_cls = {
                    "MATCHED": "comp-row-matched",
                    "MISMATCH": "comp-row-mismatch",
                    "NOT_VERIFIED": "comp-row-not-verified",
                }.get(status, "comp-row-not-verified")

                ocr_display = ocr_val if ocr_val else '<em style="color:#555;">—</em>'
                db_display = db_val if db_val else '<em style="color:#555;">—</em>'

                rows_html += (
                    f'<div class="comp-row {row_cls}">'
                    f'<span class="comp-field">{field_name}</span>'
                    f'<span class="comp-ocr">{ocr_display}</span>'
                    f'<span class="comp-db">{db_display}</span>'
                    f'<span class="comp-status">{badge(status)}</span>'
                    f'</div>'
                )

            st.markdown(
                f'<div style="background:#111827;border-radius:10px;overflow:hidden;">'
                f'{header_html}{rows_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 1 — OCR RESULTS (detailed extraction)
    # ─────────────────────────────────────────────────────────────────────────

    with st.expander("📋 MODULE 1 — OCR EXTRACTION DETAILS", expanded=False):
        def fv(field_data) -> tuple:
            """Extract (value, confidence) from a field dict or string."""
            if isinstance(field_data, dict):
                return str(field_data.get("value", "")), float(field_data.get("confidence", 0.0))
            return str(field_data or ""), 0.0

        rows_html = "".join([
            field_row("Full Name",       *fv(ocr_data.get("name", {}))),
            field_row("Passport Number", *fv(ocr_data.get("passport_number", {}))),
            field_row("Passport Type",   *fv(ocr_data.get("passport_type", {}))),
            field_row("Country Code",    *fv(ocr_data.get("country_code", {}))),
            field_row("Nationality",     *fv(ocr_data.get("nationality", {}))),
            field_row("Date of Birth",   *fv(ocr_data.get("date_of_birth", {}))),
            field_row("Place of Birth",  *fv(ocr_data.get("place_of_birth", {}))),
            field_row("Sex",             *fv(ocr_data.get("sex", {}))),
            field_row("Date of Issue",   *fv(ocr_data.get("date_of_issue", {}))),
            field_row("Place of Issue",  *fv(ocr_data.get("place_of_issue", {}))),
            field_row("Date of Expiry",  *fv(ocr_data.get("date_of_expiry", {}))),
        ])
        st.markdown(
            f'<div style="background:#111827;border-radius:10px;overflow:hidden;">{rows_html}</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 3 — TAMPERING DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">🔬 MODULE 3 — TAMPERING DETECTION</div>', unsafe_allow_html=True)

    ela_status  = screening_report.get("ela", {}).get("status", "unavailable")
    ela_res     = screening_report.get("ela", {}).get("result") or {}
    meta_status = screening_report.get("metadata", {}).get("status", "unavailable")
    meta_res    = screening_report.get("metadata", {}).get("result") or {}

    # ── ELA & Original Image Side-by-side ────────────────────────────────────
    st.markdown("#### 1. Error Level Analysis (ELA)")
    if ela_status == "ok" and ela_res.get("success"):
        col_orig, col_ela_vis = st.columns(2)
        with col_orig:
            st.markdown("**Original Passport Image**")
            try:
                orig_disp = Image.open(io.BytesIO(img_bytes))
                st.image(orig_disp, use_container_width=True)
            except Exception:
                st.warning("Could not display original image.")

        with col_ela_vis:
            st.markdown("**ELA Visualization (Compression Difference)**")
            ela_img_path = ela_res.get("save_path") or ela_res.get("ela_image_path")
            if ela_img_path:
                from pathlib import Path as _P
                ela_abs = _P(ela_img_path) if _P(ela_img_path).is_absolute() else ROOT / "data" / "forensics" / "ela_output" / ela_img_path
                if ela_abs.exists():
                    try:
                        from PIL import Image as _PIL
                        ela_disp = _PIL.open(ela_abs)
                        st.image(ela_disp, use_container_width=True)
                    except Exception:
                        st.info("ELA image generated but could not be rendered.")
                else:
                    st.info("ELA visualization file not accessible.")

        mean_ela = ela_res.get("mean_ela", 0.0)
        max_ela  = ela_res.get("max_ela", 0.0)
        ela_interp = ela_res.get("interpretation") or {}
        ela_band   = ela_interp.get("status", "LIKELY_GENUINE")
        ela_label  = ela_interp.get("label", "Consistent Compression")
        ela_desc   = ela_interp.get("description", "Error Level Analysis inspected.")

        ela_badge_cls = "badge-pass" if ela_band == "LIKELY_GENUINE" else ("badge-review" if ela_band == "INCONCLUSIVE" else "badge-fail")

        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);padding:14px 18px;border-radius:10px;border-left:4px solid #f0a500;margin:10px 0;">'
            f'<span class="badge {ela_badge_cls}" style="font-size:0.85rem;">Compression Analysis: {ela_label}</span><br/>'
            f'<p style="margin:8px 0 0 0;font-size:0.88rem;color:#dddddd;">{ela_desc}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

        with st.expander("🔬 Technical ELA Metrics & Compression Settings", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean ELA", f"{mean_ela:.2f}")
            c2.metric("Max ELA",  f"{max_ela:.2f}")
            c3.metric("JPEG Quality", f"{ela_res.get('quality_used', 90)}")
            c4.metric("Scale Factor", f"{ela_res.get('scale_used', 15)}")
            st.caption("ℹ️ Technical ELA measures error level variation after 90% JPEG recompression. Higher mean/max ELA values indicate greater re-compression differences across regions.")
    elif ela_status == "failed":
        st.error(f"ELA analysis failed: {screening_report.get('ela', {}).get('error')}")
    else:
        st.info("ELA analysis unavailable.")

    # ── Metadata Analysis ─────────────────────────────────────────────────────
    st.markdown("#### 2. Metadata & File Structure Analysis")
    if meta_status == "ok" and meta_res.get("success"):
        img_info = meta_res.get("image_info", {})
        st.markdown(
            f"**File Format:** `{img_info.get('format', '—')}` &nbsp;|&nbsp; "
            f"**Dimensions:** `{img_info.get('width')} × {img_info.get('height')} px` &nbsp;|&nbsp; "
            f"**Color Mode:** `{img_info.get('mode', '—')}` &nbsp;|&nbsp; "
            f"**File Size:** `{img_info.get('file_size', 0)} bytes`"
        )

        exif = meta_res.get("exif_fields", {})
        if exif:
            with st.expander("Extracted EXIF Metadata Fields", expanded=False):
                for k, v in exif.items():
                    st.markdown(f"- **{k}**: `{v}`")
        else:
            st.info("ℹ️ No EXIF metadata embedded in this image. (Normal for PNG scans/screenshots — does NOT indicate tampering).")

        obs = meta_res.get("observations", [])
        if obs:
            st.markdown("**Metadata Observations:**")
            for ob in obs:
                st.caption(f"📌 {ob}")
    elif meta_status == "failed":
        st.error(f"Metadata analysis failed: {screening_report.get('metadata', {}).get('error')}")
    else:
        st.info("Metadata analysis unavailable.")

    # ── EfficientNet-B0 Classifier ───────────────────────────────────────────
    st.markdown("#### 3. EfficientNet-B0 Tampering Classifier")
    inf_status = screening_report.get("model_inference", {}).get("status", "unavailable")
    inf_res    = screening_report.get("model_inference", {}).get("result") or {}
    ckpt_avail = screening_report.get("model_inference", {}).get("checkpoint_available", False)

    if inf_status == "ok" and inf_res.get("success"):
        score_tam = inf_res.get("score_tampered", 0.0)
        score_gen = inf_res.get("score_genuine", 0.0)
        diff      = inf_res.get("score_difference", score_tam - score_gen)

        cl_res = inf_res.get(
            "classification_result",
            inf_res.get("band", "INCONCLUSIVE"),
        )
        cl_category = inf_res.get(
            "classification_category",
            "INCONCLUSIVE",
        )
        cl_label = inf_res.get(
            "classification_label",
            "Inconclusive — Near 50–50",
        )
        cl_desc = inf_res.get("classification_desc", "")

        n_patches = inf_res.get("n_patches", 0)
        susp_regions = inf_res.get("suspicious_regions", [])

        # ── Category-specific UI styling ─────────────────────────────────────
        genuine_categories = {
            "STRONG_GENUINE_LEAN",
            "GENUINE_LEAN",
        }

        weak_genuine_categories = {
            "WEAK_GENUINE_LEAN",
        }

        # NOTE: WEAK_TAMPERING_LEAN is also POTENTIALLY_TAMPERED (positive diff)
        tampering_categories = {
            "TAMPERING_LEAN",
            "STRONG_TAMPERING_LEAN",
            "WEAK_TAMPERING_LEAN",
        }

        weak_tampering_categories: set = set()  # merged into tampering_categories above

        if cl_category in genuine_categories:
            badge_cls = "badge-pass"
            icon_symbol = "✅"
            banner_bg = (
                "background: rgba(27, 94, 32, 0.2); "
                "border: 1px solid #4caf50;"
            )

        elif cl_category in weak_genuine_categories:
            # Genuine-side but too close to zero — officer review needed
            badge_cls = "badge-review"
            icon_symbol = "⚠️"
            banner_bg = (
                "background: rgba(230, 81, 0, 0.15); "
                "border: 1px solid #ff9800;"
            )

        elif cl_category in tampering_categories:
            # All tampering variants (including WEAK_TAMPERING_LEAN) use red fail badge
            badge_cls = "badge-fail"
            icon_symbol = "🚨"
            banner_bg = (
                "background: rgba(183, 28, 28, 0.2); "
                "border: 1px solid #f44336;"
            )

        else:
            # INCONCLUSIVE
            badge_cls = "badge-review"
            icon_symbol = "⚠️"
            banner_bg = (
                "background: rgba(230, 81, 0, 0.2); "
                "border: 1px solid #ff9800;"
            )

        st.markdown(
            f'<div style="{banner_bg} padding:14px 18px; '
            f'border-radius:10px; margin:12px 0;">'
            f'<span class="badge {badge_cls}" style="font-size:0.9rem;">'
            f'{icon_symbol} Classifier Verdict: {cl_label}'
            f'</span>'
            f'<p style="margin:8px 0 0 0; font-size:0.88rem; '
            f'color:#e0e0e0;">{cl_desc}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Scores ───────────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Tampered Score", f"{score_tam:.3f}")
        c2.metric("Genuine Score", f"{score_gen:.3f}")
        c3.metric(
            "Score Difference",
            f"{diff:+.3f}",
            help="Difference = Tampered Score - Genuine Score",
        )
        c4.metric("Patches Scanned", f"{n_patches}")

        # ── Score Difference Interpretation ──────────────────────────────────
        with st.expander("📊 Score Difference Interpretation", expanded=False):
            st.markdown(
                """
                | Score Difference | Classification | Action |
                |---|---|---|
                | **≤ -0.25** | Strongly Genuine (High Confidence) | ✅ Clear |
                | **-0.25 to -0.10** | Genuine-Leaning (Moderate Confidence) | ✅ Clear |
                | **-0.10 to 0.00** | Borderline Genuine (Very Small Gap) | ⚠️ Officer Review |
                | **= 0.00** | Inconclusive (Exact 50–50) | ⚠️ Manual Verification |
                | **0.00 to +0.10** | Borderline Tampering (Tampered Side Leads) | 🚨 Verification Mandatory |
                | **+0.10 to +0.25** | Tampering-Leaning (Moderate Risk) | 🚨 Verification Required |
                | **> +0.25** | Strongly Tampered (High Risk) | 🚨 Fraud Investigation |
                """
            )

            st.caption(
                "KEY RULE: Score Difference = Tampered Score − Genuine Score. "
                "Exact zero (0.000) = mathematical 50-50. "
                "ANY positive value means the model assigns higher probability to the tampered class — "
                "even a small positive gap (e.g. +0.100) is classified as Borderline Tampering and flagged. "
                "Negative values indicate the genuine class leads."
            )

        # ── Suspicious regions ────────────────────────────────────────────────
        if susp_regions:
            st.markdown(
                "**🚨 Suspicious Patch Regions "
                "(Highest Tampering Confidence):**"
            )

            for idx, s_reg in enumerate(susp_regions, 1):
                box = s_reg.get("box", [])
                p_score = s_reg.get("score", 0.0)

                st.caption(
                    f"• **Region {idx}**: "
                    f"Box `[x0={box[0]}, y0={box[1]}, "
                    f"x1={box[2]}, y1={box[3]}]` — "
                    f"Tampered Confidence: **{p_score:.1%}**"
                )

        elif cl_category in genuine_categories:
            st.success(
                "✅ Scanned patches show no strong localized tampering anomalies."
            )

        st.caption(
            "ℹ️ KEY: Difference = Tampered − Genuine. "
            "≤ -0.25 = Strongly Genuine; "
            "-0.25 to -0.10 = Genuine-Leaning; "
            "-0.10 to 0.00 = Borderline Genuine (review); "
            "= 0.00 = Exact 50-50 (inconclusive); "
            "ANY positive value = Tampered side leads — flagged. "
            "0.00 to +0.10 = Borderline Tampered (verify); "
            "+0.10 to +0.25 = Tampering-Leaning; "
            "> +0.25 = Strongly Tampered."
        )

    elif not ckpt_avail or inf_res.get("band") == "MODEL_UNAVAILABLE":
        st.warning(
            "🔒 **Classifier Unavailable / Not Trained**: "
            "Trained EfficientNet-B0 checkpoint file not found. "
            "Classifier-based scoring is skipped for this document."
        )

    elif inf_status == "failed":
        st.error(
            f"Model inference failed: "
            f"{screening_report.get('model_inference', {}).get('error')}"
        )

    else:
        st.info("Model inference unavailable.")

    # ─────────────────────────────────────────────────────────────────────────
    # MODULE 4 — FACE VERIFICATION & AGE CONTEXT
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">👤 MODULE 4 — DATABASE-BASED FACE VERIFICATION</div>', unsafe_allow_html=True)

    face_status = screening_report.get("face_verification", {}).get("status", "unavailable")
    face_res    = screening_report.get("face_verification", {}).get("result") or {}

    if face_status == "ok" and face_res:
        verif_status = face_res.get("face_verification_status", "UNAVAILABLE")
        policy_case  = face_res.get("policy_case", "UNKNOWN")
        raw_dist     = face_res.get("raw_distance")
        thresh       = face_res.get("model_threshold")
        derived_sim  = face_res.get("derived_similarity_score")
        exp_text     = face_res.get("human_readable_explanation", "")
        db_portrait  = face_res.get("database_portrait_path")

        verdict_map = {
            "MATCH_HIGH_SIMILARITY": ("verdict-verified", "✅ MATCH — Face Verified"),
            "BORDERLINE_SIMILARITY": ("verdict-needs-review", "⚠️ INCONCLUSIVE — Manual Review Required"),
            "MISMATCH_POSSIBLE_AGING": ("verdict-needs-review", "⚠️ NO MATCH — Natural Aging Possible"),
            "MISMATCH_RECENT_DOCUMENT": ("verdict-not-found", "🚨 NO MATCH — Face Not Verified"),
            "MISMATCH_UNKNOWN_AGE": ("verdict-needs-review", "⚠️ NO MATCH — Unverified Issue Date"),
            "DATABASE_RECORD_NOT_FOUND": ("verdict-not-found", "🚫 DATABASE RECORD NOT FOUND — Face Verification Skipped"),
            "DATABASE_PORTRAIT_UNAVAILABLE": ("verdict-needs-review", "⚠️ DATABASE PORTRAIT UNAVAILABLE — Face Verification Skipped"),
            "FACE_DETECTION_FAILED": ("verdict-needs-review", "⚠️ FACE DETECTION FAILED — Manual Review Required"),
            "VERIFICATION_UNAVAILABLE": ("verdict-not-found", "❌ UNAVAILABLE — Manual Review Required"),
        }

        v_cls, v_title = verdict_map.get(policy_case, ("verdict-needs-review", f"⚠️ Status: {verif_status}"))

        st.markdown(
            f'<div class="verdict-box {v_cls}">{v_title}</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Derived Similarity", f"{derived_sim:.1f}%" if derived_sim is not None else "N/A")
        c2.metric("ArcFace Distance", f"{raw_dist:.4f}" if raw_dist is not None else "N/A")
        c3.metric("Model Threshold", f"{thresh:.4f}" if thresh is not None else "N/A")
        c4.metric("Document Age", f"{face_res.get('elapsed_years')} yrs" if face_res.get('elapsed_years') is not None else "N/A")

        if exp_text:
            st.info(f"💡 **Policy Analysis**: {exp_text}")

        # Reference Portrait vs Current Photo Side-by-Side
        col_ref_photo, col_curr_photo = st.columns(2)
        with col_ref_photo:
            st.markdown("**1. Database Passport Reference Portrait (SQLite Record)**")
            if db_portrait and Path(db_portrait).exists():
                try:
                    st.image(Image.open(db_portrait), use_container_width=True, caption=f"Reference Portrait ({Path(db_portrait).name})")
                except Exception as exc:
                    st.warning(f"Could not load database portrait image: {exc}")
            else:
                st.info("ℹ️ Database reference portrait path not set or file not found on disk.")

        with col_curr_photo:
            st.markdown("**2. Current Face Photo (Uploaded by Officer)**")
            if uploaded_face is not None:
                try:
                    st.image(Image.open(io.BytesIO(uploaded_face.getvalue())), use_container_width=True, caption="Live Face Photo")
                except Exception as exc:
                    st.warning(f"Could not preview current face photo: {exc}")
            else:
                st.info("ℹ️ Current face photo was not uploaded for comparison.")

    elif face_status == "failed":
        st.error(f"Module 4 Face Verification failed: {screening_report.get('face_verification', {}).get('error')}")
    else:
        st.info("ℹ️ Face verification unavailable (Current face photo was not uploaded).")

    # ─────────────────────────────────────────────────────────────────────────
    # UNIFIED SCREENING SUMMARY — ADVISORY RISK ASSESSMENT
    # ─────────────────────────────────────────────────────────────────────────

    st.markdown('<div class="section-header">⚡ UNIFIED SCREENING SUMMARY — ADVISORY RISK</div>',
                unsafe_allow_html=True)

    risk_status = screening_report.get("risk_assessment", {}).get("status", "unavailable")
    risk_res    = screening_report.get("risk_assessment", {}).get("result") or {}

    if risk_status == "ok" and risk_res:
        level       = risk_res.get("risk_level", "INSUFFICIENT_DATA")
        score_val   = risk_res.get("risk_score", 0)
        manual_rev  = risk_res.get("manual_review_recommended", True)

        level_cls = {
            "LOW":              "verdict-verified",
            "MEDIUM":           "verdict-needs-review",
            "HIGH":             "verdict-not-found",
            "INSUFFICIENT_DATA":"verdict-not-found",
        }.get(level, "verdict-not-found")

        level_icon = {
            "LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨", "INSUFFICIENT_DATA": "❓"
        }.get(level, "❓")

        st.markdown(
            f'<div class="verdict-box {level_cls}">{level_icon} ADVISORY RISK LEVEL: {level} '
            f'(Score: {score_val:.0f})</div>',
            unsafe_allow_html=True,
        )

        if manual_rev:
            st.error("🔍 **Manual Review Recommended**: Officer must inspect document before approval.")

        # Separated Evidence Breakdown
        col_db_ev, col_tamp_ev = st.columns(2)

        with col_db_ev:
            st.markdown("#### 📋 Database Inconsistencies (Module 2)")
            val_res_dict = screening_report.get("validation", {}).get("result") or {}
            db_match_flag = val_res_dict.get("db_match", False)
            comparisons_list = val_res_dict.get("field_comparisons", [])

            if not db_match_flag or val_res_dict.get("overall_status") == "NOT_FOUND":
                st.error("🚫 **Passport Record Not Found in Database**\n\nThe passport number extracted from the document does not match any record in the prototype database. All identity fields (Name, Date of Birth, Expiry, Issue Place) could not be verified.")
            else:
                mismatch_fields = [c for c in comparisons_list if c.get("status") == "MISMATCH"]
                unverified_fields = [c for c in comparisons_list if c.get("status") == "NOT_VERIFIED"]

                if mismatch_fields:
                    st.error("⚠️ **Confirmed Field Mismatches Detected:**")
                    for m in mismatch_fields:
                        fn  = m.get("field_name", "Field")
                        ocr = m.get("ocr_value", "—")
                        db  = m.get("db_value", "—")
                        st.markdown(f"• **{fn} Mismatch**: Document shows `{ocr}`, but Database record has `{db}`.")

                if unverified_fields:
                    st.warning("❓ **Unverified Fields (Missing or Extra Data):**")
                    for u in unverified_fields:
                        fn = u.get("field_name", "Field")
                        st.caption(f"• **{fn}**: Field could not be automatically cross-checked.")

                if not mismatch_fields and not unverified_fields:
                    st.success("✅ **All Fields Verified**: Every extracted OCR field matches the database record.")

        with col_tamp_ev:
            st.markdown("#### 🔬 Image Tampering Analysis (Module 3)")

            # ELA summary line
            ela_res_dict = screening_report.get("ela", {}).get("result") or {}
            ela_interp_dict = ela_res_dict.get("interpretation") or {}
            ela_label_str = ela_interp_dict.get("label", "Consistent Compression")

            # Classifier summary line
            inf_res_dict = screening_report.get("model_inference", {}).get("result") or {}
            cl_label_str = inf_res_dict.get("classification_label", "Model Unavailable")
            cl_res_str = inf_res_dict.get("classification_result", "INCONCLUSIVE")
            diff_val = inf_res_dict.get("score_difference", 0.0)

            st.markdown(f"• **Compression Analysis (ELA)**: `{ela_label_str}`")
            st.markdown(f"• **Classifier Verdict**: `{cl_label_str}` (Difference: {diff_val:+.3f})")

            cl_category_str = inf_res_dict.get("classification_category", "")
            if cl_category_str in {"STRONG_TAMPERING_LEAN", "TAMPERING_LEAN"}:
                st.error("🚨 **Tampering Alert**: Tampered score clearly dominates. Document must be sent for manual verification.")
            elif cl_category_str == "WEAK_TAMPERING_LEAN":
                st.error("⚠️ **Tampering Detected (Borderline)**: Tampered score exceeds genuine score. Even a small positive difference means the model voted tampered. Secondary officer verification is mandatory.")
            elif cl_res_str == "INCONCLUSIVE":
                st.warning("⚠️ **Inconclusive (Exact 50–50)**: Score difference is zero — equal probability between genuine and tampered. Flagged for manual verification.")
            elif cl_category_str == "WEAK_GENUINE_LEAN":
                st.warning("⚠️ **Borderline Genuine**: Genuine score leads by a very narrow margin. Tampering cannot be fully excluded. Secondary review advised.")
            elif cl_res_str == "LIKELY_GENUINE":
                st.success("✅ **Likely Genuine**: No localized tampering patterns flagged across document regions.")
            else:
                st.warning("⚠️ **Result Uncertain**: Manual verification recommended.")

        unavail = risk_res.get("unavailable_signals", [])
        if unavail:
            with st.expander("ℹ️ Unavailable Analysis Signals", expanded=False):
                for sig in unavail:
                    st.caption(f"• {sig}")

        st.caption(f"⚖️ {risk_res.get('disclaimer', '')}")

    elif risk_status == "failed":
        st.error(f"Risk assessment failed: {screening_report.get('risk_assessment', {}).get('error')}")
    else:
        st.info("Risk assessment unavailable.")

    # ── Module status summary ────────────────────────────────────────────────
    mod_statuses = screening_report.get("module_statuses", {})
    if mod_statuses:
        with st.expander("Module Status Summary", expanded=False):
            status_icons = {"ok": "✅", "failed": "❌", "unavailable": "⚪"}
            for mod, stat in mod_statuses.items():
                icon = status_icons.get(stat, "—")
                st.markdown(f"{icon} **{mod.replace('_', ' ').title()}**: {stat}")

    # ── Report export ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📥 EXPORT SCREENING REPORT</div>',
                unsafe_allow_html=True)

    col_json, col_csv, col_pdf = st.columns(3)

    screening_id = screening_report.get("screening_id", "report")

    def _safe_json(obj):
        if isinstance(obj, dict):  return {k: _safe_json(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_safe_json(i) for i in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None: return obj
        return str(obj)

    with col_json:
        try:
            report_json = json.dumps(_safe_json(screening_report), indent=2)
            st.download_button(
                label="⬇️ Download JSON Report",
                data=report_json,
                file_name=f"screening_{screening_id[:8]}.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"JSON export unavailable: {exc}")

    with col_csv:
        try:
            import csv as _csv
            val_r  = (screening_report.get("validation") or {}).get("result") or {}
            ela_r  = (screening_report.get("ela") or {}).get("result") or {}
            inf_r  = (screening_report.get("model_inference") or {}).get("result") or {}
            risk_r = (screening_report.get("risk_assessment") or {}).get("result") or {}

            csv_row = {
                "screening_id":    screening_id,
                "timestamp":       screening_report.get("timestamp"),
                "filename":        filename,
                "ocr_status":      mod_statuses.get("ocr"),
                "validation":      val_r.get("overall_status"),
                "db_match":        val_r.get("db_match"),
                "ela_mean":        (ela_r.get("mean_ela")),
                "ela_max":         (ela_r.get("max_ela")),
                "model_prediction":inf_r.get("classification_label", inf_r.get("predicted_label")),
                "score_genuine":   inf_r.get("score_genuine"),
                "score_tampered":  inf_r.get("score_tampered"),
                "score_difference":inf_r.get("score_difference"),
                "risk_level":      risk_r.get("risk_level"),
                "risk_score":      risk_r.get("risk_score"),
                "manual_review":   risk_r.get("manual_review_recommended"),
            }

            buf = io.StringIO()
            writer = _csv.DictWriter(buf, fieldnames=list(csv_row.keys()))
            writer.writeheader()
            writer.writerow(csv_row)

            st.download_button(
                label="⬇️ Download CSV Report",
                data=buf.getvalue(),
                file_name=f"screening_{screening_id[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"CSV export unavailable: {exc}")

    with col_pdf:
        try:
            from app.utils.pdf_generator import generate_screening_pdf_report
            pdf_bytes = generate_screening_pdf_report(screening_report)
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"screening_report_{screening_id[:8]}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"PDF export unavailable: {exc}")

    st.caption(
        "⚠️ PROTOTYPE DISCLAIMER: Exported reports are NOT official government documents. "
        "A human officer retains full responsibility for the final screening decision."
    )


