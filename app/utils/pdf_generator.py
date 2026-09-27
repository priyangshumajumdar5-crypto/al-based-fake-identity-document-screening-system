"""
PDF Report Generator Utility — Module 3 & Unified Screening Report.

Generates a clean, downloadable PDF report (bytes) from screening pipeline data
without requiring external third-party PDF dependencies.
"""
from __future__ import annotations

import datetime
import io
from typing import Any, Dict, List, Optional


def generate_screening_pdf_report(report: Dict[str, Any]) -> bytes:
    """
    Generate a standalone PDF binary document for the passport screening report.

    Args:
        report: Unified screening report dictionary from run_screening().

    Returns:
        PDF binary bytes ready for download or saving.
    """
    lines: List[str] = []

    def add_line(text: str):
        lines.append(text)

    # ── Extract Report Data ───────────────────────────────────────────────────
    screening_id = report.get("screening_id", "N/A")
    timestamp    = report.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat())
    filename     = report.get("filename", "passport.jpg")

    val_data  = (report.get("validation", {}).get("result") or {})
    ela_data  = (report.get("ela", {}).get("result") or {})
    inf_data  = (report.get("model_inference", {}).get("result") or {})
    face_data = (report.get("face_verification", {}).get("result") or {})
    risk_data = (report.get("risk_assessment", {}).get("result") or {})

    risk_level = risk_data.get("risk_level", "UNKNOWN")
    risk_score = risk_data.get("risk_score", 0.0)

    # ── Build Printable Text Content ─────────────────────────────────────────
    add_line("================================================================================")
    add_line("                   AI PASSPORT FRAUD SCREENING REPORT                           ")
    add_line("                 INDIA - BANGLADESH BORDER CHECKPOINT PROTOTYPE                  ")
    add_line("================================================================================")
    add_line("")
    add_line(f"Screening ID : {screening_id}")
    add_line(f"Timestamp    : {timestamp}")
    add_line(f"Document File: {filename}")
    add_line("")
    add_line("--------------------------------------------------------------------------------")
    add_line("1. OVERALL ADVISORY RISK ASSESSMENT")
    add_line("--------------------------------------------------------------------------------")
    add_line(f"  * Risk Level       : {risk_level}")
    add_line(f"  * Risk Score       : {risk_score:.1f} pts")
    add_line(f"  * Recommendation   : {'MANUAL OFFICER REVIEW RECOMMENDED' if risk_data.get('manual_review_recommended', True) else 'STANDARD INSPECTION'}")
    add_line("")

    # Module 2 Breakdown
    add_line("--------------------------------------------------------------------------------")
    add_line("2. MODULE 2: DATABASE VALIDATION BREAKDOWN")
    add_line("--------------------------------------------------------------------------------")
    db_status = val_data.get("overall_status", "NOT_RUN")
    add_line(f"  * Overall Database Status: {db_status}")
    add_line(f"  * Database Record Match  : {'MATCHED' if val_data.get('db_match') else 'NOT MATCHED / RECORD NOT FOUND'}")
    add_line("")
    add_line("  Field Comparisons:")
    comparisons = val_data.get("field_comparisons", [])
    if comparisons:
        for c in comparisons:
            fn  = c.get("field_name", "Field")
            ocr = c.get("ocr_value", "—")
            db  = c.get("db_value", "—")
            st  = c.get("status", "NOT_VERIFIED")
            add_line(f"    - {fn:<18} | OCR: {ocr:<20} | DB: {db:<20} | Status: {st}")
    else:
        add_line("    - No field comparison data available (Passport record not found in database).")
    add_line("")

    # Module 3 Breakdown
    add_line("--------------------------------------------------------------------------------")
    add_line("3. MODULE 3: TAMPERING ANALYSIS (ELA & EFFICIENTNET-B0 CLASSIFIER)")
    add_line("--------------------------------------------------------------------------------")

    # ELA
    ela_interp = ela_data.get("interpretation") or {}
    mean_ela = ela_data.get("mean_ela", 0.0)
    max_ela  = ela_data.get("max_ela", 0.0)
    add_line("  [3A. Error Level Analysis (ELA)]")
    add_line(f"    - Technical Metrics  : Mean ELA = {mean_ela:.2f}, Max ELA = {max_ela:.2f}")
    add_line(f"    - User Interpretation: {ela_interp.get('label', 'Standard Compression')}")
    add_line(f"    - Explanation        : {ela_interp.get('description', 'ELA compression variance inspected.')}")
    add_line("")

    # EfficientNet-B0
    score_tam = inf_data.get("score_tampered", 0.0)
    score_gen = inf_data.get("score_genuine", 0.0)
    diff      = inf_data.get("score_difference", score_tam - score_gen)
    cl_label  = inf_data.get("classification_label", inf_data.get("band", "MODEL_UNAVAILABLE"))
    cl_desc   = inf_data.get("classification_desc", "")

    add_line("  [3B. EfficientNet-B0 Tampering Classifier]")
    add_line(f"    - Decision Classification: {cl_label}")
    add_line(f"    - Tampered Score         : {score_tam:.3f}")
    add_line(f"    - Genuine Score          : {score_gen:.3f}")
    add_line(f"    - Score Difference       : {diff:+.3f} (Tampered Score - Genuine Score)")
    add_line(f"    - Patches Scanned        : {inf_data.get('n_patches', 0)}")
    add_line(f"    - Explanation            : {cl_desc}")
    add_line("")

    # Module 4 Breakdown
    add_line("--------------------------------------------------------------------------------")
    add_line("4. MODULE 4: FACE VERIFICATION & AGING CONTEXT")
    add_line("--------------------------------------------------------------------------------")
    fv_status = face_data.get("face_verification_status", "UNAVAILABLE")
    fv_policy = face_data.get("policy_case", "NOT_RUN")
    add_line(f"  * Status        : {fv_status}")
    add_line(f"  * Policy Outcome: {fv_policy}")
    add_line(f"  * Distance      : {face_data.get('raw_distance', 'N/A')}")
    add_line(f"  * Explanation   : {face_data.get('human_readable_explanation', 'No face comparison performed.')}")
    add_line("")

    add_line("================================================================================")
    add_line("                       OFFICER VERIFICATION SIGN-OFF                            ")
    add_line("================================================================================")
    add_line("Officer Name: ___________________________    Badge ID: ________________________")
    add_line("Decision:     [  ] PASSED    [  ] REJECTED    [  ] REFERRED TO SECONDARY")
    add_line("Signature:    ___________________________    Date    : ________________________")
    add_line("================================================================================")

    # ── Convert text lines into valid PDF 1.4 binary document ────────────────
    return _build_simple_pdf(lines)


def _build_simple_pdf(lines: List[str]) -> bytes:
    """
    Construct a clean PDF 1.4 document from text lines using pure Python.
    Zero external dependencies required.
    """
    # PDF Page dimensions & margins (Letter / A4 size: 612 x 792 points)
    page_w, page_h = 612, 792
    margin_x = 40
    start_y  = 750
    line_h   = 11
    font_sz  = 8.5

    # Group lines into pages (approx 62 lines per page)
    lines_per_page = 62
    page_chunks = [lines[i:i + lines_per_page] for i in range(0, len(lines), lines_per_page)]

    # We will build PDF objects
    pdf_objects: List[bytes] = []

    def new_obj(content: bytes) -> int:
        pdf_objects.append(content)
        return len(pdf_objects)

    # Catalog & Page Tree placeholder indices
    # Obj 1: Catalog
    # Obj 2: Pages Parent
    # Obj 3: Font (Courier)
    font_obj_id = 3

    page_obj_ids: List[int] = []

    # Create stream objects for each page
    for chunk in page_chunks:
        stream_lines = []
        stream_lines.append("BT")
        stream_lines.append(f"/F1 {font_sz} Tf")
        stream_lines.append(f"{line_h} TL")
        stream_lines.append(f"{margin_x} {start_y} Td")

        for line in chunk:
            # Escape PDF text characters (backslashes and parentheses)
            safe_text = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({safe_text}) '")

        stream_lines.append("ET")
        stream_data = "\n".join(stream_lines).encode("latin-1")

        # Create Stream Obj
        stream_obj_str = (
            f"<< /Length {len(stream_data)} >>\n"
            f"stream\n"
        ).encode("latin-1") + stream_data + b"\nendstream"

        stream_id = new_obj(stream_obj_str)

        # Create Page Obj (will reference stream_id)
        # We'll create page objects after 3 core objects
        page_obj_ids.append(stream_id)

    # Now construct object array properly:
    # Obj 1: Catalog
    # Obj 2: Pages
    # Obj 3: Font
    # Obj 4..: Page Objs and Stream Objs

    final_objs: Dict[int, bytes] = {}
    next_id = 1

    catalog_id = 1
    pages_id   = 2
    font_id    = 3
    next_id    = 4

    font_obj_bytes = b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"
    final_objs[font_id] = font_obj_bytes

    real_page_ids = []
    for stream_id in page_obj_ids:
        p_id = next_id
        next_id += 1
        s_id = next_id
        next_id += 1

        stream_bytes = pdf_objects[stream_id - 1]
        final_objs[s_id] = stream_bytes

        page_bytes = (
            f"<< /Type /Page /Parent {pages_id} 0 R "
            f"/MediaBox [0 0 {page_w} {page_h}] "
            f"/Contents {s_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("latin-1")
        final_objs[p_id] = page_bytes
        real_page_ids.append(p_id)

    # Pages Obj
    kids_str = " ".join([f"{pid} 0 R" for pid in real_page_ids])
    pages_bytes = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(real_page_ids)} >>".encode("latin-1")
    final_objs[pages_id] = pages_bytes

    # Catalog Obj
    catalog_bytes = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1")
    final_objs[catalog_id] = catalog_bytes

    # Write PDF stream
    pdf_buf = io.BytesIO()
    pdf_buf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: Dict[int, int] = {}
    max_id = max(final_objs.keys())

    for obj_i in range(1, max_id + 1):
        offsets[obj_i] = pdf_buf.tell()
        pdf_buf.write(f"{obj_i} 0 obj\n".encode("latin-1"))
        pdf_buf.write(final_objs[obj_i])
        pdf_buf.write(b"\nendobj\n")

    xref_offset = pdf_buf.tell()
    pdf_buf.write(f"xref\n0 {max_id + 1}\n".encode("latin-1"))
    pdf_buf.write(b"0000000000 65535 f \n")
    for obj_i in range(1, max_id + 1):
        pdf_buf.write(f"{offsets[obj_i]:010d} 00000 n \n".encode("latin-1"))

    pdf_buf.write(
        f"trailer\n<< /Size {max_id + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    )

    return pdf_buf.getvalue()
