# AI-Based Indian Passport Screening System
## Phase 1 Prototype — India → Bangladesh Border Checkpoint

> **⚠️ DISCLAIMER: PROTOTYPE — SYNTHETIC TEST DATA ONLY**
> This system is NOT connected to any government database. It is NOT an official Government of India or Government of Bangladesh system. All data is fictional. Final decisions remain with authorised human officers.

---

## Overview

An AI-assisted document screening prototype for the India→Bangladesh checkpoint. The system helps an authorised officer verify whether a presented Indian passport contains internally consistent information and matches a controlled synthetic test database.

### What This System Does
- Reads passport information via OCR (EasyOCR)
- Parses and validates the Machine Readable Zone (MRZ) per ICAO 9303
- Runs rule-based validation (expiry, format, cross-field consistency)
- Compares extracted data against a controlled SQLite prototype database
- Displays results on a Streamlit dashboard

### What This System Does NOT Do
- Connect to any real government database
- Make legally binding immigration decisions
- Detect tampering (Module 3 — coming later)
- Perform face verification (Module 4 — coming later)

---

## Phase 1 Scope

| Module | Status | Description |
|--------|--------|-------------|
| **Module 1 — OCR** | ✅ Implemented | EasyOCR field extraction + MRZ parsing |
| **Module 2 — Document Validation** | ✅ Implemented | Rule-based validation + SQLite DB comparison |
| Module 3 — Tampering Detection | 🔜 Not yet | EfficientNet-B0, ELA, U-Net |
| Module 4 — Face Verification | 🔜 Not yet | DeepFace, ArcFace, RetinaFace |

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| OCR | EasyOCR |
| Image Processing | OpenCV, Pillow, NumPy |
| Database | SQLite (stdlib) |
| Data | Pandas |
| Validation | Pydantic v2 |
| Testing | Pytest |

---

## Project Structure

```
prototype/
│
├── app/
│   ├── main.py                   ← FastAPI application
│   ├── api/
│   │   ├── ocr_routes.py         ← POST /api/v1/ocr
│   │   └── validation_routes.py  ← POST /api/v1/validate, /screen
│   ├── core/
│   │   ├── config.py             ← Central configuration
│   │   └── logging_config.py     ← Structured logging
│   ├── database/
│   │   ├── database.py           ← SQLite connection management
│   │   ├── models.py             ← DDL + SQL queries
│   │   └── seed.py               ← Synthetic data seeder
│   ├── ocr/
│   │   ├── engine.py             ← EasyOCR singleton wrapper
│   │   ├── preprocessing.py      ← Image preprocessing pipeline
│   │   ├── field_extractor.py    ← Template + heuristic extraction
│   │   ├── mrz_parser.py         ← ICAO 9303 MRZ parser + check digits
│   │   └── schemas.py            ← Pydantic OCR schemas
│   ├── validation/
│   │   ├── rules.py              ← Individual validation rules
│   │   ├── consistency.py        ← Cross-field consistency checks
│   │   ├── validator.py          ← Validation orchestrator + DB lookup
│   │   └── schemas.py            ← Pydantic validation schemas
│   └── utils/
│       ├── image_utils.py        ← Image loading/encoding helpers
│       └── date_utils.py         ← Date parsing + normalisation
│
├── frontend/
│   └── streamlit_app.py          ← Streamlit dashboard
│
├── data/
│   ├── synthetic/                ← Synthetic test images
│   │   └── generate_test_image.py
│   ├── samples/                  ← Test samples (not committed)
│   └── database/                 ← SQLite DB files (not committed)
│
├── tests/
│   ├── test_ocr.py               ← Preprocessing + utils tests
│   ├── test_mrz.py               ← MRZ parser + check digit tests
│   ├── test_validation.py        ← Validation rules + integration tests
│   ├── test_database.py          ← DB connection + seed tests
│   └── test_api.py               ← FastAPI endpoint tests
│
├── scripts/
│   └── seed_database.py          ← Standalone DB seeder
│
├── conftest.py                   ← Pytest configuration
├── pytest.ini                    ← Pytest settings
├── requirements.txt              ← Python dependencies
├── .env.example                  ← Environment variable template
├── .gitignore                    ← Git ignore rules
├── run.py                        ← Convenience launcher
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10 or newer
- pip

### 1. Clone the repository

```bash
git clone <repository-url>
cd prototype
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** EasyOCR requires PyTorch. The first install may take several minutes. The first time OCR runs, models (~100MB) will be downloaded automatically.

### 4. Copy environment file

```bash
copy .env.example .env        # Windows
cp .env.example .env          # Linux/macOS
```

Edit `.env` if needed (defaults work for local development).

---

## Database Setup

Initialise and seed the SQLite database with synthetic test records:

```bash
python scripts/seed_database.py
```

This is **idempotent** — safe to run multiple times.

### Synthetic Test Records

| Passport Number | Name | Status |
|-----------------|------|--------|
| A1234567 | ARJUN KUMAR SHARMA | VALID |
| B2345678 | PRIYA MEENAKSHI IYER | VALID |
| C3456789 | RAHUL SINGH PATEL | VALID |
| D4567890 | SUNITA DEVI VERMA | VALID |
| E5678901 | VIKRAM NAIR KRISHNAN | VALID |
| F6789012 | ANANYA ROY CHATTERJEE | EXPIRED |
| G7890123 | DEEPAK GUPTA MISHRA | EXPIRED |
| H8901234 | KAVITA SHARMA JOSHI | FLAGGED_FOR_REVIEW |
| J0123456 | MOHAN DAS MALHOTRA | FLAGGED_FOR_REVIEW |
| K1234560 | TEST INVALID RECORD | INVALID |

---

## Generate Synthetic Test Images

```bash
python data/synthetic/generate_test_image.py
```

Generates clearly synthetic (non-real) passport-style PNG images for testing.

---

## Running the Application

### FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Or using the launcher:

```bash
python run.py api
```

- API base: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Streamlit Dashboard

Open a **second terminal**:

```bash
streamlit run frontend/streamlit_app.py
```

Or using the launcher:

```bash
python run.py ui
```

> The dashboard calls the FastAPI server automatically. If the API is offline, it falls back to local processing.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Health check |
| GET | `/docs` | OpenAPI documentation |
| POST | `/api/v1/ocr` | Upload passport image → OCR result |
| POST | `/api/v1/validate` | Submit fields → validation result |
| POST | `/api/v1/screen` | Upload image → OCR + Validation combined |

### Example: `/api/v1/validate`

**Request:**
```json
{
  "fields": {
    "passport_number": {"value": "A1234567", "confidence": 0.95},
    "name": {"value": "ARJUN KUMAR SHARMA", "confidence": 0.92},
    "date_of_birth": {"value": "1990-05-15", "confidence": 0.91},
    "date_of_issue": {"value": "2019-03-20", "confidence": 0.90},
    "date_of_expiry": {"value": "2029-03-19", "confidence": 0.90},
    "country_code": {"value": "IND", "confidence": 0.99},
    "passport_type": {"value": "P", "confidence": 0.99}
  }
}
```

**Response:**
```json
{
  "validation_status": "PASS",
  "checks": {
    "required_fields": {"status": "PASS", "reason": ""},
    "passport_expiry":  {"status": "PASS", "reason": "VALID (expires 2029-03-19)"},
    "database_match":   {"status": "PASS", "reason": "Database record found and fields match."}
  },
  "database_record_found": true,
  "recommendation": "All automated checks passed. Final decision rests with the authorised officer."
}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_mrz.py -v
pytest tests/test_validation.py -v
pytest tests/test_api.py -v
```

> **Note:** Tests that involve EasyOCR use mocking to avoid requiring the ML model during test runs. The MRZ and validation tests run entirely without any ML dependencies.

---

## Synthetic Data Policy

- **All passport data in this system is fictional.** No real names, passport numbers, dates, or identities are used.
- **No real passport images.** Synthetic images are generated using Pillow (text on template).
- **No real faces.** Photo regions contain placeholder graphics only.
- **Never commit** `.env`, database files (`*.db`), or uploaded images.
- See `.gitignore` for excluded paths.

---

## Architecture — Future Integration

Modules 3 and 4 can be integrated into the existing `ScreeningPipeline` pattern in `validation_routes.py`:

```python
# Future: Module 3
# tampering_result = run_tampering_detection(img)  # EfficientNet-B0 + ELA

# Future: Module 4
# face_result = run_face_verification(img, portrait)  # DeepFace / ArcFace
```

The `/api/v1/screen` endpoint already returns `tampering_detection` and `face_verification` fields (currently `"NOT_IMPLEMENTED"`), ready to be populated by future modules.

---

## Known Limitations

1. **OCR accuracy** depends on image quality. Blurry or low-resolution images reduce field extraction accuracy.
2. **Template regions** in `field_extractor.py` are tuned for the synthetic passport template. Real passport layouts will require template recalibration.
3. **MRZ check digits** in the synthetic test images may not all be mathematically correct (the images are for visual testing; exact MRZ arithmetic is validated in unit tests).
4. **No GPU** by default. Set `OCR_GPU=true` in `.env` if a CUDA GPU is available for faster OCR.
5. **First OCR run** will download EasyOCR models (~100MB). Subsequent runs are fast.
6. **Database** is a controlled prototype test dataset only. It does not represent real border control records.

---

## Security Notes

- No secrets or API keys required.
- No real PII processed or stored.
- Uploaded images are not persisted to disk by default.
- The system logs minimal information — no identity data is written to logs.

---

*Phase 1 Prototype — AI Document Screening System*
*India → Bangladesh Border Checkpoint Demo*
*NOT AN OFFICIAL GOVERNMENT SYSTEM*
