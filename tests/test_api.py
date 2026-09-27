"""
Pytest tests for FastAPI endpoints.

Tests: /api/v1/validate and /api/v1/screen endpoints.
Note: /api/v1/ocr requires an image upload + EasyOCR, tested via mocking.
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_image() -> bytes:
    """Create a minimal valid PNG image for upload tests."""
    img = Image.new("RGB", (100, 100), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _valid_fields():
    return {
        "name":           {"value": "ARJUN KUMAR SHARMA", "confidence": 0.95},
        "passport_number":{"value": "A1234567",           "confidence": 0.92},
        "date_of_birth":  {"value": "1990-05-15",         "confidence": 0.91},
        "date_of_issue":  {"value": "2019-03-20",         "confidence": 0.90},
        "date_of_expiry": {"value": "2029-03-19",         "confidence": 0.90},
        "country_code":   {"value": "IND",                "confidence": 0.98},
        "passport_type":  {"value": "P",                  "confidence": 0.99},
        "sex":            {"value": "M",                  "confidence": 0.97},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_docs_available(self):
        resp = client.get("/docs")
        assert resp.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/validate
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateEndpoint:
    def test_valid_fields_returns_200(self):
        payload = {"fields": _valid_fields()}
        resp = client.post("/api/v1/validate", json=payload)
        assert resp.status_code == 200

    def test_response_has_validation_status(self):
        payload = {"fields": _valid_fields()}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert "validation_status" in data

    def test_response_has_checks(self):
        payload = {"fields": _valid_fields()}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert "checks" in data
        checks = data["checks"]
        assert "required_fields" in checks
        assert "passport_expiry" in checks
        assert "database_match" in checks

    def test_valid_passport_passes_required_fields(self):
        payload = {"fields": _valid_fields()}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert data["checks"]["required_fields"]["status"] == "PASS"

    def test_expired_passport_fails_expiry_check(self):
        fields = _valid_fields()
        fields["date_of_expiry"]["value"] = "2020-01-01"
        payload = {"fields": fields}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert data["checks"]["passport_expiry"]["status"] == "FAIL"

    def test_missing_passport_number_fails(self):
        fields = _valid_fields()
        fields["passport_number"]["value"] = ""
        payload = {"fields": fields}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert data["checks"]["required_fields"]["status"] == "FAIL"

    def test_unknown_passport_not_found(self):
        fields = _valid_fields()
        fields["passport_number"]["value"] = "Z9999999"
        payload = {"fields": fields}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert data["database_record_found"] is False

    def test_wrong_country_code_fails(self):
        fields = _valid_fields()
        fields["country_code"]["value"] = "USA"
        payload = {"fields": fields}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert data["checks"]["country_code"]["status"] == "FAIL"

    def test_response_has_recommendation(self):
        payload = {"fields": _valid_fields()}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert "recommendation" in data
        assert len(data["recommendation"]) > 5

    def test_with_mrz_data(self):
        payload = {
            "fields": _valid_fields(),
            "mrz": {
                "line1": "P<INDSHARMA<<ARJUN<KUMAR<<<<<<<<<<<<<<<<<<<<",
                "line2": "A12345674IND9005152M2903199<<<<<<<<<<<<<<<2",
                "all_check_digits_valid": True,
                "check_digits": {
                    "passport_number": {"valid": True, "value": "4", "expected": "4"},
                    "date_of_birth":   {"valid": True, "value": "2", "expected": "2"},
                    "date_of_expiry":  {"valid": True, "value": "9", "expected": "9"},
                    "optional_data":   {"valid": True, "value": "0", "expected": "0"},
                    "composite":       {"valid": True, "value": "2", "expected": "2"},
                },
                "parse_errors": [],
            },
        }
        resp = client.post("/api/v1/validate", json=payload)
        assert resp.status_code == 200

    def test_invalid_passport_number_format(self):
        fields = _valid_fields()
        fields["passport_number"]["value"] = "12345678"  # no leading letter
        payload = {"fields": fields}
        resp = client.post("/api/v1/validate", json=payload)
        data = resp.json()
        assert data["checks"]["passport_number_format"]["status"] == "FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# /api/v1/ocr (mocked — avoids requiring EasyOCR in unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class TestOCREndpoint:
    def test_no_file_returns_422(self):
        resp = client.post("/api/v1/ocr")
        assert resp.status_code == 422

    def test_empty_file_returns_error(self):
        resp = client.post(
            "/api/v1/ocr",
            files={"file": ("empty.png", b"", "image/png")},
            data={"use_template": "true"},
        )
        # Should return 400 or error in body, not crash
        assert resp.status_code in (400, 422, 200)

    @patch("app.api.ocr_routes.extract_all_fields")
    @patch("app.api.ocr_routes.get_ocr_engine")
    def test_valid_image_upload_succeeds(self, mock_engine, mock_extractor):
        """Mock OCR to avoid EasyOCR model loading in CI."""
        mock_extractor.return_value = {
            "name":           {"value": "ARJUN KUMAR SHARMA", "confidence": 0.95, "raw": ""},
            "passport_number":{"value": "A1234567",           "confidence": 0.92, "raw": ""},
            "date_of_birth":  {"value": "1990-05-15",         "confidence": 0.91, "raw": ""},
            "date_of_issue":  {"value": "2019-03-20",         "confidence": 0.90, "raw": ""},
            "date_of_expiry": {"value": "2029-03-19",         "confidence": 0.90, "raw": ""},
            "country_code":   {"value": "IND",                "confidence": 0.98, "raw": ""},
            "passport_type":  {"value": "P",                  "confidence": 0.99, "raw": ""},
            "sex":            {"value": "M",                  "confidence": 0.97, "raw": ""},
            "nationality":    {"value": "INDIAN",             "confidence": 0.96, "raw": ""},
            "place_of_birth": {"value": "NEW DELHI",          "confidence": 0.88, "raw": ""},
            "place_of_issue": {"value": "NEW DELHI",          "confidence": 0.87, "raw": ""},
            "mrz_line1":      {"value": "P<INDSHARMA<<ARJUN<KUMAR<<<<<<<<<<<<<<<<<<<<", "confidence": 0.91, "raw": ""},
            "mrz_line2":      {"value": "A12345674IND9005152M2903199<<<<<<<<<<<<<<<2", "confidence": 0.91, "raw": ""},
        }
        mock_engine_instance = MagicMock()
        mock_engine_instance.read.return_value = []
        mock_engine.return_value = mock_engine_instance

        img_bytes = _make_test_image()
        resp = client.post(
            "/api/v1/ocr",
            files={"file": ("test.png", img_bytes, "image/png")},
            data={"use_template": "true"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "fields" in data
        assert data["fields"]["passport_number"]["value"] == "A1234567"
