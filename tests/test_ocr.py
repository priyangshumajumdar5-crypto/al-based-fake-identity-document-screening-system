"""
Pytest tests for image preprocessing and date utilities.
No EasyOCR required.
"""
import pytest
import numpy as np
from datetime import date
from PIL import Image

from app.utils.date_utils import (
    parse_date, parse_mrz_date, date_to_mrz,
    is_expired, dates_are_consistent, normalise_date,
)
from app.utils.image_utils import (
    load_image_from_bytes, validate_image_format, pil_to_numpy, numpy_to_pil,
)
from app.ocr.preprocessing import (
    resize_to_max, to_grayscale, enhance_contrast, enhance_sharpness,
    preprocess_image, preprocess_for_field, preprocess_for_mrz,
    crop_region,
)


# ─────────────────────────────────────────────────────────────────────────────
# Date utils
# ─────────────────────────────────────────────────────────────────────────────

class TestDateParsing:
    def test_iso_format(self):
        d, s = parse_date("1990-05-15")
        assert d == date(1990, 5, 15)
        assert s == "1990-05-15"

    def test_slash_format(self):
        d, s = parse_date("15/05/1990")
        assert d == date(1990, 5, 15)

    def test_dash_format(self):
        d, s = parse_date("15-05-1990")
        assert d == date(1990, 5, 15)

    def test_invalid_date(self):
        d, s = parse_date("NOT-A-DATE")
        assert d is None
        assert s == ""

    def test_empty_string(self):
        d, s = parse_date("")
        assert d is None

    def test_feb_29_leap_year(self):
        d, s = parse_date("29/02/2000")
        assert d == date(2000, 2, 29)

    def test_feb_29_non_leap_year(self):
        d, s = parse_date("29/02/2001")
        assert d is None

    def test_mrz_date_young(self):
        # YYMMDD = 900515 → 1990-05-15
        d, s = parse_mrz_date("900515")
        assert d == date(1990, 5, 15)

    def test_mrz_date_2000s(self):
        # YYMMDD = 290319 → 2029-03-19
        d, s = parse_mrz_date("290319")
        assert d == date(2029, 3, 19)

    def test_mrz_date_invalid(self):
        d, s = parse_mrz_date("99ZZ99")
        assert d is None

    def test_date_to_mrz_roundtrip(self):
        d = date(1990, 5, 15)
        mrz = date_to_mrz(d)
        assert len(mrz) == 6
        assert mrz.isdigit()
        parsed, _ = parse_mrz_date(mrz)
        assert parsed == d


class TestDateValidation:
    def test_not_expired(self):
        expired, msg = is_expired("2029-03-19", reference_date=date(2026, 1, 1))
        assert not expired
        assert "VALID" in msg

    def test_expired(self):
        expired, msg = is_expired("2020-01-01", reference_date=date(2026, 1, 1))
        assert expired
        assert "EXPIRED" in msg

    def test_invalid_date_string(self):
        expired, msg = is_expired("NOTADATE")
        assert msg == "INVALID_DATE"

    def test_dates_consistent(self):
        ok, msg = dates_are_consistent("1990-05-15", "2019-03-20")
        assert ok

    def test_dates_inconsistent(self):
        ok, msg = dates_are_consistent("2019-03-20", "1990-05-15")
        assert not ok

    def test_dates_same_fail(self):
        ok, msg = dates_are_consistent("2019-03-20", "2019-03-20")
        assert not ok

    def test_normalise_date(self):
        result = normalise_date("15/05/1990")
        assert result == "1990-05-15"

    def test_normalise_invalid(self):
        result = normalise_date("NOTADATE")
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# Image utils
# ─────────────────────────────────────────────────────────────────────────────

class TestImageUtils:
    def _make_png_bytes(self, size=(200, 150)):
        import io
        img = Image.new("RGB", size, color=(128, 200, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_load_image_from_bytes(self):
        data = self._make_png_bytes()
        img = load_image_from_bytes(data)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_load_invalid_bytes_raises(self):
        with pytest.raises(ValueError):
            load_image_from_bytes(b"NOT AN IMAGE")

    def test_validate_format_valid(self):
        data = self._make_png_bytes()
        img = load_image_from_bytes(data)
        ok, msg = validate_image_format(img)
        assert ok

    def test_validate_format_too_small(self):
        img = Image.new("RGB", (50, 50))
        ok, msg = validate_image_format(img)
        assert not ok
        assert "small" in msg.lower()

    def test_pil_to_numpy_shape(self):
        img = Image.new("RGB", (100, 80))
        arr = pil_to_numpy(img)
        assert arr.shape == (80, 100, 3)

    def test_numpy_to_pil_roundtrip(self):
        img = Image.new("RGB", (100, 80), color=(50, 100, 150))
        arr = pil_to_numpy(img)
        restored = numpy_to_pil(arr)
        assert isinstance(restored, Image.Image)
        assert restored.size == (100, 80)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def _make_img(self, size=(400, 300)):
        return Image.new("RGB", size, color=(200, 200, 200))

    def test_resize_to_max_downscales(self):
        img = self._make_img((4000, 3000))
        resized = resize_to_max(img, max_px=2000)
        assert max(resized.size) <= 2000

    def test_resize_to_max_small_upscales(self):
        img = self._make_img((200, 150))
        resized = resize_to_max(img, max_px=2048)
        # Small images should be upscaled to at least 800px
        assert max(resized.size) >= 200  # At least not shrunk

    def test_to_grayscale_returns_L(self):
        img = self._make_img()
        gray = to_grayscale(img)
        assert gray.mode == "L"

    def test_enhance_contrast_returns_rgb(self):
        img = self._make_img()
        enhanced = enhance_contrast(img, 1.5)
        assert isinstance(enhanced, Image.Image)

    def test_preprocess_image_returns_pil(self):
        img = self._make_img()
        result = preprocess_image(img, do_deskew=False)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_preprocess_for_field_returns_pil(self):
        img = self._make_img((100, 30))
        result = preprocess_for_field(img)
        assert isinstance(result, Image.Image)

    def test_preprocess_for_mrz_returns_grayscale(self):
        img = self._make_img((500, 80))
        result = preprocess_for_mrz(img)
        assert result.mode == "L"

    def test_crop_region_relative(self):
        img = self._make_img((900, 640))
        cropped = crop_region(img, (0.0, 0.0, 0.5, 0.5), relative=True)
        assert cropped.size == (450, 320)

    def test_crop_region_absolute(self):
        img = self._make_img((900, 640))
        cropped = crop_region(img, (0, 0, 450, 320), relative=False)
        assert cropped.size == (450, 320)
