"""
Synthetic Passport Image Generator — Phase 1 Prototype

Generates clearly synthetic, fictional passport-style images for testing.

IMPORTANT:
  - All data is FICTIONAL — no real people, no real passport numbers.
  - Images are clearly labelled "SYNTHETIC TEST DOCUMENT".
  - No real faces are used.
  - These images are for prototype/demo purposes ONLY.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
# Passport data (matches synthetic DB records)
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_PASSPORTS = [
    {
        "passport_number": "A1234567",
        "passport_type": "P",
        "country_code": "IND",
        "surname": "SHARMA",
        "given_names": "ARJUN KUMAR",
        "nationality": "INDIAN",
        "date_of_birth": "15/05/1990",
        "place_of_birth": "NEW DELHI",
        "sex": "M",
        "date_of_issue": "20/03/2019",
        "date_of_expiry": "19/03/2029",
        "place_of_issue": "NEW DELHI",
        # MRZ line 1: P<IND + surname<<given names (padded to 44)
        "mrz_line1": "P<INDSHARMA<<ARJUN<KUMAR<<<<<<<<<<<<<<<<<<<<",
        # MRZ line 2: passnumber+cd+IND+YYMMDD+cd+sex+YYMMDD+cd+opt+cd+composite
        "mrz_line2": "A12345674IND9005152M2903199<<<<<<<<<<<<<<<2",
        "status": "VALID",
    },
    {
        "passport_number": "F6789012",
        "passport_type": "P",
        "country_code": "IND",
        "surname": "CHATTERJEE",
        "given_names": "ANANYA ROY",
        "nationality": "INDIAN",
        "date_of_birth": "04/07/1980",
        "place_of_birth": "KOLKATA",
        "sex": "F",
        "date_of_issue": "28/02/2013",
        "date_of_expiry": "27/02/2023",
        "place_of_issue": "KOLKATA",
        "mrz_line1": "P<INDCHATTERJEE<<ANANYA<ROY<<<<<<<<<<<<<<<<<",
        "mrz_line2": "F67890120IND8007044F2302279<<<<<<<<<<<<<<<4",
        "status": "EXPIRED",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Image rendering
# ─────────────────────────────────────────────────────────────────────────────

WIDTH  = 900
HEIGHT = 640

# Colours
BG_COLOUR   = (245, 242, 235)   # Off-white parchment
NAVY        = (10,  40,  80)
GOLD        = (180, 140, 30)
DARK_GREY   = (40,  40,  50)
MRZ_BG      = (230, 230, 230)
RED_WARN    = (180, 20, 20)


def _load_font(size: int, bold: bool = False):
    """Try to load a system font; fall back to PIL default."""
    candidates = [
        "arial.ttf", "Arial.ttf",
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf",
        "FreeSans.ttf", "LiberationSans-Regular.ttf",
        "Helvetica.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _draw_passport(data: dict) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOUR)
    draw = ImageDraw.Draw(img)

    # ── Background pattern ────────────────────────────────────────────────────
    # Subtle grid
    for x in range(0, WIDTH, 30):
        draw.line([(x, 0), (x, HEIGHT)], fill=(235, 232, 220), width=1)
    for y in range(0, HEIGHT, 30):
        draw.line([(0, y), (WIDTH, y)], fill=(235, 232, 220), width=1)

    # ── Top bar ───────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (WIDTH, 52)], fill=NAVY)
    draw.rectangle([(0, 48), (WIDTH, 54)], fill=GOLD)

    f_title = _load_font(14, bold=True)
    draw.text((WIDTH // 2, 16), "REPUBLIC OF INDIA", fill=(255, 220, 120),
              font=f_title, anchor="mm")
    draw.text((WIDTH // 2, 35), "PASSPORT — SYNTHETIC TEST DOCUMENT", fill=(180, 180, 180),
              font=_load_font(10), anchor="mm")

    # ── Passport type / country ───────────────────────────────────────────────
    f_field = _load_font(11)
    f_value = _load_font(14)
    f_label = _load_font(9)

    def label_value(x, y, lbl, val, col=DARK_GREY):
        draw.text((x, y), lbl, fill=(100, 100, 110), font=f_label)
        draw.text((x, y + 14), val, fill=col, font=f_value)

    label_value(30, 70, "TYPE", data["passport_type"])
    label_value(120, 70, "COUNTRY CODE", data["country_code"])
    label_value(550, 70, "PASSPORT NO.", data["passport_number"], col=NAVY)

    # ── Divider ───────────────────────────────────────────────────────────────
    draw.line([(25, 105), (WIDTH - 25, 105)], fill=GOLD, width=1)

    # ── Photo placeholder ─────────────────────────────────────────────────────
    photo_box = (30, 115, 200, 295)
    draw.rectangle(photo_box, fill=(200, 200, 210), outline=NAVY, width=2)
    draw.text(
        (115, 205), "SYNTHETIC\nPHOTO\nPLACEHOLDER",
        fill=(100, 100, 130), font=_load_font(9), anchor="mm",
    )

    # ── Field section ─────────────────────────────────────────────────────────
    lx = 220  # left column x

    label_value(lx, 120, "SURNAME", data["surname"])
    label_value(lx, 160, "GIVEN NAMES", data["given_names"])
    label_value(lx, 200, "NATIONALITY", data["nationality"])
    label_value(lx, 240, "DATE OF BIRTH", data["date_of_birth"])
    label_value(lx + 260, 240, "SEX", data["sex"])
    label_value(lx, 280, "PLACE OF BIRTH", data["place_of_birth"])

    label_value(30, 320, "DATE OF ISSUE", data["date_of_issue"])
    label_value(280, 320, "DATE OF EXPIRY", data["date_of_expiry"])
    label_value(30, 360, "PLACE OF ISSUE", data["place_of_issue"])

    # ── Expiry colour indicator ────────────────────────────────────────────────
    if data.get("status") == "EXPIRED":
        draw.text((700, 320), "⚠ EXPIRED", fill=RED_WARN, font=_load_font(13))

    # ── Divider ───────────────────────────────────────────────────────────────
    draw.line([(25, 405), (WIDTH - 25, 405)], fill=GOLD, width=1)

    # ── Synthetic warning ─────────────────────────────────────────────────────
    draw.text(
        (WIDTH // 2, 420),
        "⚠ SYNTHETIC TEST DOCUMENT — NOT A REAL PASSPORT ⚠",
        fill=RED_WARN, font=_load_font(10), anchor="mm",
    )

    # ── MRZ section ───────────────────────────────────────────────────────────
    mrz_y = 440
    draw.rectangle([(0, mrz_y), (WIDTH, HEIGHT)], fill=MRZ_BG)
    draw.line([(0, mrz_y), (WIDTH, mrz_y)], fill=NAVY, width=2)

    draw.text((30, mrz_y + 8), "MACHINE READABLE ZONE (SYNTHETIC)", fill=(80, 80, 100), font=_load_font(9))

    f_mrz = _load_font(16)
    draw.text((30, mrz_y + 30), data["mrz_line1"], fill=(10, 10, 30), font=f_mrz)
    draw.text((30, mrz_y + 60), data["mrz_line2"], fill=(10, 10, 30), font=f_mrz)

    # ── Border ────────────────────────────────────────────────────────────────
    draw.rectangle([(0, 0), (WIDTH - 1, HEIGHT - 1)], outline=NAVY, width=3)

    return img


def generate_all(output_dir: Path) -> None:
    """Generate all synthetic passport images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for data in SYNTHETIC_PASSPORTS:
        img = _draw_passport(data)
        fname = f"synthetic_passport_{data['passport_number']}.png"
        fpath = output_dir / fname
        img.save(str(fpath), "PNG")
        print(f"  Generated: {fpath.name}")


if __name__ == "__main__":
    out = Path(__file__).parent
    print("\nGenerating synthetic passport images...")
    generate_all(out)
    print("\n✅ Done. Images saved to:", out)
