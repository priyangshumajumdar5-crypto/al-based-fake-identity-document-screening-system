"""
Build a draft passport database CSV from actual passport images in the passport/ directory
using app.ocr.extract_passport_data.
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ocr import extract_passport_data


def process_passport_images(
    passport_dir: str = "passport",
    output_csv: str = "data/draft_passport_database.csv",
) -> List[Dict[str, Any]]:
    """
    Scan all passport images, extract OCR data, and write to a CSV file.

    Args:
        passport_dir: Directory containing passport image files.
        output_csv: Output path for the draft CSV.

    Returns:
        List of extracted row dicts.
    """
    dir_path = Path(passport_dir)
    if not dir_path.exists():
        print(f"Error: Directory '{passport_dir}' not found.")
        return []

    # Get sorted image files
    image_files = sorted(
        [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        ],
        key=lambda p: (
            int(re.search(r"\d+", p.stem).group()) if re.search(r"\d+", p.stem) else 0,
            p.name,
        ),
    )

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_filename",
        "passport_number",
        "name",
        "date_of_birth",
        "place_of_birth",
        "place_of_issue",
        "date_of_issue",
        "date_of_expiry",
        "gender",
        "status",
    ]

    rows: List[Dict[str, Any]] = []

    print(f"Found {len(image_files)} passport image(s) in '{passport_dir}'. Starting OCR processing...\n")

    for img_path in image_files:
        filename = img_path.name
        print(f"Processing {filename}...")
        try:
            res = extract_passport_data(str(img_path))
            surname = res.get("surname", "").strip()
            given_names = res.get("given_names", "").strip()
            combined_name = f"{surname} {given_names}".strip()

            row = {
                "image_filename": filename,
                "passport_number": res.get("passport_number", "").strip(),
                "name": combined_name,
                "date_of_birth": res.get("date_of_birth", "").strip(),
                "place_of_birth": res.get("place_of_birth", "").strip(),
                "place_of_issue": res.get("place_of_issue", "").strip(),
                "date_of_issue": res.get("date_of_issue", "").strip(),
                "date_of_expiry": res.get("date_of_expiry", "").strip(),
                "gender": res.get("sex", "").strip(),
                "status": "",
            }
            rows.append(row)
        except Exception as exc:
            print(f"  Warning: Failed to process {filename}: {exc}")
            rows.append({
                "image_filename": filename,
                "passport_number": "",
                "name": "",
                "date_of_birth": "",
                "place_of_birth": "",
                "place_of_issue": "",
                "date_of_issue": "",
                "date_of_expiry": "",
                "gender": "",
                "status": "",
            })

    # Write to CSV
    with open(out_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} record(s) to '{output_csv}'.")
    return rows


if __name__ == "__main__":
    process_passport_images()
