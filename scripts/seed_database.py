"""
Standalone database seeder script.

Run: python scripts/seed_database.py

Creates the SQLite database, creates tables, and inserts all
synthetic test records. Safe to run multiple times (idempotent).
"""
import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.database.seed import seed_database

if __name__ == "__main__":
    print("=" * 60)
    print("  AI Passport Screening System — Database Seeder")
    print("  Phase 1 Prototype — SYNTHETIC DATA ONLY")
    print("=" * 60)
    seed_database()
    print("=" * 60)
