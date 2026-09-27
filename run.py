"""
Convenience launcher for FastAPI and Streamlit.

Usage:
    python run.py api        → start FastAPI on http://127.0.0.1:8000
    python run.py ui         → start Streamlit dashboard
    python run.py seed       → seed the SQLite database
    python run.py generate   → generate synthetic passport images
    python run.py all        → seed + start API (use two terminals for UI)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def run_api():
    print("\n🚀 Starting FastAPI server → http://127.0.0.1:8000")
    print("   API Docs: http://127.0.0.1:8000/docs\n")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--reload",
         "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(ROOT),
    )


def run_ui():
    print("\n🖥️  Starting Streamlit Dashboard")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "frontend/streamlit_app.py"],
        cwd=str(ROOT),
    )


def run_seed():
    print("\n🗄️  Seeding database with synthetic test records...")
    subprocess.run(
        [sys.executable, "scripts/seed_database.py"],
        cwd=str(ROOT),
    )


def run_generate():
    print("\n🖼️  Generating synthetic passport images...")
    subprocess.run(
        [sys.executable, "data/synthetic/generate_test_image.py"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "api":
        run_api()
    elif cmd == "ui":
        run_ui()
    elif cmd == "seed":
        run_seed()
    elif cmd == "generate":
        run_generate()
    elif cmd == "all":
        run_seed()
        run_generate()
        run_api()
    else:
        print(__doc__)
