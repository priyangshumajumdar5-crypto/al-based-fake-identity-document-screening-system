"""
Pytest shared fixtures and test configuration.
Ensures the project root is on sys.path for all tests.
"""
import sys
from pathlib import Path

# Add project root to path so `app.*` imports work
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
