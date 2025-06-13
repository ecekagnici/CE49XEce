# tests/conftest.py

import sys
from pathlib import Path

# Locate project root and add src/ to Python path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root / "src"))
