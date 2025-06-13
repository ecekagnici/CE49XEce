# src/utils.py

"""
Common utility functions (currently unused directly by tests,
but available for future extensions).
"""

import json
from pathlib import Path


def load_json(path):
    """Load JSON from file and return as dict."""
    with Path(path).open('r') as f:
        return json.load(f)


def file_extension(path):
    """Return the lowercase file extension, e.g. '.csv' or '.json'."""
    return Path(path).suffix.lower()
