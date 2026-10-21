"""Pytest configuration and environment fixtures for ml_service tests."""

from pathlib import Path
import sys

# Ensure both repository root and ml_service directory are discoverable in sys.path
_current_dir = Path(__file__).resolve().parent
_ml_service_dir = _current_dir.parent
_repo_dir = _ml_service_dir.parent

for p in (_repo_dir, _ml_service_dir):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
