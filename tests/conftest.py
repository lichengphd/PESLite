"""Import the package from ``src/`` without installing it."""

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "src" / "peslite" / "configs"
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield
