"""Run fast integrity checks without retraining the models."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "tests" / "test_integrity.py"), run_name="__main__")
