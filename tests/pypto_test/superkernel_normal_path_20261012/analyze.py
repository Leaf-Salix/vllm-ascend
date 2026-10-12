"""Recompute the completed retry; preserve the failed first attempt separately."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "retry1/analyze.py"), run_name="__main__")
