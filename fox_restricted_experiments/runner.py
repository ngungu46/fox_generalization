"""Compatibility entry point; implementation lives in fox_restricted.legacy.runner."""
from pathlib import Path
import importlib
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "library"))
if __name__ == "__main__":
    runpy.run_module("fox_restricted.legacy.runner", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("fox_restricted.legacy.runner")
