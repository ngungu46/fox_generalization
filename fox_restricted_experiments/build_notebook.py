"""Build the readable library-importing Colab notebook."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "build_colab.py"), run_name="__main__")
