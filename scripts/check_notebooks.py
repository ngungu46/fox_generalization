"""Validate notebook schema and enforce short, inspectable code cells."""

import ast
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
for path in sorted((root / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    for index, cell in enumerate(code_cells):
        ast.parse(cell.source, filename=f"{path.name}:code-{index}")
        assert len(cell.source.splitlines()) <= 30, f"Long cell in {path.name}"
        assert (
            "write_text(" not in cell.source
        ), "Move source/config writing out of notebooks"
        assert (
            "exec(" not in cell.source
        ), "Notebook must not execute hidden source strings"
        assert not cell.outputs, "Commit clean notebooks without execution outputs"
    print(f"{path.name}: {len(code_cells)} short code cells, valid schema and Python")
