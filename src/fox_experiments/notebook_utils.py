"""Presentation and export helpers; training logic does not live here."""

from __future__ import annotations

from pathlib import Path
import zipfile


def archive_results(destination, folders, include_checkpoints=False):
    """Archive reports with stable paths; keep large checkpoint files optional."""
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for label, directory in folders.items():
            directory = Path(directory)
            if not directory.is_dir():
                continue
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                if not include_checkpoints and path.suffix in (
                    ".pt",
                    ".pth",
                    ".safetensors",
                ):
                    continue
                if path.resolve() in (destination, temporary):
                    continue
                archive.write(path, str(Path(label) / path.relative_to(directory)))
    temporary.replace(destination)
    return destination
