"""NPZ array IO helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_npz_arrays(path: str | Path, **arrays: object) -> Path:
    """Save named arrays to a compressed NPZ archive."""
    if not arrays:
        raise ValueError("At least one named array is required.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """Load all arrays from an NPZ archive into memory."""
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
