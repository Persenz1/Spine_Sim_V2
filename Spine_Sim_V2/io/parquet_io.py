"""Parquet and preview CSV IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_parquet(df: Any, path: str | Path) -> Path:
    """Write a dataframe to Parquet using pyarrow."""
    _require_parquet_dependencies()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def read_parquet(path: str | Path) -> Any:
    """Read a Parquet file into a pandas dataframe."""
    pd = _require_parquet_dependencies()
    return pd.read_parquet(Path(path), engine="pyarrow")


def write_preview_csv(df: Any, path: str | Path, max_rows: int = 5000) -> Path:
    """Write a small CSV preview of a dataframe.

    CSV is a human preview format only. Parquet remains the authoritative table
    format for simulation data products.
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = df.head(max_rows) if hasattr(df, "head") else df
    preview.to_csv(output_path, index=False)
    return output_path


def _require_parquet_dependencies() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet IO requires pandas and pyarrow. Install project "
            "dependencies with `python3 -m pip install -e .`."
        ) from exc

    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet IO requires pyarrow. Install project dependencies with "
            "`python3 -m pip install -e .`."
        ) from exc

    return pd
