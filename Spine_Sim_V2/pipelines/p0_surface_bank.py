"""P0: surface bank generation and audit pipeline."""

from __future__ import annotations

from pathlib import Path

from Spine_Sim_V2.surfaces.bank import SurfaceBank, create_surface_bank


def run(
    *,
    bank_id: str,
    surfaces: str,
    n_per_kind: int,
    resolution: int,
    size_x_mm: float,
    size_y_mm: float,
    tip_radius_mm: float,
    outdir: str | Path,
    base_seed: int = 20260616,
    overwrite: bool = False,
) -> SurfaceBank:
    """Generate the Phase 0/2 surface bank data products."""
    return create_surface_bank(
        bank_id=bank_id,
        surface_kinds=surfaces,
        n_per_kind=n_per_kind,
        resolution_cells_per_mm=resolution,
        size_x_mm=size_x_mm,
        size_y_mm=size_y_mm,
        tip_radius_mm=tip_radius_mm,
        outdir=outdir,
        base_seed=base_seed,
        overwrite=overwrite,
    )
