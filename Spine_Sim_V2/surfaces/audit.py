"""surface bank 审查辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Spine_Sim_V2.surfaces.bank import SurfaceBank


SURFACE_STATISTICS_REQUIRED_FIELDS: tuple[str, ...] = (
    "surface_bank_id",
    "surface_id",
    "surface_kind",
    "seed",
    "dx_mm",
    "dy_mm",
    "size_x_mm",
    "size_y_mm",
    "tip_radius_mm",
    "rq_raw_mm",
    "ra_raw_mm",
    "hpv_raw_mm",
    "rq_eff_mm",
    "ra_eff_mm",
    "hpv_eff_mm",
    "slope_mean_deg",
    "slope_p50_deg",
    "slope_p95_deg",
    "slope_max_deg",
    "candidate_density_preload_free",
    "valid",
    "reject_reason",
)


def audit_surface_bank(surface_bank: str | Path | SurfaceBank) -> dict[str, Any]:
    """校验 surface bank 完整性并返回审查信息。"""
    bank = surface_bank if isinstance(surface_bank, SurfaceBank) else SurfaceBank.open(surface_bank)
    stats = bank.load_statistics()
    missing_fields = sorted(set(SURFACE_STATISTICS_REQUIRED_FIELDS) - set(stats.columns))
    missing_files: list[str] = []
    for surface_id in stats["surface_id"].tolist():
        if not bank.surface_path(str(surface_id)).exists():
            missing_files.append(str(surface_id))

    image_files = sorted(
        str(path.relative_to(bank.root))
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.svg", "*.pdf")
        for path in bank.root.rglob(suffix)
    )
    valid = not missing_fields and not missing_files and not image_files
    return {
        "surface_bank_id": bank.bank_id,
        "root": str(bank.root),
        "n_surfaces": int(len(stats)),
        "surface_kinds": sorted(str(kind) for kind in stats["surface_kind"].unique()),
        "missing_fields": missing_fields,
        "missing_files": missing_files,
        "image_files_inside_bank": image_files,
        "valid": valid,
    }


def sample_surface_ids_by_kind(
    surface_bank: str | Path | SurfaceBank,
    *,
    sample_per_kind: int,
    seed: int = 20260616,
) -> dict[str, list[str]]:
    """按表面类别返回可复现的 ``surface_id`` 抽样结果。"""
    if sample_per_kind <= 0:
        raise ValueError("sample_per_kind must be positive.")
    bank = surface_bank if isinstance(surface_bank, SurfaceBank) else SurfaceBank.open(surface_bank)
    stats = bank.load_statistics()
    samples: dict[str, list[str]] = {}
    for surface_kind, group in stats.groupby("surface_kind", sort=True):
        sampled = group.sample(n=min(sample_per_kind, len(group)), random_state=seed)
        samples[str(surface_kind)] = [str(value) for value in sampled["surface_id"].tolist()]
    return samples
