"""Grouped statistics for screening stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Spine_Sim_V2.io.parquet_io import read_parquet, write_parquet


GROUP_KEYS = ["candidate_id", "surface_kind", "w_total_n"]


def grouped_statistics(summary: Any, *, stage_name: str) -> Any:
    """Aggregate case summary by candidate, surface kind, and preload."""
    pd = _require_pandas()
    grouped = (
        summary.groupby(GROUP_KEYS, dropna=False)
        .agg(
            n_cases=("case_id", "count"),
            n_success=("load_success", "sum"),
            success_probability=("load_success", "mean"),
            success_rate=("load_success", "mean"),
            f_t_lim_n_mean=("f_t_lim_n", "mean"),
            f_t_lim_n_median=("f_t_lim_n", "median"),
            f_t_lim_n_p05=("f_t_lim_n", lambda s: s.quantile(0.05)),
            f_t_lim_n_p95=("f_t_lim_n", lambda s: s.quantile(0.95)),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total", "mean"),
            n_eff_kish_mean=("n_eff_kish", "mean"),
            n_eng_mean=("n_eng", "mean"),
            eta_max_mean=("eta_max", "mean"),
            r_fail_search_mean=("r_fail_search", "mean"),
            r_sat_n_mean=("r_sat_n", "mean"),
            r_sat_y_mean=("r_sat_y", "mean"),
            r_side_contact_risk_mean=("r_side_contact_risk", "mean"),
            cascade_failure_rate=("cascade_failure", "mean"),
            normal_range_insufficient_rate=("normal_range_insufficient", "mean"),
            alpha_p_deg=("alpha_p_deg", "first"),
            spring_k_n_per_m=("spring_k_n_per_m", "first"),
            rows=("rows", "first"),
            cols=("cols", "first"),
            pitch_t_mm=("pitch_t_mm", "first"),
            pitch_l_mm=("pitch_l_mm", "first"),
            array_type=("array_type", "first"),
        )
        .reset_index()
    )
    grouped.insert(0, "stage", stage_name)
    grouped.insert(
        1,
        "group_id",
        grouped.apply(
            lambda row: f"{row['candidate_id']}__{row['surface_kind']}__W{row['w_total_n']}",
            axis=1,
        ),
    )
    grouped.insert(2, "group_by", [GROUP_KEYS] * len(grouped))
    return grouped


def write_grouped_statistics_for_stage(stage_dir: str | Path) -> Any:
    """Read a stage summary, write grouped statistics, and return them."""
    stage_path = Path(stage_dir)
    summary = read_parquet(stage_path / "data" / "stage_summary.parquet")
    stage_name = _stage_name_from_dir(stage_path)
    grouped = grouped_statistics(summary, stage_name=stage_name)
    write_parquet(grouped, stage_path / "data" / "stage_grouped_statistics.parquet")
    return grouped


def _stage_name_from_dir(stage_path: Path) -> str:
    name = stage_path.name
    if name.startswith("P2"):
        return "p2_compliant_k_alpha"
    if name.startswith("P3"):
        return "p3_rigid_alpha"
    return name


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Grouped statistics require pandas.") from exc
    return pd
