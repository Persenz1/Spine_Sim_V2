"""筛选阶段的分组统计工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Spine_Sim_V2.io.parquet_io import read_parquet, write_parquet


GROUP_KEYS = ["candidate_id", "surface_kind", "w_total_n"]


def grouped_statistics(summary: Any, *, stage_name: str) -> Any:
    """按候选、表面类别和预载分组聚合 case summary。"""
    pd = _require_pandas()
    summary = _with_derived_columns(summary)
    # 先按 candidate × surface_kind × w_total_n 聚合，后续评分再等权合并，
    # 防止某类表面或某档预载样本更多时压过其他条件。
    grouped = (
        summary.groupby(GROUP_KEYS, dropna=False)
        .agg(
            n_cases=("case_id", "count"),
            n_success=("load_success", "sum"),
            n_engagement_success=("engagement_success", "sum"),
            success_probability=("load_success", "mean"),
            success_rate=("load_success", "mean"),
            engagement_success_probability=("engagement_success", "mean"),
            **_distribution_aggs("f_t_lim_n"),
            **_distribution_aggs("f_t_lim_over_w_total"),
            **_distribution_aggs("n_eff_kish"),
            **_distribution_aggs("n_eng"),
            **_distribution_aggs("eta_max"),
            **_distribution_aggs("r_uncontacted"),
            **_distribution_aggs("r_fail_search"),
            **_distribution_aggs("r_sat_n"),
            **_distribution_aggs("r_sat_y"),
            **_distribution_aggs("r_side_contact_risk"),
            **_distribution_aggs("r_slip"),
            **_distribution_aggs("r_overload"),
            **_distribution_aggs("r_micro_damage_risk"),
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


def _with_derived_columns(summary: Any) -> Any:
    """补齐旧数据或派生数据列，使后处理对历史输出更稳健。"""
    pd = _require_pandas()
    frame = summary.copy()
    if "r_uncontacted" not in frame.columns and "r_con" in frame.columns:
        frame["r_uncontacted"] = 1.0 - frame["r_con"]
    defaults = {
        "engagement_success": False,
        "r_uncontacted": 0.0,
        "r_sat_y": 0.0,
        "r_slip": 0.0,
        "r_overload": 0.0,
        "r_micro_damage_risk": float("nan"),
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    return frame


def _distribution_aggs(column: str, prefix: str | None = None) -> dict[str, tuple[str, Any]]:
    """返回文档要求的 mean/median/std/p05/p25/p75/p95/min/max 命名聚合。"""
    name = prefix or column
    return {
        f"{name}_mean": (column, "mean"),
        f"{name}_median": (column, "median"),
        f"{name}_std": (column, "std"),
        f"{name}_p05": (column, lambda s: s.quantile(0.05)),
        f"{name}_p25": (column, lambda s: s.quantile(0.25)),
        f"{name}_p75": (column, lambda s: s.quantile(0.75)),
        f"{name}_p95": (column, lambda s: s.quantile(0.95)),
        f"{name}_min": (column, "min"),
        f"{name}_max": (column, "max"),
    }


def write_grouped_statistics_for_stage(stage_dir: str | Path) -> Any:
    """读取阶段 summary，写出分组统计表并返回结果。"""
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
