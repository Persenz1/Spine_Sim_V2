"""筛选阶段的分组统计工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Spine_Sim_V2.io.parquet_io import parquet_columns, read_parquet, write_parquet


GROUP_KEYS = ["candidate_id", "surface_kind", "w_total_n"]
DISTRIBUTION_COLUMNS = (
    "f_t_lim_n",
    "f_t_lim_over_w_total",
    "n_eff_kish",
    "n_eng",
    "eta_max",
    "r_uncontacted",
    "r_fail_search",
    "r_sat_n",
    "r_sat_y",
    "r_side_contact_risk",
    "r_slip",
    "r_overload",
    "r_micro_damage_risk",
)
SUMMARY_ANALYSIS_COLUMNS = tuple(
    dict.fromkeys(
        [
            *GROUP_KEYS,
            "case_id",
            "load_success",
            "engagement_success",
            *DISTRIBUTION_COLUMNS,
            "r_con",
            "cascade_failure",
            "normal_range_insufficient",
            "alpha_p_deg",
            "spring_k_n_per_m",
            "rows",
            "cols",
            "pitch_t_mm",
            "pitch_l_mm",
            "array_type",
        ]
    )
)
QUANTILES = (0.05, 0.25, 0.75, 0.95)
QUANTILE_SUFFIX = {0.05: "p05", 0.25: "p25", 0.75: "p75", 0.95: "p95"}


def grouped_statistics(summary: Any, *, stage_name: str) -> Any:
    """按候选、表面类别和预载分组聚合 case summary。"""
    pd = _require_pandas()
    summary = _with_derived_columns(summary)
    # 先按 candidate × surface_kind × w_total_n 聚合，后续评分再等权合并，
    # 防止某类表面或某档预载样本更多时压过其他条件。
    grouped = aggregate_summary_statistics(
        summary,
        group_keys=GROUP_KEYS,
        first_columns=(
            "alpha_p_deg",
            "spring_k_n_per_m",
            "rows",
            "cols",
            "pitch_t_mm",
            "pitch_l_mm",
            "array_type",
        ),
        include_success_rate=True,
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


def read_summary_for_statistics(path: str | Path) -> Any:
    """只读取统计阶段必要的 summary 列，降低 pandas 内存峰值。"""
    available = set(parquet_columns(path))
    selected = [column for column in SUMMARY_ANALYSIS_COLUMNS if column in available]
    return read_parquet(path, columns=selected)


def aggregate_summary_statistics(
    summary: Any,
    *,
    group_keys: list[str] | tuple[str, ...],
    first_columns: list[str] | tuple[str, ...] = (),
    include_success_rate: bool = False,
) -> Any:
    """按指定 group keys 聚合 summary 统计量。

    分位数使用一次 DataFrameGroupBy.quantile 批量计算，避免在
    ``groupby.agg`` 中为每个列、每个分位数创建 lambda，在 P5/P6 大表上更省时间和内存。
    """
    frame = _with_derived_columns(summary)
    for column in _required_stat_columns(first_columns):
        if column not in frame.columns:
            frame[column] = _default_for_column(column)

    dist_columns = [column for column in DISTRIBUTION_COLUMNS if column in frame.columns]
    grouped_obj = frame.groupby(list(group_keys), dropna=False)
    aggs: dict[str, tuple[str, Any]] = {
        "n_cases": ("case_id", "count"),
        "n_success": ("load_success", "sum"),
        "n_engagement_success": ("engagement_success", "sum"),
        "success_probability": ("load_success", "mean"),
        "engagement_success_probability": ("engagement_success", "mean"),
        "cascade_failure_rate": ("cascade_failure", "mean"),
        "normal_range_insufficient_rate": ("normal_range_insufficient", "mean"),
    }
    if include_success_rate:
        aggs["success_rate"] = ("load_success", "mean")
    for column in dist_columns:
        aggs.update(_distribution_non_quantile_aggs(column))
    for column in first_columns:
        aggs[column] = (column, "first")

    base = grouped_obj.agg(**aggs).reset_index()
    quantiles = _grouped_quantiles(frame, group_keys=group_keys, columns=dist_columns)
    if not quantiles.empty:
        base = base.merge(quantiles, on=list(group_keys), how="left")
    return base


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


def _required_stat_columns(first_columns: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            [
                "case_id",
                "load_success",
                "engagement_success",
                "cascade_failure",
                "normal_range_insufficient",
                *DISTRIBUTION_COLUMNS,
                *first_columns,
            ]
        )
    )


def _default_for_column(column: str) -> Any:
    if column in {"case_id", "array_type"}:
        return None
    if column in {"load_success", "engagement_success", "cascade_failure", "normal_range_insufficient"}:
        return False
    return float("nan")


def _distribution_non_quantile_aggs(column: str, prefix: str | None = None) -> dict[str, tuple[str, Any]]:
    name = prefix or column
    return {
        f"{name}_mean": (column, "mean"),
        f"{name}_median": (column, "median"),
        f"{name}_std": (column, "std"),
        f"{name}_min": (column, "min"),
        f"{name}_max": (column, "max"),
    }


def _grouped_quantiles(
    summary: Any,
    *,
    group_keys: list[str] | tuple[str, ...],
    columns: list[str],
) -> Any:
    pd = _require_pandas()
    if not columns:
        return pd.DataFrame(columns=list(group_keys))
    quantiles = summary.groupby(list(group_keys), dropna=False)[columns].quantile(list(QUANTILES))
    if quantiles.empty:
        return pd.DataFrame(columns=list(group_keys))
    table = quantiles.unstack(level=-1)
    table.columns = [
        f"{column}_{QUANTILE_SUFFIX.get(float(q), f'q{int(float(q) * 100):02d}')}"
        for column, q in table.columns
    ]
    return table.reset_index()


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
    summary = read_summary_for_statistics(stage_path / "data" / "stage_summary.parquet")
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
