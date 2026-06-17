"""P6 最终 Monte Carlo 统计及 P7/P8 后处理。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Spine_Sim_V2.analysis.scoring import score_high, score_low
from Spine_Sim_V2.analysis.statistics import _distribution_aggs, _with_derived_columns
from Spine_Sim_V2.core.types import SchemaField, stage_spines_schema, stage_summary_schema
from Spine_Sim_V2.io.manifest import create_manifest, write_manifest
from Spine_Sim_V2.io.parquet_io import read_parquet, write_parquet
from Spine_Sim_V2.io.schema_io import dataframe_schema, write_schema


FINAL_GROUP_KEYS = [
    "candidate_id",
    "array_type",
    "surface_kind",
    "w_total_n",
    "alpha_p_deg",
    "spring_k_n_per_m",
    "rows",
    "cols",
    "pitch_t_mm",
    "pitch_l_mm",
]


def write_final_analysis(stage_dir: str | Path) -> tuple[Any, Any, Any]:
    """写出 P6 分组统计、最终排名、收敛统计和候选清单。"""
    stage_path = Path(stage_dir)
    data_dir = stage_path / "data"
    summary_path = data_dir / "final_summary.parquet"
    spines_path = data_dir / "final_spines.parquet"
    _require_file(summary_path, "final summary")
    _require_file(spines_path, "final spines")
    summary = read_parquet(summary_path)
    grouped = final_grouped_statistics(summary)
    rankings = rank_final_candidates(grouped)
    convergence = convergence_statistics(summary)
    write_parquet(grouped, data_dir / "final_grouped_statistics.parquet")
    write_parquet(rankings, data_dir / "final_rankings.parquet")
    write_parquet(convergence, data_dir / "convergence_statistics.parquet")
    _write_final_candidates_json(rankings, data_dir / "final_candidates.json")
    return grouped, rankings, convergence


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} data product: {path}")


def p6_schema_collection(grouped: Any, rankings: Any, convergence: Any) -> dict[str, tuple[SchemaField, ...]]:
    """返回 P6 所有 final 数据产品的 schema 元数据。"""
    return {
        "final_summary": stage_summary_schema,
        "final_spines": stage_spines_schema,
        "final_grouped_statistics": dataframe_schema(grouped),
        "final_rankings": dataframe_schema(rankings),
        "convergence_statistics": dataframe_schema(convergence),
    }


def final_grouped_statistics(summary: Any) -> Any:
    """按 P6 正式 Monte Carlo 维度聚合 summary。"""
    summary = _with_derived_columns(summary)
    # P6 保留 candidate × surface_kind × w_total_n 等键，便于后续表面泛化和预载效率分析。
    grouped = (
        summary.groupby(FINAL_GROUP_KEYS, dropna=False)
        .agg(
            n_cases=("case_id", "count"),
            n_success=("load_success", "sum"),
            n_engagement_success=("engagement_success", "sum"),
            success_probability=("load_success", "mean"),
            engagement_success_probability=("engagement_success", "mean"),
            **_distribution_aggs("f_t_lim_n"),
            **_distribution_aggs("f_t_lim_over_w_total"),
            **_distribution_aggs("eta_max"),
            **_distribution_aggs("n_eff_kish"),
            **_distribution_aggs("n_eng"),
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
        )
        .reset_index()
    )
    grouped.insert(0, "stage", "p6_final_3d_monte_carlo")
    grouped.insert(
        1,
        "group_id",
        grouped.apply(
            lambda row: "__".join(
                [
                    str(row["candidate_id"]),
                    str(row["surface_kind"]),
                    f"W{row['w_total_n']}",
                ]
            ),
            axis=1,
        ),
    )
    grouped.insert(2, "group_by", [FINAL_GROUP_KEYS] * len(grouped))
    return grouped


def rank_final_candidates(grouped: Any) -> Any:
    """基于 P6 分组统计对最终候选排序。"""
    scored = grouped.copy()
    scored["success_probability_score_row"] = score_high(scored["success_probability"])
    scored["force_score_row"] = score_high(scored["f_t_lim_n_mean"])
    scored["efficiency_score_row"] = score_high(scored["f_t_lim_over_w_total_mean"])
    scored["n_eff_kish_score_row"] = score_high(scored["n_eff_kish_mean"])
    scored["low_eta_score_row"] = score_low(scored["eta_max_mean"])
    scored["low_search_failure_score_row"] = score_low(scored["r_fail_search_mean"])
    scored["low_saturation_score_row"] = score_low(
        0.5 * scored["r_sat_n_mean"].fillna(0.0) + 0.5 * scored["r_sat_y_mean"].fillna(0.0)
    )
    failure_value = (
        scored["r_side_contact_risk_mean"].fillna(0.0)
        + scored["cascade_failure_rate"].fillna(0.0)
        + scored["normal_range_insufficient_rate"].fillna(0.0)
    ) / 3.0
    scored["low_failure_score_row"] = score_low(failure_value)
    robustness = (
        scored.groupby(["candidate_id", "surface_kind"], dropna=False)["success_probability"]
        .mean()
        .reset_index()
        .groupby("candidate_id")["success_probability"]
        .quantile(0.25)
        .rename("surface_robustness_value")
        .reset_index()
    )
    preload = (
        scored.loc[scored["w_total_n"].isin([0.5, 1.0])]
        .groupby("candidate_id", dropna=False)["success_probability"]
        .mean()
        .rename("preload_robustness_value")
        .reset_index()
    )
    candidate = (
        scored.groupby("candidate_id", dropna=False)
        .agg(
            success_probability=("success_probability", "mean"),
            success_probability_score=("success_probability_score_row", "mean"),
            force_score=("force_score_row", "mean"),
            efficiency_score=("efficiency_score_row", "mean"),
            n_eff_kish_score=("n_eff_kish_score_row", "mean"),
            low_eta_score=("low_eta_score_row", "mean"),
            low_search_failure_score=("low_search_failure_score_row", "mean"),
            low_saturation_score=("low_saturation_score_row", "mean"),
            low_failure_score=("low_failure_score_row", "mean"),
            f_t_lim_n_mean=("f_t_lim_n_mean", "mean"),
            f_t_lim_n_p05=("f_t_lim_n_p05", "mean"),
            f_t_lim_n_p95=("f_t_lim_n_p95", "mean"),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total_mean", "mean"),
            eta_max_mean=("eta_max_mean", "mean"),
            n_eff_kish_mean=("n_eff_kish_mean", "mean"),
            n_eng_mean=("n_eng_mean", "mean"),
            r_uncontacted_mean=("r_uncontacted_mean", "mean"),
            r_fail_search_mean=("r_fail_search_mean", "mean"),
            r_sat_n_mean=("r_sat_n_mean", "mean"),
            r_sat_y_mean=("r_sat_y_mean", "mean"),
            r_side_contact_risk_mean=("r_side_contact_risk_mean", "mean"),
            r_slip_mean=("r_slip_mean", "mean"),
            r_overload_mean=("r_overload_mean", "mean"),
            r_micro_damage_risk_mean=("r_micro_damage_risk_mean", "mean"),
            n_cases=("n_cases", "sum"),
            array_type=("array_type", "first"),
            rows=("rows", "first"),
            cols=("cols", "first"),
            pitch_t_mm=("pitch_t_mm", "first"),
            pitch_l_mm=("pitch_l_mm", "first"),
            spring_k_n_per_m=("spring_k_n_per_m", "first"),
            alpha_p_deg=("alpha_p_deg", "first"),
        )
        .reset_index()
    )
    candidate = candidate.merge(robustness, on="candidate_id", how="left")
    candidate = candidate.merge(preload, on="candidate_id", how="left")
    candidate["surface_robustness_score"] = score_high(candidate["surface_robustness_value"])
    candidate["preload_robustness_score"] = score_high(
        candidate["preload_robustness_value"].fillna(candidate["success_probability"])
    )
    # P6 沿用 P5 风格指标，用更大的 Monte Carlo 样本做最终比较，而非绝对墙面预测。
    candidate["score_total"] = (
        0.25 * candidate["success_probability_score"]
        + 0.20 * candidate["force_score"]
        + 0.10 * candidate["efficiency_score"]
        + 0.10 * candidate["surface_robustness_score"]
        + 0.10 * candidate["preload_robustness_score"]
        + 0.10 * candidate["n_eff_kish_score"]
        + 0.08 * candidate["low_eta_score"]
        + 0.03 * candidate["low_search_failure_score"]
        + 0.02 * candidate["low_saturation_score"]
        + 0.02 * candidate["low_failure_score"]
    )
    candidate = candidate.sort_values(
        ["score_total", "success_probability", "f_t_lim_n_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    candidate.insert(0, "stage", "p6_final_3d_monte_carlo")
    candidate["rank"] = candidate.index + 1
    candidate["selected"] = True
    return candidate[
        [
            "stage",
            "rank",
            "candidate_id",
            "array_type",
            "score_total",
            "success_probability",
            "surface_robustness_value",
            "preload_robustness_value",
            "f_t_lim_n_mean",
            "f_t_lim_n_p05",
            "f_t_lim_n_p95",
            "f_t_lim_over_w_total_mean",
            "eta_max_mean",
            "n_eff_kish_mean",
            "n_eng_mean",
            "r_uncontacted_mean",
            "r_fail_search_mean",
            "r_sat_n_mean",
            "r_sat_y_mean",
            "r_side_contact_risk_mean",
            "r_slip_mean",
            "r_overload_mean",
            "r_micro_damage_risk_mean",
            "n_cases",
            "rows",
            "cols",
            "pitch_t_mm",
            "pitch_l_mm",
            "alpha_p_deg",
            "spring_k_n_per_m",
            "success_probability_score",
            "force_score",
            "efficiency_score",
            "surface_robustness_score",
            "preload_robustness_score",
            "n_eff_kish_score",
            "low_eta_score",
            "low_search_failure_score",
            "low_saturation_score",
            "low_failure_score",
        ]
    ]


def convergence_statistics(summary: Any) -> Any:
    """随着每类表面样本数增加，计算统计收敛检查点。"""
    pd = _require_pandas()
    if "surface_index_within_kind" not in summary.columns:
        summary = summary.copy()
        summary["surface_index_within_kind"] = summary.groupby(["surface_kind", "surface_id"]).ngroup()
    max_n = int(summary["surface_index_within_kind"].max()) + 1 if len(summary) else 0
    checkpoints = _convergence_checkpoints(max_n)
    records: list[dict[str, Any]] = []
    for candidate_id, candidate_rows in summary.groupby("candidate_id", dropna=False):
        for n_used in checkpoints:
            subset = candidate_rows.loc[candidate_rows["surface_index_within_kind"] < n_used]
            if subset.empty:
                continue
            records.append(
                {
                    "candidate_id": candidate_id,
                    "array_type": subset["array_type"].iloc[0],
                    "n_surfaces_used": int(n_used),
                    "n_cases": int(len(subset)),
                    "f_t_lim_n_mean": float(subset["f_t_lim_n"].mean()),
                    "f_t_lim_n_p05": float(subset["f_t_lim_n"].quantile(0.05)),
                    "f_t_lim_n_p95": float(subset["f_t_lim_n"].quantile(0.95)),
                    "success_probability": float(subset["load_success"].mean()),
                    "eta_max_mean": float(subset["eta_max"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def run_p7_surface_generalization(
    *,
    p6_dir: str | Path,
    outdir: str | Path = "outputs/P7_surface_generalization",
) -> Path:
    """仅从已保存 P6 数据生成 P7 表面泛化结果，不重新仿真。"""
    stage_dir = Path(outdir)
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "figures_report").mkdir(parents=True, exist_ok=True)
    (stage_dir / "reports").mkdir(parents=True, exist_ok=True)
    summary = read_parquet(Path(p6_dir) / "data" / "final_summary.parquet")
    stats, rankings = surface_generalization(summary)
    write_parquet(stats, data_dir / "surface_generalization_statistics.parquet")
    write_parquet(rankings, data_dir / "surface_generalization_rankings.parquet")
    _write_p7_report(stage_dir, rankings)
    write_schema(
        stage_dir,
        {
            "surface_generalization_statistics": dataframe_schema(stats),
            "surface_generalization_rankings": dataframe_schema(rankings),
        },
    )
    write_manifest(
        create_manifest(
            project_name="Spine_Sim_V2",
            model_version="P7_surface_generalization",
            surface_bank_id=_first_or_none(summary, "surface_bank_id"),
            random_seed_policy="No simulation; deterministic post-processing of P6 final_summary.parquet.",
            parameter_grid={"p6_dir": str(p6_dir)},
            n_cases_expected=int(len(summary)),
            n_cases_completed=int(len(summary)),
            failed_cases=[],
            notes="P7 reads P6 data and does not rerun simulation.",
        ),
        stage_dir,
    )
    return stage_dir


def run_p8_preload_efficiency(
    *,
    p6_dir: str | Path,
    outdir: str | Path = "outputs/P8_preload_efficiency",
) -> Path:
    """仅从已保存 P6 数据生成 P8 预载效率结果，不重新仿真。"""
    stage_dir = Path(outdir)
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "figures_report").mkdir(parents=True, exist_ok=True)
    (stage_dir / "reports").mkdir(parents=True, exist_ok=True)
    summary = read_parquet(Path(p6_dir) / "data" / "final_summary.parquet")
    stats = preload_efficiency(summary)
    write_parquet(stats, data_dir / "preload_efficiency_statistics.parquet")
    _write_p8_report(stage_dir, stats)
    write_schema(stage_dir, {"preload_efficiency_statistics": dataframe_schema(stats)})
    write_manifest(
        create_manifest(
            project_name="Spine_Sim_V2",
            model_version="P8_preload_efficiency",
            surface_bank_id=_first_or_none(summary, "surface_bank_id"),
            random_seed_policy="No simulation; deterministic post-processing of P6 final_summary.parquet.",
            parameter_grid={"p6_dir": str(p6_dir)},
            n_cases_expected=int(len(summary)),
            n_cases_completed=int(len(summary)),
            failed_cases=[],
            notes="P8 reads P6 data and does not rerun simulation.",
        ),
        stage_dir,
    )
    return stage_dir


def surface_generalization(summary: Any) -> tuple[Any, Any]:
    """计算 P7 各表面类别下的泛化表现和排名漂移。"""
    stats = (
        summary.groupby(
            [
                "candidate_id",
                "array_type",
                "surface_kind",
                "rows",
                "cols",
                "pitch_t_mm",
                "pitch_l_mm",
                "alpha_p_deg",
                "spring_k_n_per_m",
            ],
            dropna=False,
        )
        .agg(
            n_cases=("case_id", "count"),
            success_probability=("load_success", "mean"),
            f_t_lim_n_mean=("f_t_lim_n", "mean"),
            f_t_lim_n_p05=("f_t_lim_n", lambda s: s.quantile(0.05)),
            f_t_lim_n_p95=("f_t_lim_n", lambda s: s.quantile(0.95)),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total", "mean"),
            eta_max_mean=("eta_max", "mean"),
        )
        .reset_index()
    )
    stats["surface_rank"] = stats.groupby("surface_kind")["success_probability"].rank(
        method="first",
        ascending=False,
    )
    global_rank = (
        stats.groupby("candidate_id", dropna=False)["success_probability"]
        .mean()
        .rank(method="first", ascending=False)
        .rename("global_rank")
        .reset_index()
    )
    rankings = stats.merge(global_rank, on="candidate_id", how="left")
    rankings["rank_shift"] = rankings["surface_rank"] - rankings["global_rank"]
    rankings = rankings.sort_values(["surface_kind", "surface_rank", "candidate_id"]).reset_index(drop=True)
    return stats, rankings


def preload_efficiency(summary: Any) -> Any:
    """从 P6 summary 计算 P8 预载响应曲线数据。"""
    return (
        summary.groupby(
            [
                "candidate_id",
                "array_type",
                "w_total_n",
                "rows",
                "cols",
                "pitch_t_mm",
                "pitch_l_mm",
                "alpha_p_deg",
                "spring_k_n_per_m",
            ],
            dropna=False,
        )
        .agg(
            n_cases=("case_id", "count"),
            success_probability=("load_success", "mean"),
            f_t_lim_n_mean=("f_t_lim_n", "mean"),
            f_t_lim_n_p05=("f_t_lim_n", lambda s: s.quantile(0.05)),
            f_t_lim_n_p95=("f_t_lim_n", lambda s: s.quantile(0.95)),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total", "mean"),
            f_t_lim_over_w_total_p05=("f_t_lim_over_w_total", lambda s: s.quantile(0.05)),
            f_t_lim_over_w_total_p95=("f_t_lim_over_w_total", lambda s: s.quantile(0.95)),
            eta_max_mean=("eta_max", "mean"),
        )
        .reset_index()
    )


def _write_final_candidates_json(rankings: Any, path: Path) -> None:
    records = []
    for _, row in rankings.sort_values("rank").iterrows():
        records.append({key: _json_value(row[key]) for key in rankings.columns})
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _convergence_checkpoints(max_n: int) -> list[int]:
    if max_n <= 0:
        return []
    preferred = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
    values = [n for n in preferred if n <= max_n]
    values.extend([max_n // 4, max_n // 2, max_n])
    return sorted({int(n) for n in values if n > 0})


def _write_p7_report(stage_dir: Path, rankings: Any) -> None:
    lines = [
        "# P7 Surface Generalization",
        "",
        "Generated from P6 final summary data without rerunning simulation.",
        "",
        "| surface_kind | rank | candidate_id | success_probability | rank_shift |",
        "|---|---:|---|---:|---:|",
    ]
    for _, row in rankings.sort_values(["surface_kind", "surface_rank"]).iterrows():
        lines.append(
            f"| {row['surface_kind']} | {float(row['surface_rank']):.0f} | {row['candidate_id']} | {float(row['success_probability']):.4f} | {float(row['rank_shift']):.0f} |"
        )
    (stage_dir / "reports" / "surface_generalization_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_p8_report(stage_dir: Path, stats: Any) -> None:
    lines = [
        "# P8 Preload Efficiency",
        "",
        "Generated from P6 final summary data without rerunning simulation.",
        "",
        "| candidate_id | w_total_n | success_probability | f_t_lim_n_mean | efficiency_mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in stats.sort_values(["candidate_id", "w_total_n"]).iterrows():
        lines.append(
            f"| {row['candidate_id']} | {float(row['w_total_n']):.3g} | {float(row['success_probability']):.4f} | {float(row['f_t_lim_n_mean']):.6g} | {float(row['f_t_lim_over_w_total_mean']):.6g} |"
        )
    (stage_dir / "reports" / "preload_efficiency_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _first_or_none(df: Any, column: str) -> Any:
    if column not in df.columns or df.empty:
        return None
    value = df[column].dropna()
    return None if value.empty else _json_value(value.iloc[0])


def _json_value(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ModuleNotFoundError, TypeError, ValueError):
        pass
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Final Monte Carlo analysis requires pandas.") from exc
    return pd
