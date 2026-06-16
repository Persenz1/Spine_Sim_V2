"""P2/P3/P5 筛选阶段的候选评分、排序与入选规则。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Spine_Sim_V2.analysis.scoring import score_high, score_low
from Spine_Sim_V2.analysis.statistics import grouped_statistics
from Spine_Sim_V2.io.parquet_io import read_parquet, write_parquet


def analyze_stage(stage_dir: str | Path) -> tuple[Any, Any]:
    """从已保存阶段数据重新计算分组统计和候选排名。"""
    stage_path = Path(stage_dir)
    summary_path = stage_path / "data" / "stage_summary.parquet"
    spines_path = stage_path / "data" / "stage_spines.parquet"
    _require_file(summary_path, "stage summary")
    _require_file(spines_path, "stage spines")
    summary = read_parquet(summary_path)
    stage_kind = infer_stage_kind(stage_path)
    grouped = grouped_statistics(summary, stage_name=stage_kind)
    rankings = rank_candidates(grouped, stage_kind=stage_kind)
    data_dir = stage_path / "data"
    write_parquet(grouped, data_dir / "stage_grouped_statistics.parquet")
    write_parquet(rankings, data_dir / "stage_rankings.parquet")
    write_selection_reason(stage_path, rankings, stage_kind=stage_kind)
    write_stage_report(stage_path, grouped, rankings, stage_kind=stage_kind)
    write_selected_candidates(stage_path, rankings, stage_kind=stage_kind)
    return grouped, rankings


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} data product: {path}")


def rank_candidates(grouped: Any, *, stage_kind: str) -> Any:
    """根据阶段类型调用对应的候选排序规则。"""
    if stage_kind.startswith("p2") or "P2" in stage_kind:
        return _rank_p2(grouped)
    if stage_kind.startswith("p3") or "P3" in stage_kind:
        return _rank_p3(grouped)
    if stage_kind.startswith("p5") or "P5" in stage_kind:
        return _rank_p5(grouped, stage_kind=stage_kind)
    raise ValueError(f"Unsupported stage kind for ranking: {stage_kind}")


def infer_stage_kind(stage_dir: str | Path) -> str:
    name = Path(stage_dir).name
    if name.startswith("P2"):
        return "p2_compliant_k_alpha"
    if name.startswith("P3"):
        return "p3_rigid_alpha"
    if name.startswith("P5a"):
        return "p5a_array_pitch_coarse"
    if name.startswith("P5b"):
        return "p5b_array_pitch_refine"
    summary_path = Path(stage_dir) / "data" / "stage_summary.parquet"
    if summary_path.exists():
        summary = read_parquet(summary_path)
        stages = (
            set(str(item) for item in summary["stage"].dropna().unique())
            if "stage" in summary.columns
            else set()
        )
        if any("p5" in stage for stage in stages):
            return "p5b_array_pitch_refine" if "P5b" in name else "p5a_array_pitch_coarse"
        if set(summary["array_type"].dropna().unique()) == {"rigid"}:
            return "p3_rigid_alpha"
    return "p2_compliant_k_alpha"


def write_selection_reason(stage_dir: str | Path, rankings: Any, *, stage_kind: str) -> Path:
    """写出简要的 ``selection_reason.md`` 入选理由报告。"""
    path = Path(stage_dir) / "reports" / "selection_reason.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rankings.loc[rankings["selected"] == True].copy()  # noqa: E712
    lines = [
        "# Selection Reason",
        "",
        f"Stage: `{stage_kind}`",
        "",
        "Selected candidates are marked by the ranking rules and diversity constraints.",
        "",
        "| rank | candidate_id | score_total | reason |",
        "|---:|---|---:|---|",
    ]
    for _, row in selected.sort_values("rank").iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['candidate_id']} | {float(row['score_total']):.4f} | {row.get('selection_reason', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_stage_report(stage_dir: str | Path, grouped: Any, rankings: Any, *, stage_kind: str) -> Path:
    """基于已保存分析产品写出简要 ``stage_report.md``。"""
    path = Path(stage_dir) / "reports" / "stage_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = rankings.loc[rankings["selected"] == True].copy() if "selected" in rankings.columns else rankings.head(10).copy()  # noqa: E712
    lines = [
        "# Stage Report",
        "",
        f"Stage: `{stage_kind}`",
        "",
        f"Grouped rows: {len(grouped)}",
        f"Ranking rows: {len(rankings)}",
        "",
        "## Selected Candidates",
        "",
        "| rank | candidate_id | array_type | score_total | success_probability | f_t_lim_n_mean |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, row in selected.sort_values("rank").iterrows():
        lines.append(
            "| {rank} | {candidate_id} | {array_type} | {score_total:.4f} | {success:.4f} | {force:.6g} |".format(
                rank=int(row.get("rank", 0)),
                candidate_id=row.get("candidate_id", ""),
                array_type=row.get("array_type", ""),
                score_total=float(row.get("score_total", 0.0)),
                success=float(row.get("success_probability", 0.0)),
                force=float(row.get("f_t_lim_n_mean", 0.0)),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_selected_candidates(stage_dir: str | Path, rankings: Any, *, stage_kind: str) -> tuple[Path, Path]:
    """将入选候选的完整参数写为 JSON 和 Parquet。"""
    pd = _require_pandas()
    stage_path = Path(stage_dir)
    data_dir = stage_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    selected = rankings.loc[rankings["selected"] == True].copy()  # noqa: E712
    records = [_selected_record(row.to_dict(), stage_kind=stage_kind) for _, row in selected.iterrows()]
    json_path = data_dir / "selected_candidates.json"
    parquet_path = data_dir / "selected_candidates.parquet"
    json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_parquet(pd.DataFrame.from_records(records), parquet_path)
    return json_path, parquet_path


def _selected_record(row: dict[str, Any], *, stage_kind: str) -> dict[str, Any]:
    array_type = row.get("array_type")
    if array_type is None:
        array_type = "compliant" if stage_kind.startswith("p2") else "rigid"
    return {
        "candidate_id": row.get("candidate_id"),
        "source_stage": stage_kind,
        "array_type": array_type,
        "rows": _as_int(row.get("rows", 1)),
        "cols": _as_int(row.get("cols", 1)),
        "pitch_t_mm": _as_float_or_none(row.get("pitch_t_mm")),
        "pitch_l_mm": _as_float_or_none(row.get("pitch_l_mm")),
        "alpha_p_deg": _as_float_or_none(row.get("alpha_p_deg")),
        "spring_k_n_per_m": _as_float_or_none(row.get("spring_k_n_per_m")),
        "score_total": _as_float_or_none(row.get("score_total")),
        "rank": _as_int(row.get("rank")),
        "selection_reason": row.get("selection_reason", ""),
    }


def _rank_p2(grouped: Any) -> Any:
    """按 P2 权重和多样性约束排序柔顺 k-alpha 候选。"""
    pd = _require_pandas()
    scored = grouped.copy()
    scored["success_probability_score_row"] = score_high(scored["success_probability"])
    scored["force_score_row"] = score_high(scored["f_t_lim_n_mean"])
    scored["efficiency_score_row"] = score_high(scored["f_t_lim_over_w_total_mean"])
    scored["low_search_failure_score_row"] = score_low(scored["r_fail_search_mean"])
    scored["low_saturation_score_row"] = score_low(scored["r_sat_n_mean"])
    robustness = (
        scored.groupby(["candidate_id", "surface_kind"], dropna=False)["success_probability"]
        .mean()
        .reset_index()
        .groupby("candidate_id")["success_probability"]
        .quantile(0.25)
        .rename("surface_robustness_value")
        .reset_index()
    )
    candidate = (
        scored.groupby("candidate_id", dropna=False)
        .agg(
            success_probability=("success_probability", "mean"),
            success_probability_score=("success_probability_score_row", "mean"),
            force_score=("force_score_row", "mean"),
            efficiency_score=("efficiency_score_row", "mean"),
            low_search_failure_score=("low_search_failure_score_row", "mean"),
            low_saturation_score=("low_saturation_score_row", "mean"),
            f_t_lim_n_mean=("f_t_lim_n_mean", "mean"),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total_mean", "mean"),
            r_fail_search_mean=("r_fail_search_mean", "mean"),
            r_sat_n_mean=("r_sat_n_mean", "mean"),
            n_cases=("n_cases", "sum"),
            array_type=("array_type", "first"),
            spring_k_n_per_m=("spring_k_n_per_m", "first"),
            alpha_p_deg=("alpha_p_deg", "first"),
        )
        .reset_index()
    )
    candidate = candidate.merge(robustness, on="candidate_id", how="left")
    candidate["surface_robustness_score"] = score_high(candidate["surface_robustness_value"])
    # P2 不只看承载力，还显式加入成功率、效率、表面稳健性和失败/饱和风险。
    candidate["score_total"] = (
        0.30 * candidate["success_probability_score"]
        + 0.20 * candidate["force_score"]
        + 0.15 * candidate["efficiency_score"]
        + 0.15 * candidate["surface_robustness_score"]
        + 0.10 * candidate["low_search_failure_score"]
        + 0.10 * candidate["low_saturation_score"]
    )
    candidate = candidate.sort_values(
        ["score_total", "success_probability", "f_t_lim_n_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    candidate["rank"] = candidate.index + 1
    selected_ids, reasons = _select_p2(candidate)
    return _format_ranking(candidate, stage="p2_compliant_k_alpha", selected_ids=selected_ids, reasons=reasons)


def _rank_p3(grouped: Any) -> Any:
    """按 P3 权重排序刚性安装角候选。"""
    scored = grouped.copy()
    scored["success_probability_score_row"] = score_high(scored["success_probability"])
    scored["force_score_row"] = score_high(scored["f_t_lim_n_mean"])
    scored["efficiency_score_row"] = score_high(scored["f_t_lim_over_w_total_mean"])
    scored["low_search_failure_score_row"] = score_low(scored["r_fail_search_mean"])
    robustness = (
        scored.groupby(["candidate_id", "surface_kind"], dropna=False)["success_probability"]
        .mean()
        .reset_index()
        .groupby("candidate_id")["success_probability"]
        .quantile(0.25)
        .rename("surface_robustness_value")
        .reset_index()
    )
    candidate = (
        scored.groupby("candidate_id", dropna=False)
        .agg(
            success_probability=("success_probability", "mean"),
            success_probability_score=("success_probability_score_row", "mean"),
            force_score=("force_score_row", "mean"),
            efficiency_score=("efficiency_score_row", "mean"),
            low_search_failure_score=("low_search_failure_score_row", "mean"),
            f_t_lim_n_mean=("f_t_lim_n_mean", "mean"),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total_mean", "mean"),
            r_fail_search_mean=("r_fail_search_mean", "mean"),
            n_cases=("n_cases", "sum"),
            array_type=("array_type", "first"),
            alpha_p_deg=("alpha_p_deg", "first"),
        )
        .reset_index()
    )
    candidate = candidate.merge(robustness, on="candidate_id", how="left")
    candidate["surface_robustness_score"] = score_high(candidate["surface_robustness_value"])
    candidate["score_total"] = (
        0.35 * candidate["success_probability_score"]
        + 0.25 * candidate["force_score"]
        + 0.15 * candidate["efficiency_score"]
        + 0.15 * candidate["surface_robustness_score"]
        + 0.10 * candidate["low_search_failure_score"]
    )
    candidate = candidate.sort_values(
        ["score_total", "success_probability", "f_t_lim_n_mean"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    candidate["rank"] = candidate.index + 1
    selected_ids, reasons = _select_p3(candidate)
    return _format_ranking(candidate, stage="p3_rigid_alpha", selected_ids=selected_ids, reasons=reasons)


def _rank_p5(grouped: Any, *, stage_kind: str) -> Any:
    """按 P5 阵列筛选权重排序候选。"""
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
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total_mean", "mean"),
            n_eff_kish_mean=("n_eff_kish_mean", "mean"),
            n_eng_mean=("n_eng_mean", "mean"),
            eta_max_mean=("eta_max_mean", "mean"),
            r_fail_search_mean=("r_fail_search_mean", "mean"),
            r_sat_n_mean=("r_sat_n_mean", "mean"),
            r_sat_y_mean=("r_sat_y_mean", "mean"),
            r_side_contact_risk_mean=("r_side_contact_risk_mean", "mean"),
            cascade_failure_rate=("cascade_failure_rate", "mean"),
            normal_range_insufficient_rate=("normal_range_insufficient_rate", "mean"),
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
    candidate["preload_robustness_score"] = score_high(candidate["preload_robustness_value"].fillna(candidate["success_probability"]))
    # P5/P6 评分强调成功率和承载力，同时惩罚载荷集中、搜索失败和饱和风险。
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
    candidate["rank"] = candidate.index + 1
    if stage_kind.startswith("p5b"):
        selected_ids, reasons = _select_p5b(candidate)
    else:
        selected_ids, reasons = _select_p5a(candidate)
    return _format_ranking(candidate, stage=stage_kind, selected_ids=selected_ids, reasons=reasons)


def _format_ranking(candidate: Any, *, stage: str, selected_ids: set[str], reasons: dict[str, str]) -> Any:
    """整理 rankings 表字段，并标记入选候选及其理由。"""
    rows = candidate.copy()
    rows.insert(0, "stage", stage)
    rows["selected"] = rows["candidate_id"].isin(selected_ids)
    rows["selection_reason"] = rows["candidate_id"].map(reasons).fillna("")
    rows["score_load"] = rows["force_score"]
    rows["score_success"] = rows["success_probability_score"]
    rows["score_uniformity"] = rows.get("low_saturation_score", rows["low_search_failure_score"])
    rows["score_search"] = rows["low_search_failure_score"]
    rows["notes"] = rows["selection_reason"]
    preferred = [
        "stage",
        "rank",
        "candidate_id",
        "score_total",
        "score_load",
        "score_success",
        "score_uniformity",
        "score_search",
        "n_cases",
        "notes",
        "selected",
        "selection_reason",
        "array_type",
        "rows",
        "cols",
        "pitch_t_mm",
        "pitch_l_mm",
        "spring_k_n_per_m",
        "alpha_p_deg",
        "success_probability",
        "surface_robustness_score",
        "preload_robustness_score",
        "efficiency_score",
        "n_eff_kish_score",
        "low_eta_score",
        "low_search_failure_score",
        "low_saturation_score",
        "low_failure_score",
        "f_t_lim_n_mean",
        "f_t_lim_over_w_total_mean",
        "n_eff_kish_mean",
        "n_eng_mean",
        "eta_max_mean",
        "r_fail_search_mean",
        "r_sat_n_mean",
        "r_sat_y_mean",
        "r_side_contact_risk_mean",
        "cascade_failure_rate",
        "normal_range_insufficient_rate",
    ]
    existing = [col for col in preferred if col in rows.columns]
    return rows[existing]


def _select_p5a(candidate: Any) -> tuple[set[str], dict[str, str]]:
    """P5a 粗筛保留刚性/柔顺各自前若干名。"""
    selected: list[str] = []
    reasons: dict[str, str] = {}
    for array_type in ("rigid", "compliant"):
        subset = candidate.loc[candidate["array_type"] == array_type]
        keep_n = min(18, len(subset))
        for _, row in subset.head(keep_n).iterrows():
            cid = str(row["candidate_id"])
            selected.append(cid)
            reasons[cid] = f"P5a top-{keep_n} {array_type} coarse-screen candidate"
    return set(selected), reasons


def _select_p5b(candidate: Any) -> tuple[set[str], dict[str, str]]:
    """P5b 按角色选择刚性 5 个和柔顺 5 个最终候选。"""
    selected: list[str] = []
    reasons: dict[str, str] = {}
    for array_type in ("rigid", "compliant"):
        subset = candidate.loc[candidate["array_type"] == array_type].copy()
        if subset.empty:
            continue
        if array_type == "rigid":
            selectors = [
                ("R1综合分最高", subset.sort_values("score_total", ascending=False)),
                ("R2承载力最高", subset.sort_values("f_t_lim_n_mean", ascending=False)),
                ("R3成功率最高", subset.sort_values("success_probability", ascending=False)),
                (
                    "R4低eta/高有效刺数",
                    subset.assign(balance_score=subset["low_eta_score"] + subset["n_eff_kish_score"]).sort_values(
                        "balance_score", ascending=False
                    ),
                ),
                (
                    "R5 60度基准或旧基准结构",
                    subset.assign(alpha_distance=(subset["alpha_p_deg"] - 60.0).abs()).sort_values(
                        ["alpha_distance", "score_total"], ascending=[True, False]
                    ),
                ),
            ]
        else:
            median_k = subset["spring_k_n_per_m"].dropna().median()
            selectors = [
                ("C1综合分最高", subset.sort_values("score_total", ascending=False)),
                ("C2承载力最高", subset.sort_values("f_t_lim_n_mean", ascending=False)),
                ("C3成功率最高", subset.sort_values("success_probability", ascending=False)),
                (
                    "C4低饱和/低搜索失败",
                    subset.assign(stability_score=subset["low_saturation_score"] + subset["low_search_failure_score"]).sort_values(
                        "stability_score", ascending=False
                    ),
                ),
                (
                    "C5 60度或中等刚度基准结构",
                    subset.assign(
                        baseline_distance=(subset["alpha_p_deg"] - 60.0).abs()
                        + (subset["spring_k_n_per_m"].fillna(median_k) - median_k).abs() / max(float(median_k), 1.0)
                    ).sort_values(["baseline_distance", "score_total"], ascending=[True, False]),
                ),
            ]
        for reason, ordered in selectors:
            _append_first_unused(ordered, selected, reasons, reason)
        while len([cid for cid in selected if cid in set(subset["candidate_id"])]) < min(5, len(subset)):
            before = len(selected)
            _append_first_unused(subset.sort_values("score_total", ascending=False), selected, reasons, f"{array_type}顺延候选")
            if len(selected) == before:
                break
    return set(selected), reasons


def _append_first_unused(ordered: Any, selected: list[str], reasons: dict[str, str], reason: str) -> None:
    for _, row in ordered.iterrows():
        cid = str(row["candidate_id"])
        if cid not in selected:
            selected.append(cid)
            reasons[cid] = reason
            return


def _select_p2(candidate: Any) -> tuple[set[str], dict[str, str]]:
    """P2 保留 6-8 个兼顾角度和刚度多样性的候选。"""
    selected: list[str] = list(candidate.head(6)["candidate_id"])
    reasons = {candidate_id: "top-6 score" for candidate_id in selected}

    def add_best(mask: Any, reason: str) -> None:
        matches = candidate.loc[mask]
        if matches.empty:
            return
        candidate_id = str(matches.iloc[0]["candidate_id"])
        if candidate_id not in selected:
            selected.append(candidate_id)
            reasons[candidate_id] = reason

    add_best(candidate["alpha_p_deg"] == 60, "diversity: keep at least one 60 deg candidate")
    if len(set(candidate.loc[candidate["candidate_id"].isin(selected), "alpha_p_deg"])) < 2:
        selected_alphas = set(candidate.loc[candidate["candidate_id"].isin(selected), "alpha_p_deg"])
        add_best(~candidate["alpha_p_deg"].isin(selected_alphas), "diversity: avoid one-angle selection")
    max_k = candidate["spring_k_n_per_m"].max()
    if set(candidate.loc[candidate["candidate_id"].isin(selected), "spring_k_n_per_m"]) == {max_k}:
        add_best(candidate["spring_k_n_per_m"] < max_k, "diversity: keep non-maximum stiffness")
    while len(set(candidate.loc[candidate["candidate_id"].isin(selected), "spring_k_n_per_m"])) < 3:
        selected_ks = set(candidate.loc[candidate["candidate_id"].isin(selected), "spring_k_n_per_m"])
        before = len(selected)
        add_best(~candidate["spring_k_n_per_m"].isin(selected_ks), "diversity: keep at least 3 stiffness levels")
        if len(selected) == before:
            break
    if len(selected) > 8:
        required = {cid for cid, reason in reasons.items() if reason != "top-6 score"}
        trimmed = []
        for cid in selected:
            if cid in required or len(trimmed) < 6:
                trimmed.append(cid)
            if len(trimmed) == 8:
                break
        selected = trimmed
    return set(selected), reasons


def _select_p3(candidate: Any) -> tuple[set[str], dict[str, str]]:
    """P3 强制保留 60 度基准，并补充表现较好的其他角度。"""
    selected: list[str] = []
    reasons: dict[str, str] = {}
    alpha60 = candidate.loc[candidate["alpha_p_deg"] == 60]
    if not alpha60.empty:
        cid = str(alpha60.iloc[0]["candidate_id"])
        selected.append(cid)
        reasons[cid] = "forced retention: 60 deg baseline"
    for _, row in candidate.iterrows():
        cid = str(row["candidate_id"])
        if cid in selected:
            continue
        selected.append(cid)
        reasons[cid] = "best non-60 angle by score"
        if len(selected) >= 3:
            break
    return set(selected), reasons


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ranking requires pandas.") from exc
    return pd


def _as_float_or_none(value: Any) -> float | None:
    try:
        if value is None or value != value:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value != value:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
