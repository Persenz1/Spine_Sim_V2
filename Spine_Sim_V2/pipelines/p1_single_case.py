"""P1 单 case 闭环仿真管线。

本模块是所有筛选阶段复用的物理主链条：从 surface bank 取表面，建立阵列，
求解局部预载，搜索首次接合，计算单刺容量，最后做事件驱动切向加载。
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from Spine_Sim_V2.core.types import (
    SingleCaseInput,
    SingleCaseResult,
    schema_field_names,
    stage_spines_schema,
    stage_summary_schema,
)
from Spine_Sim_V2.io.manifest import create_manifest, write_manifest
from Spine_Sim_V2.io.npz_io import save_npz_arrays
from Spine_Sim_V2.io.parquet_io import write_parquet, write_preview_csv
from Spine_Sim_V2.io.schema_io import write_schema
from Spine_Sim_V2.solvers.contact import (
    build_rectangular_array_geometry,
    build_stiffness_model,
    solve_preload_distribution,
)
from Spine_Sim_V2.solvers.engagement import (
    compute_capacity_n,
    compute_critical_angle_deg,
    effective_contact_angle_map_deg,
    phi_hook_min_deg,
    phi_s_deg_from_friction,
    side_contact_risk_from_angle,
)
from Spine_Sim_V2.solvers.loading import run_loading_sequence
from Spine_Sim_V2.solvers.search import search_first_engagement
from Spine_Sim_V2.surfaces.bank import SurfaceBank


def run_single_case(case: SingleCaseInput | None = None, **kwargs: Any) -> SingleCaseResult:
    """运行一个完整仿真 case，并返回 summary/spines 两张表。"""
    pd = _require_pandas()
    if case is None:
        case = SingleCaseInput(**kwargs)
    elif kwargs:
        case = replace(case, **kwargs)
    _validate_case(case)

    diagnostics: dict[str, Any] = {"solver_sequence": []}
    # case 只保存 surface_id；实际高度图始终从 surface bank 读取，避免重复落盘。
    bank = SurfaceBank.open(case.surface_bank_path)
    surface_record = bank.get_surface_record(case.surface_id)
    arrays = bank.load_surface_arrays(case.surface_id)
    height_filtered = arrays["height_filtered"]
    diagnostics["solver_sequence"].append("surface_loaded")

    dx_mm = float(surface_record["dx_mm"])
    dy_mm = float(surface_record["dy_mm"])
    stiffness = build_stiffness_model(case)
    geometry = build_rectangular_array_geometry(
        rows=case.rows,
        cols=case.cols,
        pitch_t_mm=case.pitch_t_mm,
        pitch_l_mm=case.pitch_l_mm,
        surface_shape=height_filtered.shape,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
    )
    diagnostics["solver_sequence"].append("geometry_built")

    # 强制先求解初始间隙和局部预载 W_i，后面的 phi_c 与可接合区域都依赖它。
    preload = solve_preload_distribution(
        case=case,
        stiffness=stiffness,
        height_filtered=height_filtered,
        geometry=geometry,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
    )
    diagnostics["solver_sequence"].extend(["gaps_computed", "preload_solved"])

    # 第一版有效接触角来自 height_filtered 的局部坡度；接口保留给后续三维几何替换。
    phi_map_deg = effective_contact_angle_map_deg(
        height_filtered,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
    )
    phi_s_deg = phi_s_deg_from_friction(case.f_s)
    phi_hook_deg = phi_hook_min_deg(case.alpha_p_deg, phi_s_deg)
    # ``trial_force_n`` is the per-spine force demand in the phi_c formula.  It is
    # intentionally not divided by n_nom; otherwise default arrays make phi_c
    # collapse toward -phi_s and the W_i -> phi_c engagement path becomes inactive.
    per_spine_trial_force_n = case.trial_force_n

    spine_records: list[dict[str, Any]] = []
    cap_values: list[float] = []
    cap_mode_values: list[str | None] = []
    engaged_values: list[bool] = []
    search_values: list[float] = []
    tip_area_mm2 = float(np.pi * max(case.tip_radius_mm, 1e-9) ** 2)
    for idx, geom in enumerate(geometry):
        w_i = float(preload.preload_n[idx])
        # 这里必须在 W_i 已知之后计算 phi_c；无接触刺返回 NaN 并跳过接合资格。
        phi_c_deg = compute_critical_angle_deg(
            preload_n=w_i,
            target_force_n=per_spine_trial_force_n,
            f_s=case.f_s,
        )
        if w_i <= 0.0:
            # 无局部预载时不允许进入有效接合；仍调用搜索函数以保留一致状态输出。
            search_result = search_first_engagement(
                phi_map_deg=phi_map_deg,
                x0_mm=geom["x_mm"],
                y0_mm=geom["y_mm"],
                search_travel_mm=case.search_travel_mm,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                phi_c_deg=phi_c_deg,
                phi_hook_min_deg=phi_hook_deg,
            )
            cap_result = compute_capacity_n(
                preload_n=0.0,
                phi_eng_deg=float("nan"),
                f_s=case.f_s,
                F_ref_star_n=case.F_ref_star_n,
            )
            side_risk = False
        else:
            # 接触刺沿有限切向行程搜索第一个满足 phi_c/phi_hook 阈值的位置。
            search_result = search_first_engagement(
                phi_map_deg=phi_map_deg,
                x0_mm=geom["x_mm"],
                y0_mm=geom["y_mm"],
                search_travel_mm=case.search_travel_mm,
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                phi_c_deg=phi_c_deg,
                phi_hook_min_deg=phi_hook_deg,
                height_filtered=height_filtered,
                z_tip_mm=float(preload.height_at_spines_mm[idx]),
            )
            phi_eng = (
                float(search_result.phi_eng_deg)
                if search_result.phi_eng_deg is not None
                else float("nan")
            )
            cap_result = compute_capacity_n(
                preload_n=w_i,
                phi_eng_deg=phi_eng,
                f_s=case.f_s,
                F_ref_star_n=case.F_ref_star_n,
            )
            side_risk = bool(search_result.side_contact_risk) or side_contact_risk_from_angle(
                phi_eng_deg=phi_eng,
                spine_diameter_mm=case.spine_diameter_mm,
                tip_radius_mm=case.tip_radius_mm,
            )

        engaged_values.append(bool(search_result.engaged))
        cap_values.append(float(cap_result.cap_n))
        cap_mode_values.append(cap_result.cap_mode)
        search_values.append(
            float(search_result.search_distance_mm)
            if search_result.search_distance_mm is not None
            else float("nan")
        )
        state = search_result.state
        if w_i <= 0.0:
            state = "no_contact"
        contact_pressure = float(w_i / tip_area_mm2) if w_i > 0.0 else 0.0
        micro_damage_risk = (
            bool(w_i > 0.0 and contact_pressure >= case.damage_pressure_threshold_n_per_mm2)
            if case.damage_pressure_threshold_n_per_mm2 is not None
            else None
        )
        spine_records.append(
            {
                "case_id": case.case_id,
                "candidate_id": case.candidate_id,
                "surface_id": case.surface_id,
                "spine_id": geom["spine_id"],
                "row": geom["row"],
                "col": geom["col"],
                "array_type": case.array_type,
                "x_mm": geom["x_mm"],
                "y_mm": geom["y_mm"],
                "gap_mm": _finite_or_none(preload.gap_mm[idx]),
                "alpha_p_deg": case.alpha_p_deg,
                "pitch_t_mm": case.pitch_t_mm,
                "pitch_l_mm": case.pitch_l_mm,
                "contacted": bool(preload.contacted[idx]),
                "preload_n": float(w_i),
                "contact_pressure_proxy_n_per_mm2": contact_pressure,
                "micro_damage_risk": micro_damage_risk,
                "u_ax_used_mm": float(preload.u_ax_used_mm[idx]),
                "normal_saturated": bool(preload.normal_saturated[idx]),
                "state": state,
                "search_distance_mm": search_result.search_distance_mm,
                "engaged": bool(search_result.engaged),
                "engagement_x_mm": search_result.engagement_x_mm,
                "engagement_y_mm": search_result.engagement_y_mm,
                "phi_c_deg": _finite_or_none(phi_c_deg),
                "phi_eng_deg": search_result.phi_eng_deg,
                "phi_hook_min_deg": phi_hook_deg,
                "side_contact_risk": side_risk,
                "cap_n": float(cap_result.cap_n),
                "cap_mode": cap_result.cap_mode,
                "load_at_limit_n": 0.0,
                "failed": False,
                "failure_mode": None,
                "failure_order": None,
            }
        )

    diagnostics["solver_sequence"].append("engagement_searched")
    diagnostics["preload_solved_before_engagement"] = (
        diagnostics["solver_sequence"].index("preload_solved")
        < diagnostics["solver_sequence"].index("engagement_searched")
    )

    # 切向承载上限来自事件点：每根已接合刺达到容量后移除并重新分担。
    loading = run_loading_sequence(
        engaged=np.asarray(engaged_values, dtype=bool),
        search_distance_mm=np.asarray(search_values, dtype=float),
        cap_n=np.asarray(cap_values, dtype=float),
        k_tt_n_per_mm=stiffness.k_tt,
        cap_mode=cap_mode_values,
    )
    diagnostics["solver_sequence"].append("loading_solved")
    diagnostics["load_displacement_s"] = list(loading.event_displacements_mm)
    diagnostics["load_displacement_f"] = list(loading.event_total_force_n)
    diagnostics["failure_sequence"] = list(loading.failure_order)

    for idx, record in enumerate(spine_records):
        record["load_at_limit_n"] = float(loading.load_at_limit_n[idx])
        record["failed"] = bool(loading.failed[idx])
        record["failure_mode"] = loading.failure_mode[idx]
        record["failure_order"] = loading.failure_order[idx]
        if loading.failed[idx]:
            # 失效刺的 state 直接采用文档枚举的 slip / overload，不再使用 failed_* 复合值。
            record["state"] = loading.failure_mode[idx] or "overload"

    spines_df = pd.DataFrame.from_records(spine_records)
    _ensure_schema_columns(spines_df, schema_field_names(stage_spines_schema))

    summary_record = _build_summary_record(
        case=case,
        surface_bank=bank,
        surface_record=surface_record,
        stiffness=stiffness,
        preload=preload,
        spines_df=spines_df,
        loading=loading,
        phi_s_deg=phi_s_deg,
    )
    summary_df = pd.DataFrame.from_records([summary_record])
    _ensure_schema_columns(summary_df, schema_field_names(stage_summary_schema))
    return SingleCaseResult(
        case_summary=summary_df,
        case_spines=spines_df,
        diagnostics=diagnostics,
    )


def run(case: SingleCaseInput | None = None, **kwargs: Any) -> SingleCaseResult:
    """P1 的兼容入口，转发到 ``run_single_case``。"""
    return run_single_case(case, **kwargs)


def simulate_case_records(
    case: SingleCaseInput,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """运行单 case 并返回轻量记录字典，供并行 worker 跨进程传输。

    返回 ``(summary_record, spine_records, diagnostics)``。相比直接传 DataFrame，
    字典更易 pickle、IPC 开销更小，主进程再批量落盘，避免结果堆积内存。
    """
    result = run_single_case(case)
    summary_record = result.case_summary.iloc[0].to_dict()
    spine_records = result.case_spines.to_dict("records")
    return summary_record, spine_records, result.diagnostics


def run_single_case_sanity(
    *,
    surface_bank: str | Path = "data/surface_bank_debug",
    surface_id: str = "concrete_000000",
    outdir: str | Path = "outputs/P1_single_case_sanity",
    compliant_spring_k_n_per_m: float = 330.0,
) -> Path:
    """运行刚性和柔顺 P1 基准 case，并保存数据产品。

    本阶段只保存表格和诊断数组；图片由后续 ``plot_results.py p1`` 从数据生成。
    """
    pd = _require_pandas()
    stage_dir = Path(outdir)
    data_dir = stage_dir / "data"
    sample_root = stage_dir / "sample_cases"
    reports_dir = stage_dir / "reports"
    for path in (data_dir, sample_root, reports_dir, stage_dir / "figures_debug"):
        path.mkdir(parents=True, exist_ok=True)

    bank = SurfaceBank.open(surface_bank)
    cases = _default_sanity_cases(
        surface_bank_path=Path(surface_bank),
        surface_id=surface_id,
        compliant_spring_k_n_per_m=compliant_spring_k_n_per_m,
    )
    summary_frames = []
    spines_frames = []
    failed_cases: list[str] = []
    for case in cases:
        case_dir = sample_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = run_single_case(case)
            summary_frames.append(result.case_summary)
            spines_frames.append(result.case_spines)
            write_parquet(result.case_summary, case_dir / "case_summary.parquet")
            write_parquet(result.case_spines, case_dir / "case_spines.parquet")
            arrays = _build_case_arrays(
                bank=bank,
                case=case,
                result=result,
            )
            save_npz_arrays(case_dir / "case_arrays.npz", **arrays)
            _write_case_result_json(case_dir / "case_result.json", case, result)
        except Exception as exc:  # pragma: no cover - kept for stage manifest robustness.
            failed_cases.append(case.case_id)
            (case_dir / "case_result.json").write_text(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "case_status": "failed",
                        "error_code": type(exc).__name__,
                        "error_message": str(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            raise

    stage_summary = pd.concat(summary_frames, ignore_index=True)
    stage_spines = pd.concat(spines_frames, ignore_index=True)
    write_parquet(stage_summary, data_dir / "stage_summary.parquet")
    write_preview_csv(stage_summary, data_dir / "stage_summary_preview.csv")
    write_parquet(stage_spines, data_dir / "stage_spines.parquet")
    write_schema(stage_dir)
    manifest = create_manifest(
        project_name="Spine_Sim_V2",
        model_version="phase5_p1_single_case_sanity",
        surface_bank_id=bank.bank_id,
        surface_generator_version=None,
        probe_filter_version=None,
        random_seed_policy="P1 sanity is deterministic for a fixed surface bank and surface_id.",
        parameter_grid={
            "surface_bank": str(surface_bank),
            "surface_id": surface_id,
            "cases": [
                {
                    "case_id": case.case_id,
                    "array_type": case.array_type,
                    "rows": case.rows,
                    "cols": case.cols,
                    "pitch_t_mm": case.pitch_t_mm,
                    "pitch_l_mm": case.pitch_l_mm,
                    "alpha_p_deg": case.alpha_p_deg,
                    "spring_k_n_per_m": case.spring_k_n_per_m,
                    "w_total_n": case.w_total_n,
                }
                for case in cases
            ],
        },
        n_cases_expected=len(cases),
        n_cases_completed=len(cases) - len(failed_cases),
        failed_cases=failed_cases,
        notes=(
            "P1 single-case sanity stage. Case arrays are diagnostic copies; "
            "stage tables only reference surface_id."
        ),
    )
    write_manifest(manifest, stage_dir)
    _write_stage_report(reports_dir / "stage_report.md", stage_summary, stage_spines)
    return stage_dir


def _build_summary_record(
    *,
    case: SingleCaseInput,
    surface_bank: SurfaceBank,
    surface_record: dict[str, Any],
    stiffness: Any,
    preload: Any,
    spines_df: Any,
    loading: Any,
    phi_s_deg: float,
) -> dict[str, Any]:
    """汇总单 case 指标，并写入与阶段表一致的字段。"""
    n_nom = case.rows * case.cols
    n_con = int(spines_df["contacted"].sum())
    n_eng = int(spines_df["engaged"].sum())
    loads = spines_df["load_at_limit_n"].to_numpy(dtype=float)
    # Ω_eff 按模型定义为 0 < F_t,i ≤ cap，即在极限点真正承载的刺；
    # n_eff_count 与 n_eff_kish 统一以 load_at_limit_n > 0 为准，口径一致。
    effective_loads = loads[loads > 0.0]
    n_eff_count = int(effective_loads.size)
    load_sum = float(np.sum(loads))
    n_eff_kish = _kish_effective_count(loads)
    lsi = (
        float(np.max(effective_loads) / np.mean(effective_loads))
        if effective_loads.size > 0
        else None
    )
    spine_failure_modes = spines_df["failure_mode"].tolist()
    n_slip = sum(1 for mode in spine_failure_modes if mode == "slip")
    n_overload = sum(1 for mode in spine_failure_modes if mode == "overload")
    micro_damage = spines_df["micro_damage_risk"].dropna()
    search_distances = spines_df.loc[spines_df["engaged"], "search_distance_mm"].dropna().to_numpy(dtype=float)
    preloads = spines_df["preload_n"].to_numpy(dtype=float)
    normal_range_insufficient = bool(preload.normal_range_insufficient)
    failure_mode = _case_failure_mode(
        n_con=n_con,
        n_eng=n_eng,
        f_t_lim_n=loading.f_t_lim_n,
        w_total_n=case.w_total_n,
        normal_range_insufficient=normal_range_insufficient,
    )
    # 文档 §7.4：行程不足等情形记为 warning，并保留该 case，不外推为正常承载。
    case_status = "warning" if normal_range_insufficient else "completed"
    # 文档要求同时保存“接合成功”和“有效承载成功”，评分默认看 load_success。
    engagement_success = n_eng > 0 and case_status != "failed"
    load_threshold = max(0.01, 0.05 * case.w_total_n)
    load_success = (
        loading.f_t_lim_n >= load_threshold
        and n_eng > 0
        and not normal_range_insufficient
        and case_status != "failed"
    )
    warning_flags = list(preload.warning_flags)
    if failure_mode is not None and failure_mode not in {"capacity_limit"}:
        warning_flags.append(failure_mode)
    warning_flags = _dedupe_preserving_order(warning_flags)
    w_sat_mean_n = None
    if stiffness.spring_k_n_per_mm is not None:
        w_sat_mean_n = float(stiffness.spring_k_n_per_mm * 4.0 * np.sin(np.radians(case.alpha_p_deg)))

    return {
        "case_id": case.case_id,
        "stage": "p1_single_case",
        "case_status": case_status,
        "error_code": None,
        "warning_flags": warning_flags,
        "surface_bank_id": surface_bank.bank_id,
        "surface_id": case.surface_id,
        "surface_index_within_kind": None,
        "surface_kind": surface_record.get("surface_kind"),
        "surface_seed": surface_record.get("seed"),
        "candidate_id": case.candidate_id,
        "array_type": case.array_type,
        "rows": case.rows,
        "cols": case.cols,
        "n_nom": n_nom,
        "pitch_t_mm": case.pitch_t_mm,
        "pitch_l_mm": case.pitch_l_mm,
        "alpha_p_deg": case.alpha_p_deg,
        "spring_k_n_per_m": stiffness.spring_k_n_per_m,
        "spring_k_n_per_mm": stiffness.spring_k_n_per_mm,
        "k_nn_n_per_mm": stiffness.k_nn,
        "k_tt_n_per_mm": stiffness.k_tt,
        "k_tn_n_per_mm": stiffness.k_tn,
        "tip_radius_mm": case.tip_radius_mm,
        "spine_diameter_mm": case.spine_diameter_mm,
        "search_travel_mm": case.search_travel_mm,
        "w_total_n": case.w_total_n,
        "f_s": case.f_s,
        "phi_s_deg": phi_s_deg,
        "F_ref_star_n": case.F_ref_star_n,
        "trial_force_n": case.trial_force_n,
        "damage_pressure_threshold_n_per_mm2": case.damage_pressure_threshold_n_per_mm2,
        "surface_rq_raw_mm": surface_record.get("rq_raw_mm"),
        "surface_ra_raw_mm": surface_record.get("ra_raw_mm"),
        "surface_hpv_raw_mm": surface_record.get("hpv_raw_mm"),
        "surface_rq_eff_mm": surface_record.get("rq_eff_mm"),
        "surface_ra_eff_mm": surface_record.get("ra_eff_mm"),
        "surface_hpv_eff_mm": surface_record.get("hpv_eff_mm"),
        "surface_slope_mean_deg": surface_record.get("slope_mean_deg"),
        "surface_slope_p95_deg": surface_record.get("slope_p95_deg"),
        "candidate_density_preload_free": surface_record.get("candidate_density_preload_free"),
        "n_con": n_con,
        "n_eng": n_eng,
        "n_eff_count": n_eff_count,
        "n_eff_kish": n_eff_kish,
        "r_con": n_con / n_nom if n_nom else 0.0,
        "r_uncontacted": _fraction(n_nom - n_con, n_nom),
        "r_eng": n_eng / n_nom if n_nom else 0.0,
        "r_fail_search": _fraction(spines_df["state"].isin(["search_failed"]).sum(), n_nom),
        "search_distance_mean_mm": _mean_or_none(search_distances),
        "search_distance_p95_mm": _percentile_or_none(search_distances, 95.0),
        "normal_stroke_max_mm": stiffness.normal_stroke_max_mm,
        "u_ax_used_max_mm": _max_or_none(spines_df["u_ax_used_mm"].to_numpy(dtype=float)),
        "u_ax_used_mean_mm": _mean_or_none(spines_df["u_ax_used_mm"].to_numpy(dtype=float)),
        "w_sat_mean_n": w_sat_mean_n,
        "r_sat_n": _fraction(spines_df["normal_saturated"].sum(), n_nom),
        # 切向搜索行程饱和：刺走完全部 Δy_max 仍未接合，本版与搜索失败等价。
        "r_sat_y": _fraction(spines_df["state"].isin(["search_failed"]).sum(), n_nom),
        "normal_range_insufficient": normal_range_insufficient,
        "f_t_lim_n": float(loading.f_t_lim_n),
        "f_t_lim_over_w_total": loading.f_t_lim_n / case.w_total_n if case.w_total_n > 0.0 else None,
        "f_t_lim_per_nom_n": loading.f_t_lim_n / n_nom if n_nom else None,
        "f_t_lim_per_eff_n": loading.f_t_lim_n / n_eff_count if n_eff_count else None,
        "limit_displacement_mm": loading.limit_displacement_mm,
        "eta_max": float(np.max(loads) / load_sum) if load_sum > 0.0 else 0.0,
        "lsi": lsi,
        "w_cv": float(np.std(preloads) / np.mean(preloads)) if np.mean(preloads) > 0.0 else None,
        "engagement_success": engagement_success,
        "load_success": load_success,
        "failure_mode": failure_mode,
        "cascade_failure": bool(loading.cascade_failure),
        # slip / overload 按单刺容量受限模式区分统计，不再混为一谈。
        "r_slip": _fraction(n_slip, n_nom),
        "r_overload": _fraction(n_overload, n_nom),
        "r_side_contact_risk": _fraction(spines_df["side_contact_risk"].sum(), n_nom),
        "r_micro_damage_risk": (
            _fraction(int(micro_damage.astype(bool).sum()), n_nom)
            if len(micro_damage)
            else None
        ),
    }


def _default_sanity_cases(
    *,
    surface_bank_path: Path,
    surface_id: str,
    compliant_spring_k_n_per_m: float,
) -> list[SingleCaseInput]:
    common = {
        "surface_bank_path": surface_bank_path,
        "surface_id": surface_id,
        "rows": 2,
        "cols": 3,
        "pitch_t_mm": 4.0,
        "pitch_l_mm": 4.0,
        "alpha_p_deg": 60.0,
        "tip_radius_mm": 0.05,
        "spine_diameter_mm": 0.20,
        "search_travel_mm": 4.0,
        "w_total_n": 1.0,
        "f_s": 1.0,
        "F_ref_star_n": 0.50,
        "trial_force_n": 0.50,
    }
    return [
        SingleCaseInput(
            case_id="p1_rigid_baseline",
            candidate_id="p1_rigid_baseline",
            array_type="rigid",
            spring_k_n_per_m=None,
            **common,
        ),
        SingleCaseInput(
            case_id="p1_compliant_baseline",
            candidate_id="p1_compliant_baseline",
            array_type="compliant",
            spring_k_n_per_m=compliant_spring_k_n_per_m,
            **common,
        ),
    ]


def _build_case_arrays(
    *,
    bank: SurfaceBank,
    case: SingleCaseInput,
    result: SingleCaseResult,
) -> dict[str, Any]:
    """构造 P1 诊断数组；大规模筛选阶段不保存这些二维副本。"""
    surface_record = bank.get_surface_record(case.surface_id)
    surface_arrays = bank.load_surface_arrays(case.surface_id)
    height_raw = surface_arrays["height_raw"].astype(np.float32)
    height_filtered = surface_arrays["height_filtered"].astype(np.float32)
    dx_mm = float(surface_record["dx_mm"])
    dy_mm = float(surface_record["dy_mm"])
    slope_angle = _slope_angle_map_deg(height_filtered, dx_mm=dx_mm, dy_mm=dy_mm).astype(np.float32)
    effective_contact_angle = effective_contact_angle_map_deg(
        height_filtered,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
    ).astype(np.float32)
    spines = result.case_spines
    phi_c_values = spines["phi_c_deg"].dropna().to_numpy(dtype=float)
    phi_hook_values = spines["phi_hook_min_deg"].dropna().to_numpy(dtype=float)
    if phi_c_values.size:
        # P1 热区图只作人工诊断，真实接合仍以逐刺搜索结果为准。
        threshold = max(float(np.min(phi_c_values)), float(np.max(phi_hook_values)) if phi_hook_values.size else 0.0)
        engagement_candidate_mask = effective_contact_angle >= threshold
    else:
        engagement_candidate_mask = np.zeros_like(effective_contact_angle, dtype=bool)
    spine_x = spines["x_mm"].to_numpy(dtype=np.float32)
    spine_y = spines["y_mm"].to_numpy(dtype=np.float32)
    search_path_x, search_path_y = _build_search_path_arrays(
        spines=spines,
        search_travel_mm=case.search_travel_mm,
    )
    engagement_x = spines["engagement_x_mm"].to_numpy(dtype=np.float32)
    engagement_y = spines["engagement_y_mm"].to_numpy(dtype=np.float32)
    load_s = np.asarray(result.diagnostics.get("load_displacement_s", []), dtype=np.float32)
    load_f = np.asarray(result.diagnostics.get("load_displacement_f", []), dtype=np.float32)
    if load_s.size == 0:
        load_s = np.asarray([0.0], dtype=np.float32)
        load_f = np.asarray([0.0], dtype=np.float32)
    else:
        load_s = np.concatenate([np.asarray([0.0], dtype=np.float32), load_s])
        load_f = np.concatenate([np.asarray([0.0], dtype=np.float32), load_f])
    failure_sequence = spines["failure_order"].fillna(0).to_numpy(dtype=np.int32)
    return {
        "height_raw": height_raw,
        "height_filtered": height_filtered,
        "dx_mm": np.asarray(dx_mm, dtype=np.float32),
        "dy_mm": np.asarray(dy_mm, dtype=np.float32),
        "slope_angle": slope_angle,
        "effective_contact_angle": effective_contact_angle,
        "engagement_candidate_mask": engagement_candidate_mask.astype(bool),
        "spine_x": spine_x,
        "spine_y": spine_y,
        "search_path_x": search_path_x,
        "search_path_y": search_path_y,
        "engagement_x": engagement_x,
        "engagement_y": engagement_y,
        "load_displacement_s": load_s,
        "load_displacement_f": load_f,
        "failure_sequence": failure_sequence,
    }


def _build_search_path_arrays(*, spines: Any, search_travel_mm: float) -> tuple[np.ndarray, np.ndarray]:
    n_spines = len(spines)
    n_samples = max(2, int(np.ceil(search_travel_mm / 0.1)) + 1)
    s = np.linspace(0.0, search_travel_mm, n_samples, dtype=np.float32)
    x0 = spines["x_mm"].to_numpy(dtype=np.float32)[:, None]
    y0 = spines["y_mm"].to_numpy(dtype=np.float32)[:, None]
    return x0 + s[None, :], np.repeat(y0, n_samples, axis=1)


def _slope_angle_map_deg(
    height_filtered: np.ndarray,
    *,
    dx_mm: float,
    dy_mm: float,
) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(np.asarray(height_filtered, dtype=float), dy_mm, dx_mm)
    return np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))


def _write_case_result_json(
    path: Path,
    case: SingleCaseInput,
    result: SingleCaseResult,
) -> None:
    payload = {
        "case_input": _json_sanitize(case.__dict__),
        "summary": _json_sanitize(result.case_summary.iloc[0].to_dict()),
        "diagnostics": _json_sanitize(result.diagnostics),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_stage_report(path: Path, stage_summary: Any, stage_spines: Any) -> None:
    rows = [
        "# P1 Single Case Sanity Report",
        "",
        "This report is generated from saved P1 sanity data products.",
        "",
        "## Case Summary",
        "",
        "| case_id | array_type | n_con | n_eng | f_t_lim_n | engagement_success | load_success |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in stage_summary.iterrows():
        rows.append(
            "| {case_id} | {array_type} | {n_con} | {n_eng} | {f_t_lim_n:.6g} | {engagement_success} | {load_success} |".format(
                case_id=row["case_id"],
                array_type=row["array_type"],
                n_con=int(row["n_con"]),
                n_eng=int(row["n_eng"]),
                f_t_lim_n=float(row["f_t_lim_n"]),
                engagement_success=bool(row["engagement_success"]),
                load_success=bool(row["load_success"]),
            )
        )
    rows.extend(
        [
            "",
            "## Spine Records",
            "",
            f"Total spine rows: {len(stage_spines)}",
            "",
            "Figures are generated separately by `python scripts/plot_results.py p1 ...`.",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, np.generic):
        return _json_sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _validate_case(case: SingleCaseInput) -> None:
    if case.array_type not in {"rigid", "compliant"}:
        raise ValueError("array_type must be 'rigid' or 'compliant'.")
    if case.array_type == "rigid" and case.spring_k_n_per_m is not None:
        raise ValueError("Rigid arrays must use spring_k_n_per_m=None.")
    if case.array_type == "compliant" and case.spring_k_n_per_m is None:
        raise ValueError("Compliant arrays require spring_k_n_per_m.")
    if case.rows <= 0 or case.cols <= 0:
        raise ValueError("rows and cols must be positive.")
    for name in (
        "pitch_t_mm",
        "pitch_l_mm",
        "tip_radius_mm",
        "spine_diameter_mm",
        "search_travel_mm",
        "w_total_n",
        "F_ref_star_n",
        "trial_force_n",
    ):
        if getattr(case, name) < 0.0:
            raise ValueError(f"{name} must be non-negative.")
    if case.pitch_t_mm == 0.0 or case.pitch_l_mm == 0.0:
        raise ValueError("pitch_t_mm and pitch_l_mm must be positive.")
    if (
        case.damage_pressure_threshold_n_per_mm2 is not None
        and case.damage_pressure_threshold_n_per_mm2 < 0.0
    ):
        raise ValueError("damage_pressure_threshold_n_per_mm2 must be non-negative when provided.")


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "P1 single-case simulation requires pandas. Install dependencies with "
            "`python3 -m pip install -e .`."
        ) from exc
    return pd


def _ensure_schema_columns(df: Any, field_names: set[str]) -> None:
    missing = sorted(field_names - set(df.columns))
    if missing:
        raise RuntimeError(f"Output table is missing schema fields: {missing}")


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _mean_or_none(values: Any) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else None


def _max_or_none(values: Any) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else None


def _percentile_or_none(values: Any, percentile: float) -> float | None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, percentile)) if arr.size else None


def _fraction(count: Any, denom: int) -> float:
    return float(count) / denom if denom else 0.0


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    """去重但保持首次出现顺序。"""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _kish_effective_count(loads: Any) -> float:
    arr = np.asarray(loads, dtype=float)
    arr = arr[arr > 0.0]
    denom = float(np.sum(arr**2))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(arr) ** 2 / denom)


def _case_failure_mode(
    *,
    n_con: int,
    n_eng: int,
    f_t_lim_n: float,
    w_total_n: float,
    normal_range_insufficient: bool,
) -> str | None:
    if normal_range_insufficient:
        return "normal_range_insufficient"
    if n_con <= 0:
        return "no_contact"
    if n_eng <= 0:
        return "search_failed"
    if f_t_lim_n < max(0.01, 0.05 * w_total_n):
        return "load_below_threshold"
    return None
