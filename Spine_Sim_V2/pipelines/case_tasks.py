"""跨进程并行的 case 任务封装与失败 case 保留。

所有筛选阶段（P2/P3/P5/P6）共用同一条物理求解链。本模块把"一个 case 的输入 +
落盘所需的阶段元数据"打包成可 pickle 的 :class:`CaseJob`，并提供：

- :func:`run_case_job`：在 worker 进程内执行单 case，**异常不外抛**，而是转成一行
  完整的失败 summary（不是静默丢弃，也不会让整轮仿真崩溃）；
- :func:`build_failed_summary_record`：失败行保留 ``surface_bank_id`` 等关键字段，
  修复"失败 case 写入因 surface_bank_id 非空约束而崩溃"的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from Spine_Sim_V2.core.types import SingleCaseInput, stage_summary_schema
from Spine_Sim_V2.pipelines.p1_single_case import simulate_case_records


@dataclass(frozen=True)
class CaseJob:
    """一个待执行 case 及其阶段落盘元数据。"""

    case: SingleCaseInput
    surface_bank_id: str
    stage: str | None = None
    surface_kind: str | None = None
    surface_index_within_kind: int | None = None


def run_case_job(job: CaseJob) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    """执行单个 case 任务，返回 ``(summary_record, spine_records, failed_case_id)``。

    成功时 ``failed_case_id`` 为 ``None``；失败时返回保留字段完整的失败 summary，
    并把 ``case_id`` 作为 ``failed_case_id`` 回传，供 manifest 记录。
    """
    try:
        summary, spines, _diagnostics = simulate_case_records(job.case)
        summary["surface_bank_id"] = job.surface_bank_id
        if job.stage is not None:
            summary["stage"] = job.stage
        if job.surface_kind is not None and not summary.get("surface_kind"):
            summary["surface_kind"] = job.surface_kind
        if job.surface_index_within_kind is not None:
            summary["surface_index_within_kind"] = job.surface_index_within_kind
        return summary, spines, None
    except Exception as exc:  # 失败 case 必须保留为一行，不静默丢弃，也不崩溃整轮。
        return build_failed_summary_record(job, exc), [], job.case.case_id


def build_failed_summary_record(job: CaseJob, exc: Exception) -> dict[str, Any]:
    """为失败 case 构造保留字段完整、可空字段安全的 summary 行。"""
    record: dict[str, Any] = {field.name: None for field in stage_summary_schema}
    case = job.case
    spring_k_n_per_mm = None if case.spring_k_n_per_m is None else float(case.spring_k_n_per_m) / 1000.0
    record.update(
        {
            "case_id": case.case_id,
            "stage": job.stage,
            "case_status": "failed",
            "error_code": type(exc).__name__,
            "warning_flags": [f"{type(exc).__name__}: {exc}"],
            "surface_bank_id": job.surface_bank_id,
            "surface_id": case.surface_id,
            "surface_kind": job.surface_kind,
            "surface_index_within_kind": job.surface_index_within_kind,
            "candidate_id": case.candidate_id,
            "array_type": case.array_type,
            "rows": case.rows,
            "cols": case.cols,
            "n_nom": case.rows * case.cols,
            "pitch_t_mm": case.pitch_t_mm,
            "pitch_l_mm": case.pitch_l_mm,
            "alpha_p_deg": case.alpha_p_deg,
            "spring_k_n_per_m": case.spring_k_n_per_m,
            "spring_k_n_per_mm": spring_k_n_per_mm,
            "tip_radius_mm": case.tip_radius_mm,
            "spine_diameter_mm": case.spine_diameter_mm,
            "search_travel_mm": case.search_travel_mm,
            "w_total_n": case.w_total_n,
            "f_s": case.f_s,
            "F_ref_star_n": case.F_ref_star_n,
            "trial_force_n": case.trial_force_n,
            "damage_pressure_threshold_n_per_mm2": case.damage_pressure_threshold_n_per_mm2,
            "n_con": 0,
            "n_eng": 0,
            "n_eff_count": 0,
            "n_eff_kish": 0.0,
            "r_con": 0.0,
            "r_uncontacted": 1.0,
            "r_eng": 0.0,
            "r_fail_search": 1.0,
            "normal_range_insufficient": False,
            "f_t_lim_n": 0.0,
            "f_t_lim_over_w_total": 0.0,
            "eta_max": 0.0,
            "engagement_success": False,
            "load_success": False,
            "failure_mode": "case_failed",
            "cascade_failure": False,
            "r_slip": 0.0,
            "r_overload": 0.0,
            "r_side_contact_risk": 0.0,
            "r_micro_damage_risk": None,
        }
    )
    return record
