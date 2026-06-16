"""仿真工程共享数据类型与落盘表结构定义。

本模块只定义可复用的数据容器和 schema 元数据，不承担物理计算。
字段名保持英文和单位后缀，是为了让 Parquet/JSON 输出在脚本之间稳定传递。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseStatus:
    """每个仿真 case 都要记录的标准状态字段。"""

    case_status: str = "not_run"
    error_code: str | None = None
    warning_flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SurfaceRef:
    """指向 surface bank 中某个表面的轻量引用。"""

    surface_bank_id: str
    surface_id: str


@dataclass(frozen=True)
class SingleCaseInput:
    """单个 case 仿真所需的全部输入参数。"""

    surface_bank_path: str | Path
    surface_id: str
    array_type: str
    rows: int
    cols: int
    pitch_t_mm: float
    pitch_l_mm: float
    alpha_p_deg: float
    spring_k_n_per_m: float | None
    tip_radius_mm: float
    spine_diameter_mm: float
    search_travel_mm: float
    w_total_n: float
    f_s: float
    F_ref_star_n: float
    trial_force_n: float
    candidate_id: str = "candidate_000"
    case_id: str = "case_000"


@dataclass(frozen=True)
class StiffnessModel:
    """某个阵列配置投影到法向/切向后的刚度参数。"""

    spring_k_n_per_m: float | None
    spring_k_n_per_mm: float | None
    k_nn: float | None
    k_tt: float | None
    k_tn: float | None
    axial_stroke_max_mm: float | None
    normal_stroke_max_mm: float | None


@dataclass(frozen=True)
class SingleCaseResult:
    """单 case 的输出表格和不强制落盘的诊断信息。"""

    case_summary: Any
    case_spines: Any
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchemaField:
    """落盘表 schema 中的单个字段说明。"""

    name: str
    dtype: str
    unit: str | None
    nullable: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        """转换成可直接写入 JSON 的字典。"""
        return asdict(self)


def schema_to_dicts(schema: tuple[SchemaField, ...]) -> list[dict[str, Any]]:
    """把 schema 元组转换成 JSON 友好的字典列表。"""
    return [field_.to_dict() for field_ in schema]


def schema_field_names(schema: tuple[SchemaField, ...]) -> set[str]:
    """提取字段名集合，用于快速校验输出表是否完整。"""
    return {field_.name for field_ in schema}


def _field(
    name: str,
    dtype: str,
    unit: str | None = None,
    nullable: bool = False,
    description: str | None = None,
) -> SchemaField:
    return SchemaField(
        name=name,
        dtype=dtype,
        unit=unit,
        nullable=nullable,
        description=description or name.replace("_", " "),
    )


stage_summary_schema: tuple[SchemaField, ...] = (
    _field("case_id", "string", description="Unique case identifier."),
    _field("stage", "string", description="Pipeline stage identifier."),
    _field("case_status", "string", description="Case execution status."),
    _field("error_code", "string", nullable=True, description="Machine-readable error code."),
    _field("warning_flags", "list[string]", description="Warnings emitted during the case."),
    _field("surface_bank_id", "string", description="Surface bank identifier."),
    _field("surface_id", "string", description="Surface identifier within the bank."),
    _field(
        "surface_index_within_kind",
        "int64",
        nullable=True,
        description="Zero-based selected surface index within its surface_kind for Monte Carlo convergence.",
    ),
    _field("surface_kind", "string", description="Surface profile family or source kind."),
    _field("surface_seed", "int64", nullable=True, description="Random seed used for the surface."),
    _field("candidate_id", "string", description="Candidate geometry or parameter set identifier."),
    _field("array_type", "string", description="Array structure type, such as rigid or compliant."),
    _field("rows", "int64", description="Number of spine rows."),
    _field("cols", "int64", description="Number of spine columns."),
    _field("n_nom", "int64", description="Nominal spine count."),
    _field("pitch_t_mm", "float64", "mm", description="Tangential pitch."),
    _field("pitch_l_mm", "float64", "mm", description="Lateral pitch."),
    _field("alpha_p_deg", "float64", "deg", description="Spine pitch installation angle."),
    _field(
        "spring_k_n_per_m",
        "float64",
        "N/m",
        nullable=True,
        description="Input axial spring stiffness; null for rigid arrays.",
    ),
    _field(
        "spring_k_n_per_mm",
        "float64",
        "N/mm",
        nullable=True,
        description="Internal axial spring stiffness; null for rigid arrays.",
    ),
    _field("tip_radius_mm", "float64", "mm", description="Spine tip radius."),
    _field("spine_diameter_mm", "float64", "mm", description="Spine shaft diameter."),
    _field("search_travel_mm", "float64", "mm", description="Available tangential search travel."),
    _field("w_total_n", "float64", "N", description="Total normal preload magnitude."),
    _field("f_s", "float64", description="Static friction coefficient at the tip-surface contact."),
    _field("phi_s_deg", "float64", "deg", description="Friction angle."),
    _field("F_ref_star_n", "float64", "N", description="Reference calibrated spine-surface strength."),
    _field("trial_force_n", "float64", "N", description="Trial tangential force for success checks."),
    _field("surface_rq_raw_mm", "float64", "mm", description="Raw surface RMS roughness."),
    _field("surface_ra_raw_mm", "float64", "mm", description="Raw surface mean absolute roughness."),
    _field("surface_hpv_raw_mm", "float64", "mm", description="Raw surface peak-to-valley height."),
    _field("surface_rq_eff_mm", "float64", "mm", description="Filtered effective RMS roughness."),
    _field("surface_ra_eff_mm", "float64", "mm", description="Filtered effective mean absolute roughness."),
    _field("surface_hpv_eff_mm", "float64", "mm", description="Filtered effective peak-to-valley height."),
    _field("surface_slope_mean_deg", "float64", "deg", description="Mean effective surface slope."),
    _field("surface_slope_p95_deg", "float64", "deg", description="95th percentile effective surface slope."),
    _field(
        "candidate_density_preload_free",
        "float64",
        "1/mm",
        description="Preload-free candidate engagement density along search paths.",
    ),
    _field("n_con", "int64", description="Number of contacted spines."),
    _field("n_eng", "int64", description="Number of engaged spines."),
    _field("n_eff_count", "int64", description="Count of load-bearing effective spines."),
    _field("n_eff_kish", "float64", description="Kish effective load-bearing spine count."),
    _field("r_con", "float64", description="Contact ratio relative to nominal spine count."),
    _field("r_eng", "float64", description="Engagement ratio relative to nominal spine count."),
    _field("r_fail_search", "float64", description="Fraction that failed finite search."),
    _field("search_distance_mean_mm", "float64", "mm", nullable=True, description="Mean search distance."),
    _field("search_distance_p95_mm", "float64", "mm", nullable=True, description="95th percentile search distance."),
    _field("normal_stroke_max_mm", "float64", "mm", nullable=True, description="Maximum required normal stroke."),
    _field("u_ax_used_max_mm", "float64", "mm", nullable=True, description="Maximum axial spring compression used."),
    _field("u_ax_used_mean_mm", "float64", "mm", nullable=True, description="Mean axial spring compression used."),
    _field("w_sat_mean_n", "float64", "N", nullable=True, description="Mean preload saturation force."),
    _field("r_sat_n", "float64", nullable=True, description="Normal saturation ratio."),
    _field("r_sat_y", "float64", nullable=True, description="Tangential or stroke saturation ratio."),
    _field(
        "normal_range_insufficient",
        "bool",
        description="True when normal travel is insufficient for the case.",
    ),
    _field("f_t_lim_n", "float64", "N", nullable=True, description="Total tangential load limit."),
    _field(
        "f_t_lim_over_w_total",
        "float64",
        nullable=True,
        description="Tangential load limit normalized by total normal preload.",
    ),
    _field("f_t_lim_per_nom_n", "float64", "N", nullable=True, description="Load limit per nominal spine."),
    _field("f_t_lim_per_eff_n", "float64", "N", nullable=True, description="Load limit per effective spine."),
    _field("limit_displacement_mm", "float64", "mm", nullable=True, description="Tangential displacement at limit."),
    _field("eta_max", "float64", nullable=True, description="Maximum single-spine load share."),
    _field("w_cv", "float64", nullable=True, description="Coefficient of variation of local normal preload."),
    _field("engagement_success", "bool", description="True when engagement success criterion is met."),
    _field("load_success", "bool", description="True when load success criterion is met."),
    _field("failure_mode", "string", nullable=True, description="Dominant failure mode."),
    _field("cascade_failure", "bool", description="True when event-driven cascade failure occurs."),
    _field("r_slip", "float64", nullable=True, description="Fraction failing by slip."),
    _field("r_overload", "float64", nullable=True, description="Fraction failing by overload."),
    _field("r_side_contact_risk", "float64", nullable=True, description="Fraction with side contact risk."),
)


stage_spines_schema: tuple[SchemaField, ...] = (
    _field("case_id", "string", description="Unique case identifier."),
    _field("candidate_id", "string", description="Candidate geometry or parameter set identifier."),
    _field("surface_id", "string", description="Surface identifier within the bank."),
    _field("spine_id", "string", description="Unique spine identifier within a case."),
    _field("row", "int64", description="Spine row index."),
    _field("col", "int64", description="Spine column index."),
    _field("array_type", "string", description="Array structure type."),
    _field("x_mm", "float64", "mm", description="Initial spine x coordinate."),
    _field("y_mm", "float64", "mm", description="Initial spine y coordinate."),
    _field("gap_mm", "float64", "mm", nullable=True, description="Initial normal gap."),
    _field("alpha_p_deg", "float64", "deg", description="Spine pitch installation angle."),
    _field("pitch_t_mm", "float64", "mm", description="Tangential pitch."),
    _field("pitch_l_mm", "float64", "mm", description="Lateral pitch."),
    _field("contacted", "bool", description="True when the spine contacts the surface."),
    _field("preload_n", "float64", "N", nullable=True, description="Local normal preload magnitude."),
    _field("u_ax_used_mm", "float64", "mm", nullable=True, description="Axial compression used."),
    _field("normal_saturated", "bool", description="True when normal compliance saturates."),
    _field("state", "string", description="Final spine state."),
    _field("search_distance_mm", "float64", "mm", nullable=True, description="Search distance to engagement."),
    _field("engaged", "bool", description="True when the spine engages."),
    _field("engagement_x_mm", "float64", "mm", nullable=True, description="Engagement x coordinate."),
    _field("engagement_y_mm", "float64", "mm", nullable=True, description="Engagement y coordinate."),
    _field("phi_c_deg", "float64", "deg", nullable=True, description="Critical engagement angle."),
    _field("phi_eng_deg", "float64", "deg", nullable=True, description="Effective angle at engagement."),
    _field("phi_hook_min_deg", "float64", "deg", nullable=True, description="Minimum hook angle along path."),
    _field("side_contact_risk", "bool", description="True when side contact risk is detected."),
    _field("cap_n", "float64", "N", nullable=True, description="Single-spine tangential capacity."),
    _field("cap_mode", "string", nullable=True, description="Capacity limiting mode."),
    _field("load_at_limit_n", "float64", "N", nullable=True, description="Load carried at global limit."),
    _field("failed", "bool", description="True when the spine has failed by the limit event."),
    _field("failure_mode", "string", nullable=True, description="Spine failure mode."),
    _field("failure_order", "int64", nullable=True, description="Event order of failure."),
)


stage_grouped_statistics_schema: tuple[SchemaField, ...] = (
    _field("stage", "string", description="Pipeline stage identifier."),
    _field("group_id", "string", description="Grouped statistics identifier."),
    _field("group_by", "list[string]", description="Fields used to define the group."),
    _field("candidate_id", "string", nullable=True, description="Candidate identifier if grouped by candidate."),
    _field("surface_kind", "string", nullable=True, description="Surface family if grouped by surface kind."),
    _field("n_cases", "int64", description="Number of cases in the group."),
    _field("n_success", "int64", description="Number of successful cases in the group."),
    _field("success_rate", "float64", description="Success fraction."),
    _field("f_t_lim_n_mean", "float64", "N", nullable=True, description="Mean tangential load limit."),
    _field("f_t_lim_n_median", "float64", "N", nullable=True, description="Median tangential load limit."),
    _field("f_t_lim_n_p05", "float64", "N", nullable=True, description="5th percentile tangential load limit."),
    _field("f_t_lim_n_p95", "float64", "N", nullable=True, description="95th percentile tangential load limit."),
    _field("n_eff_kish_mean", "float64", nullable=True, description="Mean Kish effective spine count."),
    _field("eta_max_mean", "float64", nullable=True, description="Mean maximum single-spine load share."),
    _field("r_fail_search_mean", "float64", nullable=True, description="Mean finite-search failure fraction."),
)


stage_rankings_schema: tuple[SchemaField, ...] = (
    _field("stage", "string", description="Pipeline stage identifier."),
    _field("rank", "int64", description="One-based ranking position."),
    _field("candidate_id", "string", description="Candidate identifier."),
    _field("score_total", "float64", description="Overall screening score."),
    _field("score_load", "float64", nullable=True, description="Load capacity score component."),
    _field("score_success", "float64", nullable=True, description="Success robustness score component."),
    _field("score_uniformity", "float64", nullable=True, description="Load sharing uniformity score component."),
    _field("score_search", "float64", nullable=True, description="Search efficiency score component."),
    _field("n_cases", "int64", description="Number of cases supporting the ranking."),
    _field("notes", "string", nullable=True, description="Ranking notes."),
)


surface_statistics_schema: tuple[SchemaField, ...] = (
    _field("surface_bank_id", "string", description="Surface bank identifier."),
    _field("surface_id", "string", description="Surface identifier within the bank."),
    _field("surface_kind", "string", description="Proxy surface profile family."),
    _field("seed", "int64", description="Random seed used for the surface."),
    _field("dx_mm", "float64", "mm", description="Grid spacing along x."),
    _field("dy_mm", "float64", "mm", description="Grid spacing along y."),
    _field("size_x_mm", "float64", "mm", description="Surface window size along x."),
    _field("size_y_mm", "float64", "mm", description="Surface window size along y."),
    _field("tip_radius_mm", "float64", "mm", description="Probe filter tip radius."),
    _field("rq_raw_mm", "float64", "mm", description="Raw surface RMS roughness."),
    _field("ra_raw_mm", "float64", "mm", description="Raw surface mean absolute roughness."),
    _field("hpv_raw_mm", "float64", "mm", description="Raw surface peak-to-valley height."),
    _field("rq_eff_mm", "float64", "mm", description="Filtered effective RMS roughness."),
    _field("ra_eff_mm", "float64", "mm", description="Filtered effective mean absolute roughness."),
    _field("hpv_eff_mm", "float64", "mm", description="Filtered effective peak-to-valley height."),
    _field("slope_mean_deg", "float64", "deg", description="Mean effective surface slope."),
    _field("slope_p50_deg", "float64", "deg", description="Median effective surface slope."),
    _field("slope_p95_deg", "float64", "deg", description="95th percentile effective surface slope."),
    _field("slope_max_deg", "float64", "deg", description="Maximum effective surface slope."),
    _field(
        "candidate_density_preload_free",
        "float64",
        "1/mm^2",
        description="Geometry-only local peak density for bank audit; not an engagement decision.",
    ),
    _field("valid", "bool", description="True when generated arrays and statistics are valid."),
    _field("reject_reason", "string", nullable=True, description="Reason for rejecting the generated surface."),
)


SCHEMA_REGISTRY: dict[str, tuple[SchemaField, ...]] = {
    "stage_summary": stage_summary_schema,
    "stage_spines": stage_spines_schema,
    "stage_grouped_statistics": stage_grouped_statistics_schema,
    "stage_rankings": stage_rankings_schema,
    "surface_statistics": surface_statistics_schema,
}
