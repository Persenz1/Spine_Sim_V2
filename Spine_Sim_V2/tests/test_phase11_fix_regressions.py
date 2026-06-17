"""验收修复回归测试：失败 case 保留、slip/overload 区分、容量边界、并行一致性。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from Spine_Sim_V2.core.types import SingleCaseInput
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.io.schema_io import read_schema
from Spine_Sim_V2.pipelines.p2_compliant_k_alpha import run as run_p2
from Spine_Sim_V2.pipelines.p1_single_case import run_single_case
from Spine_Sim_V2.pipelines.p6_final_mc import run as run_p6
from Spine_Sim_V2.solvers.engagement import compute_capacity_n
from Spine_Sim_V2.solvers.loading import run_loading_sequence
from Spine_Sim_V2.surfaces.generator import probe_filter
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _make_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase11",
        surface_kinds="sandpaper,concrete,brick,painted_wall",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=12.0,
        size_y_mm=8.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase11",
    )


def test_capacity_boundaries_are_finite_and_unscaled():
    # 自锁区强度受限，且不随 W_total 放大
    for w in (0.5, 1.0, 5.0):
        result = compute_capacity_n(preload_n=w, phi_eng_deg=89.0, f_s=1.0, F_ref_star_n=0.5)
        assert result.cap_mode == "self_lock_strength"
        assert result.cap_n == pytest.approx(0.5)
        assert np.isfinite(result.cap_n)
    # beta <= 0 -> 无几何承载
    no_geom = compute_capacity_n(preload_n=1.0, phi_eng_deg=-50.0, f_s=1.0, F_ref_star_n=0.5)
    assert no_geom.cap_n == 0.0
    assert no_geom.cap_mode == "no_geometric_engagement"
    # W_i = 0 -> 无接合
    none = compute_capacity_n(preload_n=0.0, phi_eng_deg=80.0, f_s=1.0, F_ref_star_n=0.5)
    assert none.cap_n == 0.0
    assert none.cap_mode == "none"
    # F* 很小 -> 强度受限到 F*
    tiny = compute_capacity_n(preload_n=1.0, phi_eng_deg=30.0, f_s=1.0, F_ref_star_n=1e-4)
    assert tiny.cap_mode == "strength" and tiny.cap_n == pytest.approx(1e-4)
    # F* 很大 -> 几何受限
    huge = compute_capacity_n(preload_n=1.0, phi_eng_deg=30.0, f_s=1.0, F_ref_star_n=1e6)
    assert huge.cap_mode == "geom_friction" and np.isfinite(huge.cap_n)


def test_loading_distinguishes_slip_from_overload():
    # 几何受限 -> slip；强度受限 -> overload
    result = run_loading_sequence(
        engaged=np.array([True, True]),
        search_distance_mm=np.array([0.0, 0.0]),
        cap_n=np.array([0.3, 0.5]),
        k_tt_n_per_mm=1.0,
        cap_mode=["geom_friction", "strength"],
    )
    assert result.failure_mode[0] == "slip"
    assert result.failure_mode[1] == "overload"


def test_single_case_r_slip_and_r_overload_partition_failures(tmp_path):
    bank = _make_bank(tmp_path)
    case = SingleCaseInput(
        surface_bank_path=bank.root,
        surface_id="concrete_000000",
        array_type="compliant",
        rows=2,
        cols=3,
        pitch_t_mm=2.0,
        pitch_l_mm=2.0,
        alpha_p_deg=60.0,
        spring_k_n_per_m=330.0,
        tip_radius_mm=0.05,
        spine_diameter_mm=0.20,
        search_travel_mm=2.0,
        w_total_n=1.0,
        f_s=1.0,
        F_ref_star_n=0.50,
        trial_force_n=0.50,
        case_id="slip_overload",
        candidate_id="slip_overload",
    )
    result = run_single_case(case)
    spines = result.case_spines
    summary = result.case_summary.iloc[0]
    n_nom = case.rows * case.cols
    n_slip = int((spines["failure_mode"] == "slip").sum())
    n_overload = int((spines["failure_mode"] == "overload").sum())
    assert summary["r_slip"] == pytest.approx(n_slip / n_nom)
    assert summary["r_overload"] == pytest.approx(n_overload / n_nom)
    # 失效刺的 state 采用文档枚举（slip/overload），不再是 failed_*
    failed_states = set(spines.loc[spines["failed"], "state"].tolist())
    assert failed_states <= {"slip", "overload"}
    # 新增刚度投影与 lsi 字段
    assert summary["k_nn_n_per_mm"] is not None
    assert "contact_pressure_proxy_n_per_mm2" in spines.columns
    positive = spines.loc[spines["preload_n"] > 0.0].iloc[0]
    expected_phi_c = np.degrees(np.arctan(case.trial_force_n / positive["preload_n"])) - 45.0
    assert positive["phi_c_deg"] == pytest.approx(expected_phi_c)


def test_probe_filter_is_pointwise_closing_without_mean_shift():
    raw = np.array(
        [
            [1.0, -2.0, 1.0],
            [0.0, -3.0, 0.0],
            [1.0, -2.0, 1.0],
        ],
        dtype=float,
    )
    raw -= raw.mean()
    eff = probe_filter(raw, 0.2, dx_mm=0.2, dy_mm=0.2)
    assert np.min(eff - raw) >= -1e-7


def test_micro_damage_risk_threshold_is_recorded(tmp_path):
    bank = _make_bank(tmp_path)
    case = SingleCaseInput(
        surface_bank_path=bank.root,
        surface_id="concrete_000000",
        array_type="compliant",
        rows=2,
        cols=3,
        pitch_t_mm=2.0,
        pitch_l_mm=2.0,
        alpha_p_deg=60.0,
        spring_k_n_per_m=330.0,
        tip_radius_mm=0.05,
        spine_diameter_mm=0.20,
        search_travel_mm=2.0,
        w_total_n=1.0,
        f_s=1.0,
        F_ref_star_n=0.50,
        trial_force_n=0.50,
        damage_pressure_threshold_n_per_mm2=0.0,
        case_id="damage_risk",
        candidate_id="damage_risk",
    )
    result = run_single_case(case)
    spines = result.case_spines
    contacted = spines["preload_n"] > 0.0
    assert spines.loc[contacted, "micro_damage_risk"].eq(True).all()
    assert spines.loc[~contacted, "micro_damage_risk"].eq(False).all()
    assert result.case_summary.iloc[0]["r_micro_damage_risk"] == pytest.approx(contacted.sum() / len(spines))


def test_p6_retains_failed_cases_without_crashing(tmp_path):
    bank = _make_bank(tmp_path)
    candidates = [
        {
            "candidate_id": "GOOD",
            "array_type": "rigid",
            "rows": 2,
            "cols": 2,
            "pitch_t_mm": 4.0,
            "pitch_l_mm": 4.0,
            "alpha_p_deg": 60.0,
            "spring_k_n_per_m": None,
        },
        {
            "candidate_id": "BAD",
            "array_type": "bogus_type",  # 触发 worker 内异常
            "rows": 2,
            "cols": 2,
            "pitch_t_mm": 4.0,
            "pitch_l_mm": 4.0,
            "alpha_p_deg": 60.0,
            "spring_k_n_per_m": None,
        },
    ]
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(candidates), encoding="utf-8")

    p6_dir = run_p6(
        surface_bank=bank.root,
        selected_candidates=selected,
        n_surfaces_per_kind=1,
        surface_selection="first_n",
        outdir=tmp_path / "P6",
        workers=2,
        w_values=(0.5, 1.0),
    )
    summary = read_parquet(p6_dir / "data" / "final_summary.parquet")
    # 2 候选 × 4 类 × 1 表面 × 2 预载 = 16 行，失败 case 不丢、不崩
    assert len(summary) == 2 * 4 * 1 * 2
    failed = summary.loc[summary["case_status"] == "failed"]
    completed = summary.loc[summary["case_status"] == "completed"]
    assert len(failed) == 8
    assert len(completed) == 8
    # 修复点：失败行 surface_bank_id 非空（旧实现会因非空约束崩溃整轮）
    assert failed["surface_bank_id"].notna().all()
    manifest = json.loads((p6_dir / "manifest.json").read_text())
    assert len(manifest["failed_cases"]) == 8
    assert manifest["n_cases_completed"] == 8


def test_stage_schema_tracks_dynamic_grouped_statistics(tmp_path):
    bank = _make_bank(tmp_path)
    stage_dir = run_p2(
        surface_bank=bank.root,
        n_surfaces_per_kind=1,
        outdir=tmp_path / "P2",
        spring_values=(330.0,),
        alpha_values=(60.0,),
        w_values=(0.5,),
        workers=1,
    )
    grouped = read_parquet(stage_dir / "data" / "stage_grouped_statistics.parquet")
    schema = read_schema(stage_dir)
    schema_fields = {field["name"] for field in schema["schemas"]["stage_grouped_statistics"]}
    assert set(grouped.columns) <= schema_fields
    assert {"f_t_lim_n_std", "f_t_lim_n_min", "f_t_lim_n_max", "r_slip_mean", "r_overload_mean"} <= schema_fields
