from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from Spine_Sim_V2.analysis.final_mc import run_p7_surface_generalization, run_p8_preload_efficiency
from Spine_Sim_V2.core.types import SingleCaseInput
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.pipelines.p1_single_case import run_single_case, run_single_case_sanity
from Spine_Sim_V2.pipelines.p2_compliant_k_alpha import run as run_p2
from Spine_Sim_V2.pipelines.p3_rigid_alpha import run as run_p3
from Spine_Sim_V2.pipelines.p5_array_screen import run_p5a, run_p5b
from Spine_Sim_V2.pipelines.p6_final_mc import run as run_p6
from Spine_Sim_V2.plotting.plot_case import plot_p1_single_case_sanity
from Spine_Sim_V2.plotting.plot_stage import plot_stage
from Spine_Sim_V2.solvers.contact import (
    build_stiffness_model,
    solve_preload_distribution,
)
from Spine_Sim_V2.surfaces.bank import SurfaceBank, create_surface_bank


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


SURFACE_KINDS = ("sandpaper", "concrete", "brick", "painted_wall")


def _make_smoke_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase10_smoke",
        surface_kinds=SURFACE_KINDS,
        n_per_kind=3,
        resolution_cells_per_mm=5,
        size_x_mm=24.0,
        size_y_mm=18.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase10_smoke",
    )


def test_phase10_preload_root_solve_and_success_criteria(tmp_path):
    bank = create_surface_bank(
        bank_id="surface_bank_phase10_case",
        surface_kinds="sandpaper",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=12.0,
        size_y_mm=8.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase10_case",
    )
    case = SingleCaseInput(
        surface_bank_path=bank.root,
        surface_id="sandpaper_000000",
        array_type="compliant",
        rows=1,
        cols=3,
        pitch_t_mm=2.0,
        pitch_l_mm=2.0,
        alpha_p_deg=60.0,
        spring_k_n_per_m=1000.0,
        tip_radius_mm=0.05,
        spine_diameter_mm=0.20,
        search_travel_mm=2.0,
        w_total_n=0.75,
        f_s=1.0,
        F_ref_star_n=0.50,
        trial_force_n=0.05,
        candidate_id="root_success",
        case_id="root_success",
    )
    opened = SurfaceBank.open(bank.root)
    surface = opened.load_surface_arrays(case.surface_id)["height_filtered"]
    stiffness = build_stiffness_model(case)
    geometry = [
        {"spine_id": "s0", "row": 0, "col": 0, "x_mm": 5.6, "y_mm": 3.6},
        {"spine_id": "s1", "row": 0, "col": 1, "x_mm": 6.0, "y_mm": 3.6},
        {"spine_id": "s2", "row": 0, "col": 2, "x_mm": 6.4, "y_mm": 3.6},
    ]
    preload = solve_preload_distribution(
        case=case,
        stiffness=stiffness,
        height_filtered=surface,
        geometry=geometry,
        dx_mm=0.2,
        dy_mm=0.2,
    )
    assert float(np.sum(preload.preload_n)) == pytest.approx(case.w_total_n, rel=1e-8, abs=1e-8)
    assert preload.normal_range_insufficient is False

    result = run_single_case(case)
    summary = result.case_summary.iloc[0]
    assert summary["engagement_success"] == (summary["n_eng"] > 0)
    expected_load_success = (
        summary["f_t_lim_n"] >= max(0.01, 0.05 * summary["w_total_n"])
        and summary["n_eng"] > 0
        and not bool(summary["normal_range_insufficient"])
        and summary["case_status"] != "failed"
    )
    assert bool(summary["load_success"]) is bool(expected_load_success)


def test_phase10_full_smoke_flow_with_required_small_parameters(tmp_path):
    bank = _make_smoke_bank(tmp_path)
    opened = SurfaceBank.open(bank.root)
    stats = opened.load_statistics()
    assert stats.groupby("surface_kind")["surface_id"].count().to_dict() == {
        kind: 3 for kind in SURFACE_KINDS
    }
    arrays = opened.load_surface_arrays("concrete_000000")
    assert {"height_raw", "height_filtered"} == set(arrays)

    p1_dir = run_single_case_sanity(
        surface_bank=bank.root,
        surface_id="concrete_000000",
        outdir=tmp_path / "P1_single_case_sanity",
    )
    p1_figures = plot_p1_single_case_sanity(stage_dir=p1_dir, style="debug")
    assert len(p1_figures) == 6
    assert all(path.exists() for path in p1_figures.values())

    p2_dir = run_p2(
        surface_bank=bank.root,
        n_surfaces_per_kind=3,
        outdir=tmp_path / "P2_compliant_k_alpha_screen",
        spring_values=(100.0, 330.0),
        alpha_values=(50.0, 60.0),
        w_values=(0.5, 1.0),
    )
    p3_dir = run_p3(
        surface_bank=bank.root,
        n_surfaces_per_kind=3,
        outdir=tmp_path / "P3_rigid_alpha_screen",
        alpha_values=(50.0, 60.0),
        w_values=(0.5, 1.0),
    )
    p2_summary = read_parquet(p2_dir / "data" / "stage_summary.parquet")
    p3_summary = read_parquet(p3_dir / "data" / "stage_summary.parquet")
    assert len(p2_summary) == 2 * 2 * 2 * 4 * 3
    assert len(p3_summary) == 2 * 2 * 4 * 3
    assert (p2_dir / "data" / "selected_candidates.json").exists()
    assert (p3_dir / "data" / "selected_candidates.json").exists()

    p5a_dir = run_p5a(
        surface_bank=bank.root,
        p2_selected=p2_dir / "data" / "selected_candidates.json",
        p3_selected=p3_dir / "data" / "selected_candidates.json",
        n_surfaces_per_kind=3,
        outdir=tmp_path / "P5a_array_pitch_coarse_screen",
        rows=(2,),
        cols=(2, 3),
        pitch_t_values=(4.0,),
        pitch_l_values=(4.0,),
        w_values=(0.5, 1.0),
        max_p2_selected=3,
        max_p3_selected=3,
    )
    p5a_summary = read_parquet(p5a_dir / "data" / "stage_summary.parquet")
    p5a_selected = json.loads((p5a_dir / "data" / "selected_candidates.json").read_text())
    p2_for_p5 = json.loads((p2_dir / "data" / "selected_candidates.json").read_text())[:3]
    p3_for_p5 = json.loads((p3_dir / "data" / "selected_candidates.json").read_text())[:3]
    assert len(p5a_summary) == (len(p2_for_p5) + len(p3_for_p5)) * 2 * 4 * 3 * 2
    assert {item["array_type"] for item in p5a_selected} == {"rigid", "compliant"}
    assert not list((p5a_dir / "data").glob("*.npz"))

    p5b_dir = run_p5b(
        surface_bank=bank.root,
        p5a_selected=p5a_dir / "data" / "selected_candidates.json",
        n_surfaces_per_kind=3,
        outdir=tmp_path / "P5b_array_pitch_refine_screen",
        w_values=(0.5, 1.0),
    )
    p5b_selected = json.loads((p5b_dir / "data" / "selected_candidates.json").read_text())
    assert [item for item in p5b_selected if item["array_type"] == "rigid"]
    assert [item for item in p5b_selected if item["array_type"] == "compliant"]

    p6_candidates = [
        next(item for item in p5b_selected if item["array_type"] == "rigid"),
        next(item for item in p5b_selected if item["array_type"] == "compliant"),
    ]
    p6_candidates_path = tmp_path / "p6_two_candidates.json"
    p6_candidates_path.write_text(json.dumps(p6_candidates), encoding="utf-8")
    p6_dir = run_p6(
        surface_bank=bank.root,
        selected_candidates=p6_candidates_path,
        n_surfaces_per_kind=3,
        surface_selection="first_n",
        outdir=tmp_path / "P6_final_3d_monte_carlo",
        workers=2,
        w_values=(0.5, 1.0),
    )
    p6_summary = read_parquet(p6_dir / "data" / "final_summary.parquet")
    p6_spines = read_parquet(p6_dir / "data" / "final_spines.parquet")
    assert len(p6_summary) == 2 * 4 * 3 * 2
    assert len(p6_spines) > len(p6_summary)
    assert not list((p6_dir / "data").glob("*.npz"))
    assert (p6_dir / "data" / "convergence_statistics.parquet").exists()

    p7_dir = run_p7_surface_generalization(
        p6_dir=p6_dir,
        outdir=tmp_path / "P7_surface_generalization",
    )
    p8_dir = run_p8_preload_efficiency(
        p6_dir=p6_dir,
        outdir=tmp_path / "P8_preload_efficiency",
    )
    p7_mtime = (p7_dir / "data" / "surface_generalization_statistics.parquet").stat().st_mtime
    p8_mtime = (p8_dir / "data" / "preload_efficiency_statistics.parquet").stat().st_mtime
    assert p7_mtime >= p6_summary.shape[0] * 0
    assert p8_mtime >= p6_summary.shape[0] * 0

    shutil.rmtree(p6_dir / "figures_report", ignore_errors=True)
    p6_figures = plot_stage(p6_dir, style="report")
    assert {path.name for path in p6_figures.values()} == {
        "final_force_boxplot_by_candidate.png",
        "final_success_probability_by_candidate.png",
        "final_surface_generalization_boxplot.png",
        "final_preload_efficiency_curve.png",
        "final_rigid_vs_compliant_summary.png",
        "convergence_curve.png",
    }
    assert all(path.exists() for path in p6_figures.values())
