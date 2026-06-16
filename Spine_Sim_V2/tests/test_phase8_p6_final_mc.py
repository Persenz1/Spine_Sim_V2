from __future__ import annotations

import json

import pytest

from Spine_Sim_V2.analysis.final_mc import run_p7_surface_generalization, run_p8_preload_efficiency
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.pipelines.p6_final_mc import run as run_p6
from Spine_Sim_V2.plotting.plot_stage import plot_stage
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _make_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase8",
        surface_kinds="sandpaper,concrete,brick,painted_wall",
        n_per_kind=5,
        resolution_cells_per_mm=5,
        size_x_mm=24.0,
        size_y_mm=18.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase8",
    )


def _write_p5b_selected(tmp_path):
    records = [
        {
            "candidate_id": "R001",
            "source_stage": "p5b_array_pitch_refine",
            "array_type": "rigid",
            "rows": 2,
            "cols": 2,
            "pitch_t_mm": 4.0,
            "pitch_l_mm": 4.0,
            "alpha_p_deg": 60.0,
            "spring_k_n_per_m": None,
            "score_total": 0.9,
            "rank": 1,
            "selection_reason": "test rigid",
        },
        {
            "candidate_id": "C001",
            "source_stage": "p5b_array_pitch_refine",
            "array_type": "compliant",
            "rows": 2,
            "cols": 2,
            "pitch_t_mm": 4.0,
            "pitch_l_mm": 4.0,
            "alpha_p_deg": 60.0,
            "spring_k_n_per_m": 330.0,
            "score_total": 0.8,
            "rank": 2,
            "selection_reason": "test compliant",
        },
    ]
    path = tmp_path / "p5b_selected_candidates.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_p6_p7_p8_final_monte_carlo_smoke(tmp_path):
    bank = _make_bank(tmp_path)
    selected = _write_p5b_selected(tmp_path)
    p6_dir = run_p6(
        surface_bank=bank.root,
        selected_candidates=selected,
        n_surfaces_per_kind=5,
        surface_selection="first_n",
        outdir=tmp_path / "P6_final_3d_monte_carlo",
        workers=2,
        w_values=(0.5, 1.0),
    )

    summary = read_parquet(p6_dir / "data" / "final_summary.parquet")
    spines = read_parquet(p6_dir / "data" / "final_spines.parquet")
    grouped = read_parquet(p6_dir / "data" / "final_grouped_statistics.parquet")
    rankings = read_parquet(p6_dir / "data" / "final_rankings.parquet")
    convergence = read_parquet(p6_dir / "data" / "convergence_statistics.parquet")
    candidates = json.loads((p6_dir / "data" / "final_candidates.json").read_text())

    assert len(summary) == 2 * 4 * 5 * 2
    assert len(spines) == len(summary) * 4
    assert not list((p6_dir / "sample_cases").glob("*.npz"))
    assert (p6_dir / "sample_cases" / "sample_cases.json").exists()
    assert {"candidate_id", "array_type", "surface_kind", "w_total_n", "success_probability"} <= set(grouped)
    assert {"mean", "median", "std", "p05", "p25", "p75", "p95"} <= {
        suffix
        for column in grouped.columns
        for suffix in ("mean", "median", "std", "p05", "p25", "p75", "p95")
        if column.endswith(f"_{suffix}")
    }
    assert {"n_surfaces_used", "f_t_lim_n_mean", "f_t_lim_n_p05", "f_t_lim_n_p95", "success_probability", "eta_max_mean"} <= set(convergence)
    assert len(rankings) == 2
    assert len(candidates) == 2

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

    p7_dir = run_p7_surface_generalization(
        p6_dir=p6_dir,
        outdir=tmp_path / "P7_surface_generalization",
    )
    p8_dir = run_p8_preload_efficiency(
        p6_dir=p6_dir,
        outdir=tmp_path / "P8_preload_efficiency",
    )
    assert (p7_dir / "data" / "surface_generalization_statistics.parquet").exists()
    assert (p7_dir / "data" / "surface_generalization_rankings.parquet").exists()
    assert (p8_dir / "data" / "preload_efficiency_statistics.parquet").exists()

    p7_figures = plot_stage(p7_dir, style="report")
    p8_figures = plot_stage(p8_dir, style="report")
    assert {path.name for path in p7_figures.values()} == {
        "surface_generalization_boxplot.png",
        "surface_rank_shift_heatmap.png",
        "candidate_by_surface_success_heatmap.png",
    }
    assert {path.name for path in p8_figures.values()} == {
        "preload_force_curve.png",
        "preload_efficiency_curve.png",
        "preload_success_curve.png",
    }
    assert all(path.exists() for path in [*p7_figures.values(), *p8_figures.values()])
