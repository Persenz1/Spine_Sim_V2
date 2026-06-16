from __future__ import annotations

import json

import pytest

from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.pipelines.p5_array_screen import run_p5a, run_p5b
from Spine_Sim_V2.plotting.plot_stage import plot_stage
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _make_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase7",
        surface_kinds="sandpaper,concrete,brick,painted_wall",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=24.0,
        size_y_mm=18.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase7",
    )


def _write_selected_inputs(tmp_path):
    p2 = [
        {
            "candidate_id": "k330_alpha60",
            "source_stage": "p2_compliant_k_alpha",
            "array_type": "compliant",
            "rows": 1,
            "cols": 1,
            "pitch_t_mm": None,
            "pitch_l_mm": None,
            "alpha_p_deg": 60.0,
            "spring_k_n_per_m": 330.0,
            "score_total": 0.9,
            "rank": 1,
            "selection_reason": "test",
        },
        {
            "candidate_id": "k470_alpha70",
            "source_stage": "p2_compliant_k_alpha",
            "array_type": "compliant",
            "rows": 1,
            "cols": 1,
            "pitch_t_mm": None,
            "pitch_l_mm": None,
            "alpha_p_deg": 70.0,
            "spring_k_n_per_m": 470.0,
            "score_total": 0.8,
            "rank": 2,
            "selection_reason": "test",
        },
    ]
    p3 = [
        {
            "candidate_id": "alpha60",
            "source_stage": "p3_rigid_alpha",
            "array_type": "rigid",
            "rows": 1,
            "cols": 1,
            "pitch_t_mm": None,
            "pitch_l_mm": None,
            "alpha_p_deg": 60.0,
            "spring_k_n_per_m": None,
            "score_total": 0.9,
            "rank": 1,
            "selection_reason": "test",
        },
        {
            "candidate_id": "alpha70",
            "source_stage": "p3_rigid_alpha",
            "array_type": "rigid",
            "rows": 1,
            "cols": 1,
            "pitch_t_mm": None,
            "pitch_l_mm": None,
            "alpha_p_deg": 70.0,
            "spring_k_n_per_m": None,
            "score_total": 0.8,
            "rank": 2,
            "selection_reason": "test",
        },
    ]
    p2_path = tmp_path / "p2_selected.json"
    p3_path = tmp_path / "p3_selected.json"
    p2_path.write_text(json.dumps(p2), encoding="utf-8")
    p3_path.write_text(json.dumps(p3), encoding="utf-8")
    return p2_path, p3_path


def test_p5a_and_p5b_smoke_outputs(tmp_path):
    bank = _make_bank(tmp_path)
    p2_selected, p3_selected = _write_selected_inputs(tmp_path)
    p5a_dir = run_p5a(
        surface_bank=bank.root,
        p2_selected=p2_selected,
        p3_selected=p3_selected,
        n_surfaces_per_kind=1,
        outdir=tmp_path / "P5a_array_pitch_coarse_screen",
        rows=(2, 3, 4),
        cols=(2, 3),
        pitch_t_values=(4.0,),
        pitch_l_values=(4.0,),
        w_values=(0.5,),
        max_p2_selected=2,
        max_p3_selected=2,
    )
    p5a_summary = read_parquet(p5a_dir / "data" / "stage_summary.parquet")
    p5a_spines = read_parquet(p5a_dir / "data" / "stage_spines.parquet")
    p5a_selected = json.loads((p5a_dir / "data" / "selected_candidates.json").read_text())

    assert len(p5a_summary) == 24 * 4
    assert len(p5a_spines) > len(p5a_summary)
    assert {item["array_type"] for item in p5a_selected} == {"rigid", "compliant"}
    assert (p5a_dir / "reports" / "selection_reason.md").exists()
    assert not list((p5a_dir / "data").glob("*.npz"))

    p5b_dir = run_p5b(
        surface_bank=bank.root,
        p5a_selected=p5a_dir / "data" / "selected_candidates.json",
        n_surfaces_per_kind=1,
        outdir=tmp_path / "P5b_array_pitch_refine_screen",
        w_values=(0.5,),
    )
    p5b_summary = read_parquet(p5b_dir / "data" / "stage_summary.parquet")
    p5b_selected = json.loads((p5b_dir / "data" / "selected_candidates.json").read_text())
    rigid = [item for item in p5b_selected if item["array_type"] == "rigid"]
    compliant = [item for item in p5b_selected if item["array_type"] == "compliant"]

    assert len(p5b_summary) == len(p5a_selected) * 4
    assert len(rigid) == 5
    assert len(compliant) == 5
    for item in p5b_selected:
        assert {"candidate_id", "array_type", "rows", "cols", "pitch_t_mm", "pitch_l_mm", "alpha_p_deg"} <= set(item)
        assert item["candidate_id"].startswith(("R", "C"))

    outputs = plot_stage(p5b_dir, style="report")
    assert {path.name for path in outputs.values()} == {
        "array_force_heatmap.png",
        "array_success_heatmap.png",
        "array_n_eff_kish_heatmap.png",
        "array_eta_max_heatmap.png",
        "rigid_vs_compliant_ranking.png",
        "selected_candidates_overview.png",
    }
    assert all(path.exists() for path in outputs.values())

