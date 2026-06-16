from __future__ import annotations

import shutil

import pytest

from Spine_Sim_V2.analysis.ranking import analyze_stage
from Spine_Sim_V2.io.parquet_io import write_parquet
from Spine_Sim_V2.pipelines.p2_compliant_k_alpha import run as run_p2
from Spine_Sim_V2.plotting.plot_stage import plot_stage
from Spine_Sim_V2.plotting.plot_surface import plot_surface_audit
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _make_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase9",
        surface_kinds="sandpaper,concrete,brick,painted_wall",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=18.0,
        size_y_mm=12.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase9",
    )


def test_phase9_stage_analysis_and_style_regeneration(tmp_path):
    bank = _make_bank(tmp_path)
    stage_dir = run_p2(
        surface_bank=bank.root,
        n_surfaces_per_kind=1,
        outdir=tmp_path / "P2_compliant_k_alpha_screen",
    )
    shutil.rmtree(stage_dir / "figures_report", ignore_errors=True)

    grouped, rankings = analyze_stage(stage_dir)
    assert len(grouped) > 0
    assert len(rankings) > 0
    assert (stage_dir / "reports" / "selection_reason.md").exists()
    assert (stage_dir / "reports" / "stage_report.md").exists()

    report_outputs = plot_stage(stage_dir, style="report")
    assert {path.name for path in report_outputs.values()} == {
        "k_alpha_success_heatmap.png",
        "k_alpha_f_t_lim_heatmap.png",
        "k_alpha_saturation_heatmap.png",
        "k_alpha_efficiency_heatmap.png",
    }
    assert all(path.exists() for path in report_outputs.values())

    debug_outputs = plot_stage(stage_dir, style="debug")
    assert all(path.suffix == ".png" and path.exists() for path in debug_outputs.values())

    paper_outputs = plot_stage(stage_dir, style="paper")
    assert all(path.suffix == ".pdf" and path.exists() for path in paper_outputs.values())


def test_phase9_missing_plot_fields_report_clear_error(tmp_path):
    pd = pytest.importorskip("pandas")
    stage_dir = tmp_path / "P7_surface_generalization"
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True)
    write_parquet(
        pd.DataFrame(
            {
                "candidate_id": ["C001"],
                "surface_kind": ["concrete"],
                "f_t_lim_n_mean": [0.1],
            }
        ),
        data_dir / "surface_generalization_statistics.parquet",
    )
    write_parquet(
        pd.DataFrame(
            {
                "candidate_id": ["C001"],
                "surface_kind": ["concrete"],
                "rank_shift": [0.0],
            }
        ),
        data_dir / "surface_generalization_rankings.parquet",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        plot_stage(stage_dir, style="report")


def test_phase9_surface_audit_figures_stay_outside_surface_bank(tmp_path):
    bank = _make_bank(tmp_path)
    outdir = tmp_path / "outputs" / "P0_surface_audit"
    outputs = plot_surface_audit(
        surface_bank=bank.root,
        sample_per_kind=1,
        outdir=outdir,
        style="debug",
    )
    assert all(path.exists() for path in outputs.values())
    image_suffixes = {".png", ".svg", ".pdf"}
    assert not [
        path
        for path in bank.root.rglob("*")
        if path.suffix.lower() in image_suffixes
    ]
