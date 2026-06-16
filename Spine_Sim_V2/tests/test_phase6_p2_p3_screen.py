from __future__ import annotations

import pytest

from Spine_Sim_V2.analysis.ranking import analyze_stage
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.pipelines.p2_compliant_k_alpha import run as run_p2
from Spine_Sim_V2.pipelines.p3_rigid_alpha import run as run_p3
from Spine_Sim_V2.plotting.plot_stage import plot_stage
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _make_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase6",
        surface_kinds="sandpaper,concrete,brick,painted_wall",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=18.0,
        size_y_mm=12.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase6",
    )


def test_p2_and_p3_small_screen_outputs(tmp_path):
    bank = _make_bank(tmp_path)
    p2_dir = run_p2(
        surface_bank=bank.root,
        n_surfaces_per_kind=1,
        outdir=tmp_path / "P2_compliant_k_alpha_screen",
    )
    p3_dir = run_p3(
        surface_bank=bank.root,
        n_surfaces_per_kind=1,
        outdir=tmp_path / "P3_rigid_alpha_screen",
    )

    for stage_dir in (p2_dir, p3_dir):
        assert (stage_dir / "manifest.json").exists()
        assert (stage_dir / "schema.json").exists()
        assert (stage_dir / "data" / "stage_summary.parquet").exists()
        assert (stage_dir / "data" / "stage_spines.parquet").exists()
        assert (stage_dir / "data" / "stage_grouped_statistics.parquet").exists()
        assert (stage_dir / "data" / "stage_rankings.parquet").exists()
        assert (stage_dir / "reports" / "selection_reason.md").exists()
        analyze_stage(stage_dir)

    p2_summary = read_parquet(p2_dir / "data" / "stage_summary.parquet")
    p2_rankings = read_parquet(p2_dir / "data" / "stage_rankings.parquet")
    assert len(p2_summary) == 7 * 4 * 5 * 4 * 1
    assert len(read_parquet(p2_dir / "data" / "stage_spines.parquet")) == len(p2_summary)
    assert 6 <= int(p2_rankings["selected"].sum()) <= 8

    p3_summary = read_parquet(p3_dir / "data" / "stage_summary.parquet")
    p3_rankings = read_parquet(p3_dir / "data" / "stage_rankings.parquet")
    assert len(p3_summary) == 4 * 5 * 4 * 1
    assert len(read_parquet(p3_dir / "data" / "stage_spines.parquet")) == len(p3_summary)
    assert "alpha60" in set(p3_rankings.loc[p3_rankings["selected"], "candidate_id"])

    p2_figures = plot_stage(p2_dir, style="report")
    p3_figures = plot_stage(p3_dir, style="report")
    assert {path.name for path in p2_figures.values()} == {
        "k_alpha_success_heatmap.png",
        "k_alpha_f_t_lim_heatmap.png",
        "k_alpha_saturation_heatmap.png",
        "k_alpha_efficiency_heatmap.png",
    }
    assert {path.name for path in p3_figures.values()} == {
        "rigid_alpha_success_curve.png",
        "rigid_alpha_f_t_lim_curve.png",
        "rigid_alpha_efficiency_curve.png",
        "rigid_alpha_by_surface_boxplot.png",
    }
    assert all(path.exists() for path in [*p2_figures.values(), *p3_figures.values()])

