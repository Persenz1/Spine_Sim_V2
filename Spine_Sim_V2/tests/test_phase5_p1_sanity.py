from __future__ import annotations

import pytest

from Spine_Sim_V2.io.npz_io import load_npz_arrays
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.pipelines.p1_single_case import run_single_case_sanity
from Spine_Sim_V2.plotting.plot_case import P1_DEBUG_FIGURE_NAMES, plot_p1_single_case_sanity
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_p1_single_case_sanity_outputs_and_plots(tmp_path):
    bank = create_surface_bank(
        bank_id="surface_bank_phase5",
        surface_kinds="concrete",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=18.0,
        size_y_mm=12.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase5",
    )
    stage_dir = run_single_case_sanity(
        surface_bank=bank.root,
        surface_id="concrete_000000",
        outdir=tmp_path / "P1_single_case_sanity",
    )

    assert (stage_dir / "manifest.json").exists()
    assert (stage_dir / "schema.json").exists()
    assert (stage_dir / "data" / "stage_summary.parquet").exists()
    assert (stage_dir / "data" / "stage_summary_preview.csv").exists()
    assert (stage_dir / "data" / "stage_spines.parquet").exists()
    assert (stage_dir / "reports" / "stage_report.md").exists()
    assert not list((stage_dir / "figures_debug").glob("*.png"))

    summary = read_parquet(stage_dir / "data" / "stage_summary.parquet")
    spines = read_parquet(stage_dir / "data" / "stage_spines.parquet")
    assert set(summary["array_type"]) == {"rigid", "compliant"}
    assert len(summary) == 2
    assert len(spines) == 12
    assert "height_raw" not in summary.columns
    assert "height_filtered" not in spines.columns

    required_arrays = {
        "height_raw",
        "height_filtered",
        "slope_angle",
        "effective_contact_angle",
        "engagement_candidate_mask",
        "spine_x",
        "spine_y",
        "search_path_x",
        "search_path_y",
        "engagement_x",
        "engagement_y",
        "load_displacement_s",
        "load_displacement_f",
        "failure_sequence",
    }
    for case_id in summary["case_id"]:
        case_dir = stage_dir / "sample_cases" / str(case_id)
        assert (case_dir / "case_result.json").exists()
        assert (case_dir / "case_summary.parquet").exists()
        assert (case_dir / "case_spines.parquet").exists()
        arrays = load_npz_arrays(case_dir / "case_arrays.npz")
        assert required_arrays <= set(arrays)

    outputs = plot_p1_single_case_sanity(stage_dir=stage_dir, style="debug")
    assert {path.name for path in outputs.values()} == set(P1_DEBUG_FIGURE_NAMES)
    assert all(path.exists() for path in outputs.values())

