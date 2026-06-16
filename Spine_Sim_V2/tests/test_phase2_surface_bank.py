from __future__ import annotations

import numpy as np
import pytest

from Spine_Sim_V2.io.manifest import read_manifest
from Spine_Sim_V2.plotting.plot_surface import plot_surface_audit
from Spine_Sim_V2.surfaces.audit import (
    SURFACE_STATISTICS_REQUIRED_FIELDS,
    audit_surface_bank,
)
from Spine_Sim_V2.surfaces.bank import SurfaceBank, create_surface_bank
from Spine_Sim_V2.surfaces.profiles import list_builtin_profiles


pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def test_debug_surface_bank_can_be_generated_and_referenced(tmp_path):
    kinds = list_builtin_profiles()
    outdir = tmp_path / "surface_bank_debug"

    bank = create_surface_bank(
        bank_id="surface_bank_debug",
        surface_kinds=kinds,
        n_per_kind=2,
        resolution_cells_per_mm=5,
        size_x_mm=6.0,
        size_y_mm=4.0,
        tip_radius_mm=0.05,
        outdir=outdir,
    )

    opened = SurfaceBank.open(outdir)
    stats = opened.load_statistics()

    assert bank.root == outdir
    assert opened.bank_id == "surface_bank_debug"
    assert len(stats) == len(kinds) * 2
    assert set(SURFACE_STATISTICS_REQUIRED_FIELDS) <= set(stats.columns)
    assert set(stats["surface_kind"]) == set(kinds)
    assert stats.groupby("surface_kind")["surface_id"].count().to_dict() == {
        kind: 2 for kind in kinds
    }

    manifest = read_manifest(outdir)
    assert manifest["surface_bank_id"] == "surface_bank_debug"
    assert manifest["n_cases_expected"] == len(kinds) * 2
    assert manifest["n_cases_completed"] == len(kinds) * 2

    surface_id = str(stats.iloc[0]["surface_id"])
    arrays = opened.load_surface_arrays(surface_id)
    assert set(arrays) == {"height_raw", "height_filtered"}
    assert arrays["height_raw"].dtype == np.float32
    assert arrays["height_filtered"].dtype == np.float32
    assert arrays["height_raw"].shape == (20, 30)
    assert arrays["height_filtered"].shape == (20, 30)

    record = opened.get_surface_record(surface_id)
    assert record["surface_id"] == surface_id


def test_surface_bank_audit_and_plots_stay_outside_bank(tmp_path):
    bank_dir = tmp_path / "surface_bank_debug"
    outdir = tmp_path / "outputs" / "P0_surface_audit"
    bank = create_surface_bank(
        bank_id="surface_bank_debug",
        surface_kinds="sandpaper,concrete",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=4.0,
        size_y_mm=3.0,
        tip_radius_mm=0.05,
        outdir=bank_dir,
    )

    audit = audit_surface_bank(bank)
    assert audit["valid"] is True
    assert audit["image_files_inside_bank"] == []

    outputs = plot_surface_audit(
        surface_bank=bank_dir,
        sample_per_kind=1,
        outdir=outdir,
    )
    expected_names = {
        "surface_gallery.png",
        "filtered_surface_gallery.png",
        "slope_distribution_by_surface.png",
        "candidate_density_distribution.png",
    }
    assert {path.name for path in outputs.values()} == expected_names
    assert all(path.exists() for path in outputs.values())
    assert all(outdir in path.parents for path in outputs.values())
    assert not any(bank_dir.rglob("*.png"))

