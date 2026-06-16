from __future__ import annotations

import numpy as np
import pytest

from Spine_Sim_V2.core.types import (
    SCHEMA_REGISTRY,
    schema_field_names,
    stage_spines_schema,
    stage_summary_schema,
)
from Spine_Sim_V2.core.units import spring_k_n_per_m_to_n_per_mm
from Spine_Sim_V2.io.manifest import create_manifest, read_manifest, write_manifest
from Spine_Sim_V2.io.npz_io import load_npz_arrays, save_npz_arrays
from Spine_Sim_V2.io.parquet_io import read_parquet, write_parquet, write_preview_csv
from Spine_Sim_V2.io.schema_io import read_schema, write_schema


def test_manifest_can_be_written_and_read(tmp_path):
    manifest = create_manifest(
        project_name="phase1-test",
        model_version="test-model",
        surface_bank_id="surface_bank_v001",
        parameter_grid={"alpha_p_deg": [25.0, 35.0]},
        n_cases_expected=2,
        n_cases_completed=1,
        failed_cases=["case-002"],
        notes="unit test",
        code_version="unknown",
        created_time="2026-06-16T00:00:00Z",
    )

    path = write_manifest(manifest, tmp_path)
    loaded = read_manifest(path)

    required_keys = {
        "project_name",
        "created_time",
        "code_version",
        "model_version",
        "data_schema_version",
        "surface_bank_id",
        "surface_generator_version",
        "probe_filter_version",
        "random_seed_policy",
        "parameter_grid",
        "n_cases_expected",
        "n_cases_completed",
        "failed_cases",
        "notes",
    }
    assert path.name == "manifest.json"
    assert required_keys <= set(loaded)
    assert loaded["project_name"] == "phase1-test"
    assert loaded["failed_cases"] == ["case-002"]


def test_schema_can_be_written_and_read(tmp_path):
    path = write_schema(tmp_path)
    loaded = read_schema(path)

    assert path.name == "schema.json"
    assert "data_schema_version" in loaded
    assert set(SCHEMA_REGISTRY) <= set(loaded["schemas"])

    first_field = loaded["schemas"]["stage_summary"][0]
    assert {"name", "dtype", "unit", "nullable", "description"} <= set(first_field)


def test_required_summary_fields_exist():
    required_fields = {
        "case_id",
        "stage",
        "case_status",
        "error_code",
        "warning_flags",
        "surface_bank_id",
        "surface_id",
        "surface_kind",
        "surface_seed",
        "candidate_id",
        "array_type",
        "rows",
        "cols",
        "n_nom",
        "pitch_t_mm",
        "pitch_l_mm",
        "alpha_p_deg",
        "spring_k_n_per_m",
        "spring_k_n_per_mm",
        "tip_radius_mm",
        "spine_diameter_mm",
        "search_travel_mm",
        "w_total_n",
        "f_s",
        "phi_s_deg",
        "F_ref_star_n",
        "trial_force_n",
        "surface_rq_raw_mm",
        "surface_ra_raw_mm",
        "surface_hpv_raw_mm",
        "surface_rq_eff_mm",
        "surface_ra_eff_mm",
        "surface_hpv_eff_mm",
        "surface_slope_mean_deg",
        "surface_slope_p95_deg",
        "candidate_density_preload_free",
        "n_con",
        "n_eng",
        "n_eff_count",
        "n_eff_kish",
        "r_con",
        "r_eng",
        "r_fail_search",
        "search_distance_mean_mm",
        "search_distance_p95_mm",
        "normal_stroke_max_mm",
        "u_ax_used_max_mm",
        "u_ax_used_mean_mm",
        "w_sat_mean_n",
        "r_sat_n",
        "r_sat_y",
        "normal_range_insufficient",
        "f_t_lim_n",
        "f_t_lim_over_w_total",
        "f_t_lim_per_nom_n",
        "f_t_lim_per_eff_n",
        "limit_displacement_mm",
        "eta_max",
        "w_cv",
        "engagement_success",
        "load_success",
        "failure_mode",
        "cascade_failure",
        "r_slip",
        "r_overload",
        "r_side_contact_risk",
    }

    assert required_fields <= schema_field_names(stage_summary_schema)


def test_required_spines_fields_exist():
    required_fields = {
        "case_id",
        "candidate_id",
        "surface_id",
        "spine_id",
        "row",
        "col",
        "array_type",
        "x_mm",
        "y_mm",
        "gap_mm",
        "alpha_p_deg",
        "pitch_t_mm",
        "pitch_l_mm",
        "contacted",
        "preload_n",
        "u_ax_used_mm",
        "normal_saturated",
        "state",
        "search_distance_mm",
        "engaged",
        "engagement_x_mm",
        "engagement_y_mm",
        "phi_c_deg",
        "phi_eng_deg",
        "phi_hook_min_deg",
        "side_contact_risk",
        "cap_n",
        "cap_mode",
        "load_at_limit_n",
        "failed",
        "failure_mode",
        "failure_order",
    }

    assert required_fields <= schema_field_names(stage_spines_schema)


def test_rigid_array_spring_stiffness_allows_null():
    spring_field = next(
        field for field in stage_summary_schema if field.name == "spring_k_n_per_m"
    )
    assert spring_field.nullable is True
    assert spring_k_n_per_m_to_n_per_mm(None) is None


def test_parquet_can_be_written_and_read(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    df = pd.DataFrame(
        {
            "case_id": ["case-001", "case-002"],
            "array_type": ["rigid", "compliant"],
            "spring_k_n_per_m": [None, 1000.0],
            "w_total_n": [1.2, 1.5],
        }
    )

    parquet_path = write_parquet(df, tmp_path / "stage_summary.parquet")
    preview_path = write_preview_csv(df, tmp_path / "stage_summary_preview.csv", max_rows=1)
    loaded = read_parquet(parquet_path)

    assert parquet_path.exists()
    assert preview_path.exists()
    assert list(loaded["case_id"]) == ["case-001", "case-002"]
    assert pd.isna(loaded.loc[0, "spring_k_n_per_m"])
    assert loaded.loc[1, "spring_k_n_per_m"] == 1000.0


def test_npz_can_be_written_and_read(tmp_path):
    height_mm = np.arange(9, dtype=float).reshape(3, 3)
    mask = height_mm > 4

    path = save_npz_arrays(tmp_path / "arrays.npz", height_mm=height_mm, mask=mask)
    loaded = load_npz_arrays(path)

    assert path.exists()
    np.testing.assert_array_equal(loaded["height_mm"], height_mm)
    np.testing.assert_array_equal(loaded["mask"], mask)

