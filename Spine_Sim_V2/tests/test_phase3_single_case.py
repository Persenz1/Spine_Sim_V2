from __future__ import annotations

import numpy as np
import pytest

from Spine_Sim_V2.core.types import (
    SingleCaseInput,
    schema_field_names,
    stage_spines_schema,
    stage_summary_schema,
)
from Spine_Sim_V2.pipelines.p1_single_case import run_single_case
from Spine_Sim_V2.surfaces.bank import create_surface_bank


pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _make_bank(tmp_path):
    return create_surface_bank(
        bank_id="surface_bank_phase3",
        surface_kinds="sandpaper",
        n_per_kind=1,
        resolution_cells_per_mm=5,
        size_x_mm=6.0,
        size_y_mm=4.0,
        tip_radius_mm=0.05,
        outdir=tmp_path / "surface_bank_phase3",
    )


def _case_kwargs(bank_path, array_type: str, spring_k_n_per_m: float | None):
    return {
        "surface_bank_path": bank_path,
        "surface_id": "sandpaper_000000",
        "array_type": array_type,
        "rows": 2,
        "cols": 3,
        "pitch_t_mm": 1.0,
        "pitch_l_mm": 1.0,
        "alpha_p_deg": 60.0,
        "spring_k_n_per_m": spring_k_n_per_m,
        "tip_radius_mm": 0.05,
        "spine_diameter_mm": 0.2,
        "search_travel_mm": 2.0,
        "w_total_n": 1.0,
        "f_s": 1.0,
        "F_ref_star_n": 0.5,
        "trial_force_n": 0.05,
    }


def test_single_case_runs_rigid_and_compliant(tmp_path):
    bank = _make_bank(tmp_path)
    rigid = run_single_case(
        SingleCaseInput(
            case_id="rigid_case",
            candidate_id="rigid_candidate",
            **_case_kwargs(bank.root, "rigid", None),
        )
    )
    compliant = run_single_case(
        SingleCaseInput(
            case_id="compliant_case",
            candidate_id="compliant_candidate",
            **_case_kwargs(bank.root, "compliant", 1000.0),
        )
    )

    for result in (rigid, compliant):
        assert len(result.case_summary) == 1
        assert len(result.case_spines) == 6
        assert schema_field_names(stage_summary_schema) <= set(result.case_summary.columns)
        assert schema_field_names(stage_spines_schema) <= set(result.case_spines.columns)
        assert result.case_summary.loc[0, "surface_id"] == "sandpaper_000000"
        assert "height_raw" not in result.case_summary.columns
        assert "height_filtered" not in result.case_spines.columns
        assert result.diagnostics["preload_solved_before_engagement"] is True
        assert np.isfinite(result.case_spines["cap_n"].to_numpy(dtype=float)).all()
        assert not np.isinf(result.case_spines["cap_n"].to_numpy(dtype=float)).any()

    assert pd.isna(rigid.case_summary.loc[0, "spring_k_n_per_m"])
    assert pd.isna(rigid.case_summary.loc[0, "spring_k_n_per_mm"])
    assert compliant.case_summary.loc[0, "spring_k_n_per_m"] == 1000.0
    assert compliant.case_summary.loc[0, "spring_k_n_per_mm"] == 1.0


def test_phi_c_is_computed_only_after_positive_preload(tmp_path):
    bank = _make_bank(tmp_path)
    result = run_single_case(
        SingleCaseInput(
            case_id="order_case",
            candidate_id="order_candidate",
            **_case_kwargs(bank.root, "compliant", 1000.0),
        )
    )

    spines = result.case_spines
    positive_preload = spines["preload_n"] > 0.0
    assert positive_preload.any()
    assert spines.loc[positive_preload, "phi_c_deg"].notna().all()
    assert spines.loc[~positive_preload, "phi_c_deg"].isna().all()
    assert result.diagnostics["solver_sequence"].index("preload_solved") < result.diagnostics[
        "solver_sequence"
    ].index("engagement_searched")

