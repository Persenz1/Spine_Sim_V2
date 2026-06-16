from __future__ import annotations

import inspect

import numpy as np
import pytest

from Spine_Sim_V2.solvers.engagement import compute_capacity_n
from Spine_Sim_V2.solvers.loading import run_loading_sequence
from Spine_Sim_V2.solvers.search import search_first_engagement


def _linear_x_grid(*, dx_mm: float, dy_mm: float, nx: int, ny: int, scale: float = 1.0):
    x = np.arange(nx, dtype=float) * dx_mm
    return np.tile(scale * x, (ny, 1))


def test_search_refines_surface_penetration_event():
    dx_mm = 0.2
    dy_mm = 0.2
    height = _linear_x_grid(dx_mm=dx_mm, dy_mm=dy_mm, nx=8, ny=3)
    phi = np.zeros_like(height)

    result = search_first_engagement(
        phi_map_deg=phi,
        height_filtered=height,
        z_tip_mm=0.25,
        x0_mm=0.0,
        y0_mm=0.2,
        search_travel_mm=1.0,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        phi_c_deg=40.0,
        phi_hook_min_deg=0.0,
        search_ds_coarse=0.1,
        search_refine_tol=0.002,
    )

    assert result.engaged is False
    assert result.state == "surface_contact"
    assert result.side_contact_risk is True
    assert result.search_distance_mm == pytest.approx(0.25, abs=0.003)


def test_search_refines_first_engagement_event():
    dx_mm = 0.2
    dy_mm = 0.2
    height = np.zeros((3, 8), dtype=float)
    phi = _linear_x_grid(dx_mm=dx_mm, dy_mm=dy_mm, nx=8, ny=3, scale=20.0)

    result = search_first_engagement(
        phi_map_deg=phi,
        height_filtered=height,
        z_tip_mm=1.0,
        x0_mm=0.0,
        y0_mm=0.2,
        search_travel_mm=1.0,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        phi_c_deg=5.0,
        phi_hook_min_deg=0.0,
        search_ds_coarse=0.1,
        search_refine_tol=0.002,
    )

    assert result.engaged is True
    assert result.state == "engaged"
    assert result.search_distance_mm == pytest.approx(0.25, abs=0.003)
    assert result.engagement_x_mm == pytest.approx(0.25, abs=0.003)
    assert result.phi_eng_deg == pytest.approx(5.0, abs=0.08)


def test_search_distance_is_stable_when_dx_changes():
    def run_for_dx(dx_mm: float) -> float:
        height = np.zeros((5, int(round(1.2 / dx_mm)) + 1), dtype=float)
        phi = _linear_x_grid(
            dx_mm=dx_mm,
            dy_mm=dx_mm,
            nx=height.shape[1],
            ny=height.shape[0],
            scale=20.0,
        )
        result = search_first_engagement(
            phi_map_deg=phi,
            height_filtered=height,
            z_tip_mm=1.0,
            x0_mm=0.0,
            y0_mm=2 * dx_mm,
            search_travel_mm=1.0,
            dx_mm=dx_mm,
            dy_mm=dx_mm,
            phi_c_deg=5.0,
            phi_hook_min_deg=0.0,
        )
        assert result.search_distance_mm is not None
        return result.search_distance_mm

    coarse = run_for_dx(0.2)
    fine = run_for_dx(0.1)

    assert coarse == pytest.approx(0.25, abs=0.025)
    assert fine == pytest.approx(0.25, abs=0.015)
    assert abs(coarse - fine) < 0.03


def test_side_contact_event_can_precede_engagement():
    dx_mm = 0.2
    dy_mm = 0.2
    height = np.zeros((3, 8), dtype=float)
    phi = _linear_x_grid(dx_mm=dx_mm, dy_mm=dy_mm, nx=8, ny=3, scale=20.0)

    result = search_first_engagement(
        phi_map_deg=phi,
        height_filtered=height,
        z_tip_mm=1.0,
        x0_mm=0.0,
        y0_mm=0.2,
        search_travel_mm=1.0,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        phi_c_deg=8.0,
        phi_hook_min_deg=0.0,
        search_ds_coarse=0.1,
        search_refine_tol=0.002,
        side_contact_risk_fn=lambda s, x, y, phi_deg: s >= 0.25,
    )

    assert result.engaged is False
    assert result.state == "side_contact"
    assert result.side_contact_risk is True
    assert result.search_distance_mm == pytest.approx(0.25, abs=0.003)


def test_self_lock_capacity_is_strength_limited():
    result = compute_capacity_n(
        preload_n=1.2,
        phi_eng_deg=50.0,
        f_s=1.0,
        F_ref_star_n=0.42,
    )

    assert result.cap_n == pytest.approx(0.42)
    assert result.cap_mode == "self_lock_strength"
    assert np.isfinite(result.cap_n)


def test_tangential_failure_order_is_reproducible():
    kwargs = {
        "engaged": np.array([True, True, True]),
        "search_distance_mm": np.array([0.0, 0.1, 0.2]),
        "cap_n": np.array([0.3, 0.25, 1.0]),
        "k_tt_n_per_mm": 1.0,
    }

    first = run_loading_sequence(**kwargs)
    second = run_loading_sequence(**kwargs)

    assert first.failure_order == [1, 2, 3]
    assert second.failure_order == [1, 2, 3]
    assert first.event_displacements_mm == second.event_displacements_mm
    assert first.event_total_force_n == second.event_total_force_n


def test_loading_limit_comes_from_events_not_small_steps():
    kwargs = {
        "engaged": np.array([True, True, True]),
        "search_distance_mm": np.array([0.0, 0.1, 0.2]),
        "cap_n": np.array([0.3, 0.25, 1.0]),
        "k_tt_n_per_mm": 1.0,
    }

    result = run_loading_sequence(**kwargs)

    assert "ds" not in inspect.signature(run_loading_sequence).parameters
    assert result.event_displacements_mm == pytest.approx((0.3, 0.35, 1.2))
    assert result.f_t_lim_n == pytest.approx(max(result.event_total_force_n))
    assert result.f_t_lim_n == pytest.approx(1.0)
