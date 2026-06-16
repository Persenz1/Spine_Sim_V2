"""Finite tangential search with event refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from Spine_Sim_V2.solvers.contact import interpolate_bilinear


@dataclass(frozen=True)
class SearchResult:
    """Result of one spine's finite search."""

    engaged: bool
    search_distance_mm: float | None
    engagement_x_mm: float | None
    engagement_y_mm: float | None
    phi_eng_deg: float | None
    state: str
    side_contact_risk: bool = False


SideContactRiskFn = Callable[[float, float, float, float], bool]


def search_first_engagement(
    *,
    phi_map_deg: NDArray[np.floating],
    x0_mm: float,
    y0_mm: float,
    search_travel_mm: float,
    dx_mm: float,
    dy_mm: float,
    phi_c_deg: float,
    phi_hook_min_deg: float,
    height_filtered: NDArray[np.floating] | None = None,
    z_tip_mm: float | None = None,
    search_ds_coarse: float | None = None,
    search_refine_tol: float | None = None,
    max_refine_iter: int = 20,
    side_contact_risk_fn: SideContactRiskFn | None = None,
) -> SearchResult:
    """Search +x for the first engageable point.

    Coarse samples only bracket events. Surface penetration and first
    engagement are refined with bisection, so the returned event is not tied to
    an arbitrary fixed small step.
    """
    if search_travel_mm < 0.0:
        raise ValueError("search_travel_mm must be non-negative.")
    if dx_mm <= 0.0 or dy_mm <= 0.0:
        raise ValueError("dx_mm and dy_mm must be positive.")
    if max_refine_iter <= 0:
        raise ValueError("max_refine_iter must be positive.")
    if not np.isfinite(phi_c_deg):
        return SearchResult(False, None, None, None, None, "no_contact")

    threshold_deg = max(float(phi_c_deg), float(phi_hook_min_deg))
    step_mm = (
        float(search_ds_coarse)
        if search_ds_coarse is not None
        else max(min(dx_mm, dy_mm) * 0.5, 1e-6)
    )
    if step_mm <= 0.0:
        raise ValueError("search_ds_coarse must be positive.")
    refine_tol = (
        float(search_refine_tol)
        if search_refine_tol is not None
        else max(min(dx_mm, dy_mm) * 0.1, 1e-9)
    )
    if refine_tol <= 0.0:
        raise ValueError("search_refine_tol must be positive.")

    n_steps = max(1, int(np.ceil(search_travel_mm / step_mm)))
    distances = np.linspace(0.0, search_travel_mm, n_steps + 1)

    def phi_at(s_mm: float) -> float:
        return interpolate_bilinear(
            phi_map_deg,
            x0_mm + s_mm,
            y0_mm,
            dx_mm=dx_mm,
            dy_mm=dy_mm,
        )

    def margin_at(s_mm: float) -> float:
        phi = phi_at(s_mm)
        if not np.isfinite(phi):
            return float("nan")
        return float(phi - threshold_deg)

    def gap_at(s_mm: float) -> float:
        if height_filtered is None or z_tip_mm is None:
            return float("inf")
        h_eff = interpolate_bilinear(
            height_filtered,
            x0_mm + s_mm,
            y0_mm,
            dx_mm=dx_mm,
            dy_mm=dy_mm,
        )
        if not np.isfinite(h_eff):
            return float("nan")
        return float(z_tip_mm - h_eff)

    previous_s = float(distances[0])
    previous_margin = margin_at(previous_s)
    previous_gap = gap_at(previous_s)
    if _is_engaged(previous_margin):
        return _engaged_result(previous_s, x0_mm, y0_mm, phi_at(previous_s), side_contact_risk_fn)

    for distance_mm in distances[1:]:
        current_s = float(distance_mm)
        current_margin = margin_at(current_s)
        current_gap = gap_at(current_s)

        penetration_s = _find_surface_crossing(
            previous_s=previous_s,
            current_s=current_s,
            previous_gap=previous_gap,
            current_gap=current_gap,
            gap_at=gap_at,
            refine_tol=refine_tol,
            max_refine_iter=max_refine_iter,
        )
        engagement_s = _find_engagement_crossing(
            previous_s=previous_s,
            current_s=current_s,
            previous_margin=previous_margin,
            current_margin=current_margin,
            margin_at=margin_at,
            refine_tol=refine_tol,
            max_refine_iter=max_refine_iter,
        )
        side_contact_s = _find_side_contact_crossing(
            previous_s=previous_s,
            current_s=current_s,
            previous_phi=phi_at(previous_s),
            current_phi=phi_at(current_s),
            x0_mm=x0_mm,
            y0_mm=y0_mm,
            side_contact_risk_fn=side_contact_risk_fn,
            refine_tol=refine_tol,
            max_refine_iter=max_refine_iter,
        )
        earliest_non_engagement = _min_optional(penetration_s, side_contact_s)
        if engagement_s is not None and (
            earliest_non_engagement is None or engagement_s <= earliest_non_engagement + refine_tol
        ):
            phi_eng = phi_at(engagement_s)
            return _engaged_result(engagement_s, x0_mm, y0_mm, phi_eng, side_contact_risk_fn)
        if side_contact_s is not None and (
            penetration_s is None or side_contact_s <= penetration_s + refine_tol
        ):
            phi_side = phi_at(side_contact_s)
            return SearchResult(
                engaged=False,
                search_distance_mm=float(side_contact_s),
                engagement_x_mm=float(x0_mm + side_contact_s),
                engagement_y_mm=float(y0_mm),
                phi_eng_deg=float(phi_side) if np.isfinite(phi_side) else None,
                state="side_contact",
                side_contact_risk=True,
            )
        if penetration_s is not None:
            phi_at_contact = phi_at(penetration_s)
            return SearchResult(
                engaged=False,
                search_distance_mm=float(penetration_s),
                engagement_x_mm=float(x0_mm + penetration_s),
                engagement_y_mm=float(y0_mm),
                phi_eng_deg=float(phi_at_contact) if np.isfinite(phi_at_contact) else None,
                state="surface_contact",
                side_contact_risk=True,
            )

        previous_s = current_s
        previous_margin = current_margin
        previous_gap = current_gap
    return SearchResult(False, None, None, None, None, "search_failed")


def _is_engaged(margin: float) -> bool:
    return bool(np.isfinite(margin) and margin >= 0.0)


def _engaged_result(
    s_mm: float,
    x0_mm: float,
    y0_mm: float,
    phi_eng_deg: float,
    side_contact_risk_fn: SideContactRiskFn | None,
) -> SearchResult:
    side_risk = False
    if side_contact_risk_fn is not None and np.isfinite(phi_eng_deg):
        side_risk = bool(side_contact_risk_fn(s_mm, x0_mm + s_mm, y0_mm, phi_eng_deg))
    return SearchResult(
        engaged=True,
        search_distance_mm=float(s_mm),
        engagement_x_mm=float(x0_mm + s_mm),
        engagement_y_mm=float(y0_mm),
        phi_eng_deg=float(phi_eng_deg) if np.isfinite(phi_eng_deg) else None,
        state="engaged",
        side_contact_risk=side_risk,
    )


def _find_surface_crossing(
    *,
    previous_s: float,
    current_s: float,
    previous_gap: float,
    current_gap: float,
    gap_at: Callable[[float], float],
    refine_tol: float,
    max_refine_iter: int,
) -> float | None:
    if not (np.isfinite(previous_gap) and np.isfinite(current_gap)):
        return None
    if previous_gap < 0.0:
        return previous_s
    if not (previous_gap >= 0.0 and current_gap < 0.0):
        return None
    return _bisect_first_true(
        low=previous_s,
        high=current_s,
        predicate=lambda s: gap_at(s) < 0.0,
        refine_tol=refine_tol,
        max_refine_iter=max_refine_iter,
    )


def _find_engagement_crossing(
    *,
    previous_s: float,
    current_s: float,
    previous_margin: float,
    current_margin: float,
    margin_at: Callable[[float], float],
    refine_tol: float,
    max_refine_iter: int,
) -> float | None:
    if not (np.isfinite(previous_margin) and np.isfinite(current_margin)):
        return None
    if previous_margin >= 0.0:
        return previous_s
    if not (previous_margin < 0.0 and current_margin >= 0.0):
        return None
    return _bisect_first_true(
        low=previous_s,
        high=current_s,
        predicate=lambda s: margin_at(s) >= 0.0,
        refine_tol=refine_tol,
        max_refine_iter=max_refine_iter,
    )


def _find_side_contact_crossing(
    *,
    previous_s: float,
    current_s: float,
    previous_phi: float,
    current_phi: float,
    x0_mm: float,
    y0_mm: float,
    side_contact_risk_fn: SideContactRiskFn | None,
    refine_tol: float,
    max_refine_iter: int,
) -> float | None:
    if side_contact_risk_fn is None:
        return None
    if not (np.isfinite(previous_phi) and np.isfinite(current_phi)):
        return None
    previous_risk = bool(side_contact_risk_fn(previous_s, x0_mm + previous_s, y0_mm, previous_phi))
    current_risk = bool(side_contact_risk_fn(current_s, x0_mm + current_s, y0_mm, current_phi))
    if previous_risk:
        return previous_s
    if not current_risk:
        return None
    return _bisect_first_true(
        low=previous_s,
        high=current_s,
        predicate=lambda s: bool(
            side_contact_risk_fn(s, x0_mm + s, y0_mm, np.interp(s, [previous_s, current_s], [previous_phi, current_phi]))
        ),
        refine_tol=refine_tol,
        max_refine_iter=max_refine_iter,
    )


def _bisect_first_true(
    *,
    low: float,
    high: float,
    predicate: Callable[[float], bool],
    refine_tol: float,
    max_refine_iter: int,
) -> float:
    lo = float(low)
    hi = float(high)
    for _ in range(max_refine_iter):
        if hi - lo <= refine_tol:
            break
        mid = 0.5 * (lo + hi)
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def _min_optional(*values: float | None) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite:
        return None
    return min(finite)
