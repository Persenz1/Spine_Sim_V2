"""Array geometry, interpolation, and normal preload distribution solvers."""

from __future__ import annotations

from dataclasses import dataclass
from math import sin
from typing import Any

import numpy as np
from numpy.typing import NDArray

from Spine_Sim_V2.core.types import SingleCaseInput, StiffnessModel
from Spine_Sim_V2.core.units import deg_to_rad, spring_k_n_per_m_to_n_per_mm


AXIAL_STROKE_MAX_MM = 4.0
RIGID_NUMERICAL_PENALTY_N_PER_MM = 1.0e6


@dataclass(frozen=True)
class PreloadResult:
    """Preload solution for an array."""

    delta_n_mm: float
    gap_mm: NDArray[np.float64]
    height_at_spines_mm: NDArray[np.float64]
    preload_n: NDArray[np.float64]
    contacted: NDArray[np.bool_]
    u_ax_used_mm: NDArray[np.float64]
    normal_saturated: NDArray[np.bool_]
    normal_range_insufficient: bool
    warning_flags: tuple[str, ...]


def build_stiffness_model(case: SingleCaseInput) -> StiffnessModel:
    """Build stiffness values for rigid or compliant arrays."""
    array_type = case.array_type.lower()
    alpha_rad = deg_to_rad(case.alpha_p_deg)
    sin_alpha = sin(alpha_rad)
    if not (0.0 < case.alpha_p_deg < 90.0):
        raise ValueError("alpha_p_deg must be in the open interval (0, 90).")
    if array_type == "rigid":
        if case.spring_k_n_per_m is not None:
            raise ValueError("Rigid arrays must use spring_k_n_per_m=None.")
        return StiffnessModel(
            spring_k_n_per_m=None,
            spring_k_n_per_mm=None,
            k_nn=None,
            k_tt=None,
            k_tn=None,
            axial_stroke_max_mm=None,
            normal_stroke_max_mm=None,
        )
    if array_type != "compliant":
        raise ValueError("array_type must be 'rigid' or 'compliant'.")
    if case.spring_k_n_per_m is None:
        raise ValueError("Compliant arrays require spring_k_n_per_m.")
    spring_k_n_per_mm = spring_k_n_per_m_to_n_per_mm(case.spring_k_n_per_m)
    if spring_k_n_per_mm is None or spring_k_n_per_mm <= 0.0:
        raise ValueError("Compliant spring_k_n_per_m must be positive.")
    cos_alpha = float(np.cos(alpha_rad))
    return StiffnessModel(
        spring_k_n_per_m=float(case.spring_k_n_per_m),
        spring_k_n_per_mm=float(spring_k_n_per_mm),
        k_nn=float(spring_k_n_per_mm * sin_alpha**2),
        k_tt=float(spring_k_n_per_mm * cos_alpha**2),
        k_tn=float(spring_k_n_per_mm * sin_alpha * cos_alpha),
        axial_stroke_max_mm=AXIAL_STROKE_MAX_MM,
        normal_stroke_max_mm=float(AXIAL_STROKE_MAX_MM / sin_alpha),
    )


def build_rectangular_array_geometry(
    *,
    rows: int,
    cols: int,
    pitch_t_mm: float,
    pitch_l_mm: float,
    surface_shape: tuple[int, int],
    dx_mm: float,
    dy_mm: float,
) -> list[dict[str, Any]]:
    """Build centered rectangular array coordinates on a surface grid."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")
    if pitch_t_mm <= 0.0 or pitch_l_mm <= 0.0:
        raise ValueError("pitch_t_mm and pitch_l_mm must be positive.")
    ny, nx = surface_shape
    center_x_mm = 0.5 * (nx - 1) * dx_mm
    center_y_mm = 0.5 * (ny - 1) * dy_mm
    geometry: list[dict[str, Any]] = []
    for row in range(rows):
        for col in range(cols):
            x_mm = center_x_mm + (col - 0.5 * (cols - 1)) * pitch_t_mm
            y_mm = center_y_mm + (row - 0.5 * (rows - 1)) * pitch_l_mm
            geometry.append(
                {
                    "spine_id": f"r{row:03d}_c{col:03d}",
                    "row": row,
                    "col": col,
                    "x_mm": float(x_mm),
                    "y_mm": float(y_mm),
                }
            )
    return geometry


def interpolate_bilinear(
    values: NDArray[np.floating],
    x_mm: float,
    y_mm: float,
    *,
    dx_mm: float,
    dy_mm: float,
) -> float:
    """Bilinearly interpolate a regular height/slope grid."""
    arr = np.asarray(values, dtype=float)
    ny, nx = arr.shape
    gx = x_mm / dx_mm
    gy = y_mm / dy_mm
    if gx < 0.0 or gy < 0.0 or gx > nx - 1 or gy > ny - 1:
        return float("nan")
    x0 = int(np.floor(gx))
    y0 = int(np.floor(gy))
    x1 = min(x0 + 1, nx - 1)
    y1 = min(y0 + 1, ny - 1)
    tx = gx - x0
    ty = gy - y0
    z00 = arr[y0, x0]
    z10 = arr[y0, x1]
    z01 = arr[y1, x0]
    z11 = arr[y1, x1]
    return float(
        (1.0 - tx) * (1.0 - ty) * z00
        + tx * (1.0 - ty) * z10
        + (1.0 - tx) * ty * z01
        + tx * ty * z11
    )


def solve_preload_distribution(
    *,
    case: SingleCaseInput,
    stiffness: StiffnessModel,
    height_filtered: NDArray[np.floating],
    geometry: list[dict[str, Any]],
    dx_mm: float,
    dy_mm: float,
    k_penalty_n_per_mm: float = RIGID_NUMERICAL_PENALTY_N_PER_MM,
) -> PreloadResult:
    """Solve local preload W_i by monotonic root finding for delta_n."""
    if case.w_total_n < 0.0:
        raise ValueError("w_total_n must be non-negative.")
    heights = np.asarray(
        [
            interpolate_bilinear(
                height_filtered,
                item["x_mm"],
                item["y_mm"],
                dx_mm=dx_mm,
                dy_mm=dy_mm,
            )
            for item in geometry
        ],
        dtype=float,
    )
    warning_flags: list[str] = []
    finite_heights = np.isfinite(heights)
    if not np.all(finite_heights):
        warning_flags.append("array_outside_surface")
    if not np.any(finite_heights):
        raise ValueError("No spine projection lies inside the surface grid.")

    h_max = float(np.nanmax(heights))
    gaps = h_max - heights
    gaps[~finite_heights] = np.inf

    if case.w_total_n == 0.0:
        zeros = np.zeros(len(geometry), dtype=float)
        return PreloadResult(
            delta_n_mm=0.0,
            gap_mm=gaps,
            height_at_spines_mm=heights,
            preload_n=zeros,
            contacted=np.zeros(len(geometry), dtype=bool),
            u_ax_used_mm=zeros,
            normal_saturated=np.zeros(len(geometry), dtype=bool),
            normal_range_insufficient=False,
            warning_flags=tuple(warning_flags),
        )

    if case.array_type.lower() == "rigid":
        return _solve_rigid_preload(
            gaps=gaps,
            heights=heights,
            w_total_n=case.w_total_n,
            k_penalty_n_per_mm=k_penalty_n_per_mm,
            warning_flags=tuple(warning_flags),
        )
    return _solve_compliant_preload(
        case=case,
        stiffness=stiffness,
        gaps=gaps,
        heights=heights,
        warning_flags=tuple(warning_flags),
    )


def _solve_rigid_preload(
    *,
    gaps: NDArray[np.float64],
    heights: NDArray[np.float64],
    w_total_n: float,
    k_penalty_n_per_mm: float,
    warning_flags: tuple[str, ...],
) -> PreloadResult:
    if k_penalty_n_per_mm <= 0.0:
        raise ValueError("k_penalty_n_per_mm must be positive.")

    def total(delta_n_mm: float) -> float:
        compression = np.maximum(delta_n_mm - gaps, 0.0)
        compression[~np.isfinite(compression)] = 0.0
        return float(k_penalty_n_per_mm * np.sum(compression))

    high = _bracket_monotonic_total(total, w_total_n, initial_high=float(np.nanmin(gaps)) + 1e-6)
    delta = _bisect_monotonic_total(total, target=w_total_n, low=0.0, high=high)
    compression = np.maximum(delta - gaps, 0.0)
    compression[~np.isfinite(compression)] = 0.0
    preload = k_penalty_n_per_mm * compression
    return PreloadResult(
        delta_n_mm=float(delta),
        gap_mm=gaps,
        height_at_spines_mm=heights,
        preload_n=preload,
        contacted=preload > 1e-12,
        u_ax_used_mm=np.zeros_like(preload),
        normal_saturated=np.zeros_like(preload, dtype=bool),
        normal_range_insufficient=False,
        warning_flags=warning_flags,
    )


def _solve_compliant_preload(
    *,
    case: SingleCaseInput,
    stiffness: StiffnessModel,
    gaps: NDArray[np.float64],
    heights: NDArray[np.float64],
    warning_flags: tuple[str, ...],
) -> PreloadResult:
    if stiffness.spring_k_n_per_mm is None:
        raise ValueError("Compliant preload requires spring_k_n_per_mm.")
    alpha_rad = deg_to_rad(case.alpha_p_deg)
    sin_alpha = float(np.sin(alpha_rad))
    k_s = stiffness.spring_k_n_per_mm

    def preload_at(delta_n_mm: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        c_i = np.maximum(delta_n_mm - gaps, 0.0)
        c_i[~np.isfinite(c_i)] = 0.0
        u_ax = c_i * sin_alpha
        u_clamped = np.minimum(u_ax, AXIAL_STROKE_MAX_MM)
        preload = k_s * u_clamped * sin_alpha
        saturated = (u_ax >= AXIAL_STROKE_MAX_MM - 1e-12) & (c_i > 0.0)
        return preload, u_clamped, saturated

    max_preload, max_u_ax, max_saturated = preload_at(float(np.nanmax(gaps[np.isfinite(gaps)])) + AXIAL_STROKE_MAX_MM / sin_alpha + 1.0)
    normal_range_insufficient = float(np.sum(max_preload)) + 1e-9 < case.w_total_n
    if normal_range_insufficient:
        warning_flags = tuple([*warning_flags, "normal_range_insufficient"])
        return PreloadResult(
            delta_n_mm=float(np.nanmax(gaps[np.isfinite(gaps)])) + AXIAL_STROKE_MAX_MM / sin_alpha,
            gap_mm=gaps,
            height_at_spines_mm=heights,
            preload_n=max_preload,
            contacted=max_preload > 1e-12,
            u_ax_used_mm=max_u_ax,
            normal_saturated=max_saturated,
            normal_range_insufficient=True,
            warning_flags=warning_flags,
        )

    def total(delta_n_mm: float) -> float:
        preload, _, _ = preload_at(delta_n_mm)
        return float(np.sum(preload))

    high = _bracket_monotonic_total(total, case.w_total_n, initial_high=float(np.nanmin(gaps[np.isfinite(gaps)])) + 1e-6)
    delta = _bisect_monotonic_total(total, target=case.w_total_n, low=0.0, high=high)
    preload, u_ax, saturated = preload_at(delta)
    return PreloadResult(
        delta_n_mm=float(delta),
        gap_mm=gaps,
        height_at_spines_mm=heights,
        preload_n=preload,
        contacted=preload > 1e-12,
        u_ax_used_mm=u_ax,
        normal_saturated=saturated,
        normal_range_insufficient=False,
        warning_flags=warning_flags,
    )


def _bracket_monotonic_total(
    total_fn: Any,
    target: float,
    *,
    initial_high: float,
) -> float:
    high = max(float(initial_high), 1e-9)
    while total_fn(high) < target:
        high = max(2.0 * high, high + 1.0)
        if high > 1.0e6:
            raise RuntimeError("Failed to bracket monotonic preload solution.")
    return high


def _bisect_monotonic_total(
    total_fn: Any,
    *,
    target: float,
    low: float,
    high: float,
    max_iter: int = 80,
) -> float:
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if total_fn(mid) < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
