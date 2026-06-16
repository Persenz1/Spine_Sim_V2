"""Unit conversion helpers.

Lengths are stored in mm, forces in N, and external angles in deg.
Spring stiffness is accepted as N/m and converted internally to N/mm.
"""

from __future__ import annotations

from math import degrees, radians


def spring_k_n_per_m_to_n_per_mm(value_n_per_m: float | None) -> float | None:
    """Convert spring stiffness from N/m to N/mm.

    Rigid arrays must use ``None`` for spring stiffness, not ``0``. Passing
    ``None`` through this conversion keeps that representation intact.
    """
    if value_n_per_m is None:
        return None
    return value_n_per_m / 1000.0


def spring_k_n_per_mm_to_n_per_m(value_n_per_mm: float | None) -> float | None:
    """Convert spring stiffness from N/mm back to N/m."""
    if value_n_per_mm is None:
        return None
    return value_n_per_mm * 1000.0


def deg_to_rad(value_deg: float) -> float:
    """Convert degrees to radians for internal calculations."""
    return radians(value_deg)


def rad_to_deg(value_rad: float) -> float:
    """Convert radians to degrees for saved outputs."""
    return degrees(value_rad)
