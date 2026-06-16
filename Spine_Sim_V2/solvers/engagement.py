"""接合角与单刺承载上限计算工具。"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, tan

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CapacityResult:
    """单刺切向承载能力及控制模式。"""

    cap_n: float
    cap_mode: str


def phi_s_deg_from_friction(f_s: float) -> float:
    """由静摩擦系数计算摩擦角，单位为度。"""
    if f_s < 0.0:
        raise ValueError("f_s must be non-negative.")
    return float(np.degrees(atan(f_s)))


def phi_hook_min_deg(alpha_p_deg: float, phi_s_deg: float) -> float:
    """由安装角和摩擦角给出最低钩取坡度诊断值。"""
    return float(max(0.0, 90.0 - alpha_p_deg - phi_s_deg))


def compute_critical_angle_deg(
    *,
    preload_n: float,
    target_force_n: float,
    f_s: float,
) -> float:
    """在局部预载 ``W_i`` 已知后计算临界接合角 ``phi_c``。

    当 ``W_i <= 0`` 时，该刺没有进入接合判定的物理前提，返回 NaN。
    """
    if preload_n <= 0.0:
        return float("nan")
    if target_force_n < 0.0:
        raise ValueError("target_force_n must be non-negative.")
    return float(np.degrees(atan(target_force_n / preload_n)) - phi_s_deg_from_friction(f_s))


def effective_contact_angle_map_deg(
    height_filtered: NDArray[np.floating],
    *,
    dx_mm: float,
    dy_mm: float,
) -> NDArray[np.float64]:
    """计算第一版可替换的有效接触角图。

    当前搜索方向为 +x；按模型文档的表面法向约定，二维近似下可退化为
    ``atan(dz/dx)``。后续如需更严格三维接触角算法，可替换本函数接口。
    """
    dz_dy, dz_dx = np.gradient(np.asarray(height_filtered, dtype=float), dy_mm, dx_mm)
    return np.degrees(np.arctan(dz_dx))


def side_contact_risk_from_angle(
    *,
    phi_eng_deg: float,
    spine_diameter_mm: float,
    tip_radius_mm: float,
) -> bool:
    """给出保守的杆身/侧接触几何风险标记。"""
    if not np.isfinite(phi_eng_deg):
        return False
    diameter_ratio = spine_diameter_mm / max(tip_radius_mm, 1e-9)
    angle_threshold = 82.0 if diameter_ratio <= 8.0 else 75.0
    return bool(abs(phi_eng_deg) >= angle_threshold)


def compute_capacity_n(
    *,
    preload_n: float,
    phi_eng_deg: float,
    f_s: float,
    F_ref_star_n: float,
) -> CapacityResult:
    """计算有限的单刺容量，禁止自锁区出现 ``tan`` 发散承载。"""
    if preload_n <= 0.0 or not np.isfinite(phi_eng_deg):
        return CapacityResult(cap_n=0.0, cap_mode="no_engagement")
    if F_ref_star_n < 0.0:
        raise ValueError("F_ref_star_n must be non-negative.")
    beta_deg = phi_eng_deg + phi_s_deg_from_friction(f_s)
    if beta_deg <= 0.0:
        return CapacityResult(cap_n=0.0, cap_mode="no_geometric_capacity")
    if beta_deg >= 90.0:
        # 自锁区只表示几何上不滑脱，承载仍受综合强度上限 F_ref_star_n 限制。
        return CapacityResult(cap_n=float(F_ref_star_n), cap_mode="self_lock_strength")
    f_geom = preload_n * tan(np.radians(beta_deg))
    if f_geom <= F_ref_star_n:
        return CapacityResult(cap_n=float(f_geom), cap_mode="geom")
    return CapacityResult(cap_n=float(F_ref_star_n), cap_mode="strength")
