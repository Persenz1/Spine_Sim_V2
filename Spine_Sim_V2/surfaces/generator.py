"""surface bank 使用的合成代理表面生成器。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from numpy.typing import NDArray

from Spine_Sim_V2.surfaces.profiles import SurfaceProfile, get_surface_profile


SURFACE_GENERATOR_VERSION = "multiscale_proxy_v001"
PROBE_FILTER_VERSION = "fft_gaussian_probe_approx_v001"


@dataclass(frozen=True)
class GeneratedSurface:
    """生成的原始/滤波高度图及审查统计信息。"""

    surface_id: str
    surface_kind: str
    seed: int
    dx_mm: float
    dy_mm: float
    size_x_mm: float
    size_y_mm: float
    tip_radius_mm: float
    height_raw: NDArray[np.float32]
    height_filtered: NDArray[np.float32]
    statistics: dict[str, object]


def stable_surface_seed(
    bank_id: str,
    surface_kind: str,
    index: int,
    base_seed: int,
) -> int:
    """生成不受 Python hash 随机化影响的可复现 32 位种子。"""
    digest = sha256(f"{bank_id}:{surface_kind}:{index}:{base_seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def make_surface_id(surface_kind: str, index: int) -> str:
    """返回用于 NPZ 文件名的规范 ``surface_id``。"""
    return f"{surface_kind}_{index:06d}"


def generate_surface(
    *,
    surface_kind: str,
    surface_id: str,
    seed: int,
    resolution_cells_per_mm: int = 5,
    size_x_mm: float = 60.0,
    size_y_mm: float = 40.0,
    tip_radius_mm: float = 0.05,
) -> GeneratedSurface:
    """生成一张合成代理表面。

    生成器刻意保持为代理模型：先组合相关粗糙度、凸峰、凹坑、阶跃纹理和细纹理，
    再归一化到目标代理 ``Rq``。这些参数不代表真实材料标定值。
    """
    if resolution_cells_per_mm <= 0:
        raise ValueError("resolution_cells_per_mm must be positive.")
    if size_x_mm <= 0 or size_y_mm <= 0:
        raise ValueError("size_x_mm and size_y_mm must be positive.")
    if tip_radius_mm < 0:
        raise ValueError("tip_radius_mm must be non-negative.")

    profile = get_surface_profile(surface_kind)
    dx_mm = 1.0 / float(resolution_cells_per_mm)
    dy_mm = dx_mm
    nx = max(2, int(round(size_x_mm * resolution_cells_per_mm)))
    ny = max(2, int(round(size_y_mm * resolution_cells_per_mm)))
    rng = np.random.default_rng(seed)

    # 多尺度地形只负责提供统计形貌；真实承载解释必须等后续接触/搜索模块完成。
    height_raw = _generate_multiscale_height(
        shape=(ny, nx),
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        profile=profile,
        rng=rng,
    )
    height_raw = _normalize_to_rq(height_raw, profile.rq_mm).astype(np.float32)
    height_filtered = probe_filter(height_raw, tip_radius_mm, dx_mm=dx_mm, dy_mm=dy_mm)
    statistics = compute_surface_statistics(
        height_raw=height_raw,
        height_filtered=height_filtered,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        size_x_mm=size_x_mm,
        size_y_mm=size_y_mm,
        tip_radius_mm=tip_radius_mm,
    )

    return GeneratedSurface(
        surface_id=surface_id,
        surface_kind=surface_kind,
        seed=seed,
        dx_mm=dx_mm,
        dy_mm=dy_mm,
        size_x_mm=size_x_mm,
        size_y_mm=size_y_mm,
        tip_radius_mm=tip_radius_mm,
        height_raw=height_raw,
        height_filtered=height_filtered.astype(np.float32),
        statistics=statistics,
    )


def probe_filter(
    height_raw: NDArray[np.floating],
    tip_radius_mm: float,
    *,
    dx_mm: float,
    dy_mm: float,
) -> NDArray[np.float32]:
    """近似实现有限刺尖半径探针滤波。

    第一版用 FFT 高斯低通衰减小于刺尖尺度的空间波长。后续可以在保持函数签名
    不变的前提下，替换为更严格的形态学滚动探针实现。
    """
    if tip_radius_mm <= 0:
        return np.asarray(height_raw, dtype=np.float32).copy()
    sigma_mm = max(float(tip_radius_mm), 0.25 * max(dx_mm, dy_mm))
    filtered = _gaussian_lowpass(np.asarray(height_raw, dtype=float), dx_mm, dy_mm, sigma_mm)
    filtered -= float(np.mean(filtered))
    return filtered.astype(np.float32)


def compute_surface_statistics(
    *,
    height_raw: NDArray[np.floating],
    height_filtered: NDArray[np.floating],
    dx_mm: float,
    dy_mm: float,
    size_x_mm: float,
    size_y_mm: float,
    tip_radius_mm: float,
) -> dict[str, object]:
    """计算 surface bank 审查统计字段。"""
    raw = np.asarray(height_raw, dtype=float)
    eff = np.asarray(height_filtered, dtype=float)
    slope_deg = _slope_degrees(eff, dx_mm=dx_mm, dy_mm=dy_mm)
    return {
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "size_x_mm": size_x_mm,
        "size_y_mm": size_y_mm,
        "tip_radius_mm": tip_radius_mm,
        "rq_raw_mm": _rq(raw),
        "ra_raw_mm": _ra(raw),
        "hpv_raw_mm": _hpv(raw),
        "rq_eff_mm": _rq(eff),
        "ra_eff_mm": _ra(eff),
        "hpv_eff_mm": _hpv(eff),
        "slope_mean_deg": float(np.mean(slope_deg)),
        "slope_p50_deg": float(np.percentile(slope_deg, 50.0)),
        "slope_p95_deg": float(np.percentile(slope_deg, 95.0)),
        "slope_max_deg": float(np.max(slope_deg)),
        "candidate_density_preload_free": _candidate_density_preload_free(
            eff,
            slope_deg=slope_deg,
            size_x_mm=size_x_mm,
            size_y_mm=size_y_mm,
        ),
        "valid": bool(np.all(np.isfinite(raw)) and np.all(np.isfinite(eff)) and _rq(raw) > 0.0),
        "reject_reason": None,
    }


def _generate_multiscale_height(
    *,
    shape: tuple[int, int],
    dx_mm: float,
    dy_mm: float,
    profile: SurfaceProfile,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    lc = profile.corr_length_mm
    h0 = _correlated_noise(shape, dx_mm, dy_mm, corr_length_mm=lc, rng=rng)
    h_peak = _impulse_texture(
        shape,
        dx_mm,
        dy_mm,
        corr_length_mm=max(0.20 * lc, 1.5 * dx_mm),
        density_per_mm2=_peak_density(profile.surface_kind, lc),
        rng=rng,
        positive=True,
    )
    h_pit = _impulse_texture(
        shape,
        dx_mm,
        dy_mm,
        corr_length_mm=max(0.28 * lc, 1.5 * dx_mm),
        density_per_mm2=0.75 * _peak_density(profile.surface_kind, lc),
        rng=rng,
        positive=False,
    )
    h_step = _step_texture(shape, dx_mm=dx_mm, dy_mm=dy_mm, corr_length_mm=lc, rng=rng)
    h_texture = _correlated_noise(
        shape,
        dx_mm,
        dy_mm,
        corr_length_mm=max(0.12 * lc, dx_mm),
        rng=rng,
    )
    height = 0.58 * h0 + 0.30 * h_peak + 0.26 * h_pit + 0.18 * h_step + 0.10 * h_texture
    height -= float(np.mean(height))
    return height


def _peak_density(surface_kind: str, corr_length_mm: float) -> float:
    kind_factor = {
        "sandpaper": 1.40,
        "concrete": 0.55,
        "brick": 0.35,
        "painted_wall": 0.70,
    }.get(surface_kind, 0.60)
    return kind_factor / max(corr_length_mm**2, 0.05)


def _correlated_noise(
    shape: tuple[int, int],
    dx_mm: float,
    dy_mm: float,
    *,
    corr_length_mm: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    noise = rng.normal(0.0, 1.0, size=shape)
    return _standardize(_gaussian_lowpass(noise, dx_mm, dy_mm, sigma_mm=corr_length_mm))


def _impulse_texture(
    shape: tuple[int, int],
    dx_mm: float,
    dy_mm: float,
    *,
    corr_length_mm: float,
    density_per_mm2: float,
    rng: np.random.Generator,
    positive: bool,
) -> NDArray[np.float64]:
    ny, nx = shape
    area_mm2 = nx * dx_mm * ny * dy_mm
    n_impulses = int(rng.poisson(max(1.0, density_per_mm2 * area_mm2)))
    impulses = np.zeros(shape, dtype=float)
    ys = rng.integers(0, ny, size=n_impulses)
    xs = rng.integers(0, nx, size=n_impulses)
    amplitudes = rng.lognormal(mean=0.0, sigma=0.45, size=n_impulses)
    if not positive:
        amplitudes *= -1.0
    np.add.at(impulses, (ys, xs), amplitudes)
    return _standardize(_gaussian_lowpass(impulses, dx_mm, dy_mm, sigma_mm=corr_length_mm))


def _step_texture(
    shape: tuple[int, int],
    *,
    dx_mm: float,
    dy_mm: float,
    corr_length_mm: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    ny, nx = shape
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx_mm
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy_mm
    xx, yy = np.meshgrid(x, y)
    texture = np.zeros(shape, dtype=float)
    n_steps = int(rng.integers(1, 4))
    span = max(np.max(np.abs(x)), np.max(np.abs(y)), corr_length_mm)
    for _ in range(n_steps):
        theta = rng.uniform(0.0, np.pi)
        offset = rng.uniform(-0.6 * span, 0.6 * span)
        width = rng.uniform(0.25 * corr_length_mm, 0.90 * corr_length_mm)
        amplitude = rng.normal(0.0, 1.0)
        signed_distance = xx * np.cos(theta) + yy * np.sin(theta) - offset
        texture += amplitude * np.tanh(signed_distance / max(width, 1e-6))
    return _standardize(texture)


def _gaussian_lowpass(
    values: NDArray[np.floating],
    dx_mm: float,
    dy_mm: float,
    sigma_mm: float,
) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    ny, nx = arr.shape
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx_mm)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy_mm)
    kk_y, kk_x = np.meshgrid(ky, kx, indexing="ij")
    filt = np.exp(-0.5 * sigma_mm**2 * (kk_x**2 + kk_y**2))
    transformed = np.fft.rfftn(arr, axes=(0, 1))
    smoothed = np.fft.irfftn(transformed * filt, s=arr.shape, axes=(0, 1))
    return np.asarray(smoothed, dtype=float)


def _normalize_to_rq(values: NDArray[np.floating], target_rq_mm: float) -> NDArray[np.float64]:
    standardized = _standardize(values)
    return standardized * target_rq_mm


def _standardize(values: NDArray[np.floating]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    arr = arr - float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 1e-12:
        return np.zeros_like(arr, dtype=float)
    return arr / std


def _rq(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    return float(np.sqrt(np.mean(centered**2)))


def _ra(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    return float(np.mean(np.abs(centered)))


def _hpv(values: NDArray[np.floating]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.max(arr) - np.min(arr))


def _slope_degrees(
    height_mm: NDArray[np.floating],
    *,
    dx_mm: float,
    dy_mm: float,
) -> NDArray[np.float64]:
    dz_dy, dz_dx = np.gradient(np.asarray(height_mm, dtype=float), dy_mm, dx_mm)
    slope = np.sqrt(dz_dx**2 + dz_dy**2)
    return np.degrees(np.arctan(slope))


def _candidate_density_preload_free(
    height_mm: NDArray[np.floating],
    *,
    slope_deg: NDArray[np.floating],
    size_x_mm: float,
    size_y_mm: float,
) -> float:
    """返回仅用于表面库审查的几何局部峰密度。

    这不是接合判定，也不使用 ``W_i`` 或 ``phi_c,i``；真实可接合区域仍要等
    单 case 中局部预载求解完成后再判断。
    """
    z = np.asarray(height_mm, dtype=float)
    if z.shape[0] < 3 or z.shape[1] < 3:
        return 0.0
    center = z[1:-1, 1:-1]
    neighbors = [
        z[:-2, :-2],
        z[:-2, 1:-1],
        z[:-2, 2:],
        z[1:-1, :-2],
        z[1:-1, 2:],
        z[2:, :-2],
        z[2:, 1:-1],
        z[2:, 2:],
    ]
    local_peak = np.ones_like(center, dtype=bool)
    for neighbor in neighbors:
        local_peak &= center > neighbor
    slope_inner = np.asarray(slope_deg, dtype=float)[1:-1, 1:-1]
    slope_threshold = max(5.0, float(np.percentile(slope_inner, 65.0)))
    candidates = local_peak & (slope_inner >= slope_threshold)
    area_mm2 = max(size_x_mm * size_y_mm, 1e-12)
    return float(np.count_nonzero(candidates) / area_mm2)
