"""代理表面类别参数定义。

这里的数值只用于筛选和调试的合成表面类别，不是真实材料参数，也不能作为
墙面绝对承载预测的标定值。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceProfile:
    """一类代理表面的统计生成控制参数。"""

    surface_kind: str
    rq_mm: float
    corr_length_mm: float
    description: str


DEFAULT_SURFACE_PROFILES: dict[str, SurfaceProfile] = {
    "sandpaper": SurfaceProfile(
        surface_kind="sandpaper",
        rq_mm=0.10,
        corr_length_mm=0.45,
        description="Fine, dense roughness proxy surface.",
    ),
    "concrete": SurfaceProfile(
        surface_kind="concrete",
        rq_mm=0.22,
        corr_length_mm=1.20,
        description="Coarse mineral aggregate proxy surface.",
    ),
    "brick": SurfaceProfile(
        surface_kind="brick",
        rq_mm=0.12,
        corr_length_mm=1.70,
        description="Broad masonry texture proxy surface.",
    ),
    "painted_wall": SurfaceProfile(
        surface_kind="painted_wall",
        rq_mm=0.07,
        corr_length_mm=0.90,
        description="Smoothed wall coating proxy surface.",
    ),
}


def list_builtin_profiles() -> list[str]:
    """返回内置代理表面类别。"""
    return list(DEFAULT_SURFACE_PROFILES)


def get_surface_profile(surface_kind: str) -> SurfaceProfile:
    """按类别取表面参数；未知类别给出清晰错误。"""
    try:
        return DEFAULT_SURFACE_PROFILES[surface_kind]
    except KeyError as exc:
        known = ", ".join(list_builtin_profiles())
        raise ValueError(f"Unknown surface_kind {surface_kind!r}. Known kinds: {known}.") from exc


def parse_surface_kinds(value: str | list[str] | tuple[str, ...]) -> list[str]:
    """解析逗号分隔的表面类别列表，并逐项校验。"""
    if isinstance(value, str):
        kinds = [item.strip() for item in value.split(",") if item.strip()]
    else:
        kinds = [str(item).strip() for item in value if str(item).strip()]
    if not kinds:
        raise ValueError("At least one surface kind is required.")
    for kind in kinds:
        get_surface_profile(kind)
    return kinds
