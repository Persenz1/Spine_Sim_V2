"""P3：刚性单刺安装角验证。"""

from __future__ import annotations

from pathlib import Path

from Spine_Sim_V2.pipelines.p2_compliant_k_alpha import (
    P2_SURFACE_KINDS,
    P2_W_TOTAL_N,
    _run_screen,
)


P3_PROJECT_NAME = "P3_rigid_alpha_screen"
P3_ALPHA_P_DEG = (50.0, 60.0, 70.0, 80.0)


def run(
    *,
    surface_bank: str | Path,
    n_surfaces_per_kind: int = 100,
    outdir: str | Path = "outputs/P3_rigid_alpha_screen",
    alpha_values: tuple[float, ...] = P3_ALPHA_P_DEG,
    w_values: tuple[float, ...] = P2_W_TOTAL_N,
    surface_kinds: tuple[str, ...] = P2_SURFACE_KINDS,
    workers: int | None = None,
) -> Path:
    """运行 P3 刚性单刺 ``alpha_p_deg`` 扫描。"""
    return _run_screen(
        project_name=P3_PROJECT_NAME,
        stage_name="p3_rigid_alpha",
        surface_bank=surface_bank,
        n_surfaces_per_kind=n_surfaces_per_kind,
        outdir=outdir,
        array_type="rigid",
        spring_values=None,
        alpha_values=alpha_values,
        w_values=w_values,
        surface_kinds=surface_kinds,
        workers=workers,
    )
