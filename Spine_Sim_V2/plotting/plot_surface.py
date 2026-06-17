"""surface bank 审查图生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from Spine_Sim_V2.plotting.styles import PlotStyle, apply_plot_style, load_plot_style, require_columns
from Spine_Sim_V2.solvers.engagement import effective_contact_angle_map_deg
from Spine_Sim_V2.surfaces.audit import audit_surface_bank, sample_surface_ids_by_kind
from Spine_Sim_V2.surfaces.bank import SurfaceBank


def plot_surface_audit(
    *,
    surface_bank: str | Path | SurfaceBank,
    sample_per_kind: int = 2,
    outdir: str | Path,
    seed: int = 20260616,
    style: str | Path = "report",
) -> dict[str, Path]:
    """从 surface bank 数据生成默认审查图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style = load_plot_style(style)
    apply_plot_style(plt, plot_style)

    bank = surface_bank if isinstance(surface_bank, SurfaceBank) else SurfaceBank.open(surface_bank)
    audit = audit_surface_bank(bank)
    if not audit["valid"]:
        raise ValueError(f"Surface bank audit failed: {audit}")

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = bank.load_statistics()
    require_columns(
        stats,
        ["surface_kind", "slope_p95_deg", "candidate_density_preload_free"],
        dataset_name="surface_statistics.parquet",
    )
    if sample_per_kind <= 0:
        raise ValueError("sample_per_kind must be positive.")
    samples = sample_surface_ids_by_kind(bank, sample_per_kind=sample_per_kind, seed=seed)

    outputs: dict[str, Path] = {}
    for surface_kind in sorted(samples):
        surface_ids = samples[surface_kind]
        if not surface_ids:
            continue
        safe_kind = _safe_stem(surface_kind)
        for array_key, variant_label in (
            ("height_raw", "raw"),
            ("height_filtered", "filtered"),
        ):
            stem = f"{safe_kind}_{variant_label}_height_angle"
            outputs[stem] = plot_style.path(output_dir, stem)
            _plot_height_angle_grid(
                bank=bank,
                surface_ids=surface_ids,
                surface_kind=surface_kind,
                array_key=array_key,
                variant_label=variant_label,
                path=outputs[stem],
                plt=plt,
                plot_style=plot_style,
            )

    outputs["slope_distribution_by_surface"] = plot_style.path(
        output_dir,
        "slope_distribution_by_surface",
    )
    outputs["candidate_density_distribution"] = plot_style.path(
        output_dir,
        "candidate_density_distribution",
    )
    _plot_box_by_kind(
        stats=stats,
        value_column="slope_p95_deg",
        ylabel="Slope p95 (deg)",
        title="Slope Distribution by Surface Kind",
        path=outputs["slope_distribution_by_surface"],
        plt=plt,
        plot_style=plot_style,
    )
    _plot_box_by_kind(
        stats=stats,
        value_column="candidate_density_preload_free",
        ylabel="Candidate density (1/mm^2)",
        title="Preload-Free Candidate Density",
        path=outputs["candidate_density_distribution"],
        plt=plt,
        plot_style=plot_style,
    )
    return outputs


def _plot_height_angle_grid(
    *,
    bank: SurfaceBank,
    surface_ids: list[str],
    surface_kind: str,
    array_key: Literal["height_raw", "height_filtered"],
    variant_label: str,
    path: Path,
    plt: object,
    plot_style: PlotStyle,
) -> None:
    panels = []
    for surface_id in surface_ids:
        surface_record = bank.get_surface_record(surface_id)
        arrays = bank.load_surface_arrays(surface_id)
        height = np.asarray(arrays[array_key], dtype=float)
        dx_mm = float(surface_record["dx_mm"])
        dy_mm = float(surface_record["dy_mm"])
        angle = effective_contact_angle_map_deg(height, dx_mm=dx_mm, dy_mm=dy_mm)
        ny, nx = height.shape
        panels.append(
            {
                "surface_id": surface_id,
                "height": height,
                "angle": angle,
                "extent": [0.0, nx * dx_mm, 0.0, ny * dy_mm],
            }
        )

    height_limit = _symmetric_percentile_limit(
        np.concatenate([panel["height"].ravel() for panel in panels]),
        percentile=98.0,
    )
    angle_limit = _symmetric_percentile_limit(
        np.concatenate([panel["angle"].ravel() for panel in panels]),
        percentile=98.0,
    )
    height_title = "raw height (mm)" if array_key == "height_raw" else "probe-filtered height (mm)"

    fig, axes = plt.subplots(
        len(panels),
        2,
        figsize=plot_style.scaled_size(
            width_scale=2.4,
            height_scale=max(1.05, 0.82 * len(panels)),
        ),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(f"{surface_kind} / {variant_label}", fontsize=plot_style.font_size + 1)

    height_images = []
    angle_images = []
    for row, panel in enumerate(panels):
        height_ax = axes[row][0]
        height_image = height_ax.imshow(
            panel["height"],
            cmap="terrain",
            origin="lower",
            extent=panel["extent"],
            vmin=-height_limit,
            vmax=height_limit,
            interpolation="nearest",
            aspect="equal",
        )
        if row == 0:
            height_ax.set_title(height_title)
        _format_map_axes(height_ax)
        height_ax.text(
            0.02,
            0.98,
            panel["surface_id"],
            transform=height_ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(plot_style.font_size - 2, 6),
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2.0},
        )
        height_images.append(height_image)

        angle_ax = axes[row][1]
        angle_image = angle_ax.imshow(
            panel["angle"],
            cmap="coolwarm",
            origin="lower",
            extent=panel["extent"],
            vmin=-angle_limit,
            vmax=angle_limit,
            interpolation="nearest",
            aspect="equal",
        )
        if row == 0:
            angle_ax.set_title("effective contact angle (deg)")
        _format_map_axes(angle_ax)
        angle_ax.text(
            0.02,
            0.98,
            panel["surface_id"],
            transform=angle_ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(plot_style.font_size - 2, 6),
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 2.0},
        )
        angle_images.append(angle_image)

    height_colorbar = fig.colorbar(height_images[0], ax=axes[:, 0], fraction=0.046, pad=0.04)
    height_colorbar.set_label("height (mm)")
    angle_colorbar = fig.colorbar(angle_images[0], ax=axes[:, 1], fraction=0.046, pad=0.04)
    angle_colorbar.set_label("effective contact angle (deg)")

    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _format_map_axes(ax: object) -> None:
    ax.set_xlabel("x / mm")
    ax.set_ylabel("y / mm")
    ax.tick_params(direction="out")


def _symmetric_percentile_limit(values: np.ndarray, *, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    limit = float(np.nanpercentile(np.abs(finite), percentile))
    return max(limit, 1e-6)


def _safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _plot_box_by_kind(
    *,
    stats: object,
    value_column: str,
    ylabel: str,
    title: str,
    path: Path,
    plt: object,
    plot_style: PlotStyle,
) -> None:
    kinds = sorted(str(kind) for kind in stats["surface_kind"].unique())
    values = [
        stats.loc[stats["surface_kind"] == kind, value_column].dropna().to_numpy()
        for kind in kinds
    ]
    fig, ax = plt.subplots(
        figsize=plot_style.scaled_size(width_scale=max(1.0, 0.23 * len(kinds))),
        constrained_layout=True,
    )
    try:
        ax.boxplot(values, tick_labels=kinds, showmeans=True)
    except TypeError:  # Matplotlib < 3.9 compatibility.
        ax.boxplot(values, labels=kinds, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)
