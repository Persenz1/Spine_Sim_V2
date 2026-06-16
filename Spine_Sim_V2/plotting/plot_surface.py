"""Surface bank audit plots."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from Spine_Sim_V2.plotting.styles import PlotStyle, apply_plot_style, load_plot_style, require_columns
from Spine_Sim_V2.surfaces.audit import audit_surface_bank, sample_surface_ids_by_kind
from Spine_Sim_V2.surfaces.bank import SurfaceBank


def plot_surface_audit(
    *,
    surface_bank: str | Path | SurfaceBank,
    sample_per_kind: int = 8,
    outdir: str | Path,
    seed: int = 20260616,
    style: str | Path = "report",
) -> dict[str, Path]:
    """Generate the default Phase 2 surface bank audit figures."""
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
    samples = sample_surface_ids_by_kind(bank, sample_per_kind=sample_per_kind, seed=seed)

    outputs = {
        "surface_gallery": plot_style.path(output_dir, "surface_gallery"),
        "filtered_surface_gallery": plot_style.path(output_dir, "filtered_surface_gallery"),
        "slope_distribution_by_surface": plot_style.path(output_dir, "slope_distribution_by_surface"),
        "candidate_density_distribution": plot_style.path(output_dir, "candidate_density_distribution"),
    }
    _plot_gallery(
        bank=bank,
        samples=samples,
        array_key="height_raw",
        title="Raw Surface Height",
        path=outputs["surface_gallery"],
        plt=plt,
        plot_style=plot_style,
    )
    _plot_gallery(
        bank=bank,
        samples=samples,
        array_key="height_filtered",
        title="Filtered Surface Height",
        path=outputs["filtered_surface_gallery"],
        plt=plt,
        plot_style=plot_style,
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


def _plot_gallery(
    *,
    bank: SurfaceBank,
    samples: dict[str, list[str]],
    array_key: Literal["height_raw", "height_filtered"],
    title: str,
    path: Path,
    plt: object,
    plot_style: PlotStyle,
) -> None:
    kinds = list(samples)
    max_cols = max((len(ids) for ids in samples.values()), default=1)
    fig, axes = plt.subplots(
        len(kinds),
        max_cols,
        figsize=plot_style.scaled_size(width_scale=max(1.0, 0.42 * max_cols), height_scale=max(1.0, 0.45 * len(kinds))),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=12)
    for row, kind in enumerate(kinds):
        ids = samples[kind]
        for col in range(max_cols):
            ax = axes[row][col]
            if col >= len(ids):
                ax.axis("off")
                continue
            surface_id = ids[col]
            arrays = bank.load_surface_arrays(surface_id)
            height = arrays[array_key]
            vmax = float(np.nanpercentile(np.abs(height), 98.0))
            vmax = max(vmax, 1e-6)
            ax.imshow(height, cmap="terrain", origin="lower", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(surface_id, fontsize=max(plot_style.font_size - 2, 6))
            if col == 0:
                ax.set_ylabel(kind, fontsize=max(plot_style.font_size - 1, 7))
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


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
