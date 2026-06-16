"""P1 single-case debug plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from Spine_Sim_V2.io.npz_io import load_npz_arrays
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.plotting.styles import PlotStyle, apply_plot_style, load_plot_style, require_columns


P1_DEBUG_FIGURE_NAMES = (
    "filtered_height_with_spines.png",
    "effective_contact_angle_map.png",
    "engagement_hotspot_map.png",
    "search_path_and_engagement_map.png",
    "spine_state_map.png",
    "load_displacement_curve.png",
)


def plot_p1_single_case_sanity(
    *,
    stage_dir: str | Path,
    style: str = "debug",
) -> dict[str, Path]:
    """Generate P1 sanity debug figures from saved data products."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style = load_plot_style(style)
    apply_plot_style(plt, plot_style)

    stage_path = Path(stage_dir)
    figures_dir = stage_path / "figures_debug"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = read_parquet(stage_path / "data" / "stage_summary.parquet")
    spines = read_parquet(stage_path / "data" / "stage_spines.parquet")
    require_columns(summary, ["case_id", "array_type"], dataset_name="P1 stage_summary.parquet")
    require_columns(
        spines,
        ["case_id", "x_mm", "y_mm", "state", "engaged"],
        dataset_name="P1 stage_spines.parquet",
    )
    cases = _load_case_plot_data(stage_path, summary, spines)

    outputs = {
        "filtered_height_with_spines": plot_style.path(figures_dir, "filtered_height_with_spines"),
        "effective_contact_angle_map": plot_style.path(figures_dir, "effective_contact_angle_map"),
        "engagement_hotspot_map": plot_style.path(figures_dir, "engagement_hotspot_map"),
        "search_path_and_engagement_map": plot_style.path(figures_dir, "search_path_and_engagement_map"),
        "spine_state_map": plot_style.path(figures_dir, "spine_state_map"),
        "load_displacement_curve": plot_style.path(figures_dir, "load_displacement_curve"),
    }
    _plot_map_panels(
        cases=cases,
        key="height_filtered",
        title="Filtered Height with Spines",
        path=outputs["filtered_height_with_spines"],
        plt=plt,
        plot_style=plot_style,
        overlay_spines=True,
        cmap="terrain",
        colorbar_label="height (mm)",
    )
    _plot_map_panels(
        cases=cases,
        key="effective_contact_angle",
        title="Effective Contact Angle",
        path=outputs["effective_contact_angle_map"],
        plt=plt,
        plot_style=plot_style,
        overlay_spines=False,
        cmap="coolwarm",
        colorbar_label="angle (deg)",
    )
    _plot_map_panels(
        cases=cases,
        key="engagement_candidate_mask",
        title="Engagement Hotspot Mask",
        path=outputs["engagement_hotspot_map"],
        plt=plt,
        plot_style=plot_style,
        overlay_spines=True,
        cmap=plot_style.colormap,
        colorbar_label="candidate",
    )
    _plot_search_paths(cases=cases, path=outputs["search_path_and_engagement_map"], plt=plt, plot_style=plot_style)
    _plot_spine_states(cases=cases, path=outputs["spine_state_map"], plt=plt, plot_style=plot_style)
    _plot_load_displacement(cases=cases, path=outputs["load_displacement_curve"], plt=plt, plot_style=plot_style)
    return outputs


def _load_case_plot_data(stage_path: Path, summary: Any, spines: Any) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        case_id = str(row["case_id"])
        case_dir = stage_path / "sample_cases" / case_id
        arrays = load_npz_arrays(case_dir / "case_arrays.npz")
        case_spines = spines.loc[spines["case_id"] == case_id].copy()
        cases.append(
            {
                "case_id": case_id,
                "array_type": str(row["array_type"]),
                "summary": row.to_dict(),
                "spines": case_spines,
                "arrays": arrays,
            }
        )
    if not cases:
        raise ValueError(f"No cases found in {stage_path}")
    return cases


def _plot_map_panels(
    *,
    cases: list[dict[str, Any]],
    key: str,
    title: str,
    path: Path,
    plt: Any,
    plot_style: PlotStyle,
    overlay_spines: bool,
    cmap: str,
    colorbar_label: str,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=plot_style.scaled_size(width_scale=max(1.0, 0.8 * len(cases))),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(title, fontsize=plot_style.font_size + 2)
    for ax, case_data in zip(axes[0], cases):
        arr = np.asarray(case_data["arrays"][key])
        image = ax.imshow(arr, origin="lower", cmap=cmap, extent=_image_extent(case_data))
        if overlay_spines:
            _scatter_spines(ax, case_data["spines"])
        ax.set_title(_case_title(case_data))
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, shrink=0.78)
        cbar.set_label(colorbar_label)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _plot_search_paths(*, cases: list[dict[str, Any]], path: Path, plt: Any, plot_style: PlotStyle) -> None:
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=plot_style.scaled_size(width_scale=max(1.0, 0.8 * len(cases))),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Search Paths and Engagement Points", fontsize=plot_style.font_size + 2)
    for ax, case_data in zip(axes[0], cases):
        arrays = case_data["arrays"]
        image = ax.imshow(
            arrays["height_filtered"],
            origin="lower",
            cmap="terrain",
            extent=_image_extent(case_data),
        )
        paths_x = np.asarray(arrays["search_path_x"])
        paths_y = np.asarray(arrays["search_path_y"])
        for x, y in zip(paths_x, paths_y):
            ax.plot(x, y, color="black", linewidth=max(0.5, plot_style.line_width * 0.45), alpha=0.45)
        engagement_x = np.asarray(arrays["engagement_x"], dtype=float)
        engagement_y = np.asarray(arrays["engagement_y"], dtype=float)
        valid = np.isfinite(engagement_x) & np.isfinite(engagement_y)
        ax.scatter(engagement_x[valid], engagement_y[valid], marker="*", s=plot_style.marker_size * 18, c="red", edgecolors="white")
        _scatter_spines(ax, case_data["spines"])
        ax.set_title(_case_title(case_data))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.78).set_label("height (mm)")
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _plot_spine_states(*, cases: list[dict[str, Any]], path: Path, plt: Any, plot_style: PlotStyle) -> None:
    state_colors = {
        "engaged": "#2ca02c",
        "search_failed": "#d62728",
        "surface_contact": "#ff7f0e",
        "side_contact": "#9467bd",
        "no_contact": "#7f7f7f",
        "failed_overload": "#8c564b",
    }
    fig, axes = plt.subplots(
        1,
        len(cases),
        figsize=plot_style.scaled_size(width_scale=max(1.0, 0.8 * len(cases))),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle("Spine State Map", fontsize=plot_style.font_size + 2)
    for ax, case_data in zip(axes[0], cases):
        arr = case_data["arrays"]["height_filtered"]
        ax.imshow(arr, origin="lower", cmap="Greys", alpha=0.45, extent=_image_extent(case_data))
        spines = case_data["spines"]
        for state, group in spines.groupby("state", sort=True):
            ax.scatter(
                group["x_mm"],
                group["y_mm"],
                s=plot_style.marker_size * 14,
                label=str(state),
                c=state_colors.get(str(state), "#1f77b4"),
                edgecolors="white",
                linewidths=0.8,
            )
        ax.set_title(_case_title(case_data))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=max(plot_style.font_size - 2, 6), loc="upper right")
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _plot_load_displacement(*, cases: list[dict[str, Any]], path: Path, plt: Any, plot_style: PlotStyle) -> None:
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    for case_data in cases:
        arrays = case_data["arrays"]
        s = np.asarray(arrays["load_displacement_s"], dtype=float)
        f = np.asarray(arrays["load_displacement_f"], dtype=float)
        ax.plot(s, f, marker="o", linewidth=plot_style.line_width, markersize=plot_style.marker_size, label=_case_title(case_data))
    ax.set_title("Load-Displacement Curve")
    ax.set_xlabel("displacement (mm)")
    ax.set_ylabel("tangential force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=max(plot_style.font_size - 2, 6))
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _scatter_spines(ax: Any, spines: Any) -> None:
    ax.scatter(
        spines["x_mm"],
        spines["y_mm"],
        s=48,
        c="white",
        edgecolors="black",
        linewidths=0.8,
    )


def _case_title(case_data: dict[str, Any]) -> str:
    return f"{case_data['array_type']} ({case_data['case_id']})"


def _image_extent(case_data: dict[str, Any]) -> tuple[float, float, float, float]:
    arrays = case_data["arrays"]
    arr = arrays["height_filtered"]
    dx_mm = float(np.asarray(arrays.get("dx_mm", 1.0)))
    dy_mm = float(np.asarray(arrays.get("dy_mm", 1.0)))
    ny, nx = np.asarray(arr).shape
    return (0.0, (nx - 1) * dx_mm, 0.0, (ny - 1) * dy_mm)
