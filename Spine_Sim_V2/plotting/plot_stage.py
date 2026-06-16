"""Report plots for P2/P3 screening stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Spine_Sim_V2.analysis.ranking import analyze_stage, infer_stage_kind
from Spine_Sim_V2.io.parquet_io import read_parquet
from Spine_Sim_V2.plotting.styles import PlotStyle, apply_plot_style, load_plot_style, require_columns


def plot_stage(stage_dir: str | Path, *, style: str = "report") -> dict[str, Path]:
    """Generate default report figures for a saved stage."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style = load_plot_style(style)
    apply_plot_style(plt, plot_style)

    stage_path = Path(stage_dir)
    data_dir = stage_path / "data"
    outdir = stage_path / "figures_report"
    outdir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "final_summary.parquet").exists():
        from Spine_Sim_V2.analysis.final_mc import write_final_analysis

        if not (data_dir / "final_grouped_statistics.parquet").exists() or not (
            data_dir / "final_rankings.parquet"
        ).exists():
            write_final_analysis(stage_path)
        summary = read_parquet(data_dir / "final_summary.parquet")
        grouped = read_parquet(data_dir / "final_grouped_statistics.parquet")
        rankings = read_parquet(data_dir / "final_rankings.parquet")
        convergence = read_parquet(data_dir / "convergence_statistics.parquet")
        return _plot_p6(
            summary,
            grouped=grouped,
            rankings=rankings,
            convergence=convergence,
            outdir=outdir,
            plt=plt,
            plot_style=plot_style,
        )
    if (data_dir / "surface_generalization_statistics.parquet").exists():
        stats = read_parquet(data_dir / "surface_generalization_statistics.parquet")
        rankings = read_parquet(data_dir / "surface_generalization_rankings.parquet")
        return _plot_p7(stats, rankings=rankings, outdir=outdir, plt=plt, plot_style=plot_style)
    if (data_dir / "preload_efficiency_statistics.parquet").exists():
        stats = read_parquet(data_dir / "preload_efficiency_statistics.parquet")
        return _plot_p8(stats, outdir=outdir, plt=plt, plot_style=plot_style)
    p9_path = data_dir / "p9_sensitivity_statistics.parquet"
    if not p9_path.exists():
        p9_path = data_dir / "sensitivity_statistics.parquet"
    if p9_path.exists():
        stats = read_parquet(p9_path)
        return _plot_p9(stats, outdir=outdir, plt=plt, plot_style=plot_style)
    if not (data_dir / "stage_grouped_statistics.parquet").exists() or not (
        data_dir / "stage_rankings.parquet"
    ).exists():
        analyze_stage(stage_path)
    stage_kind = infer_stage_kind(stage_path)
    grouped = read_parquet(data_dir / "stage_grouped_statistics.parquet")
    summary = read_parquet(data_dir / "stage_summary.parquet")
    if stage_kind.startswith("p5"):
        rankings = read_parquet(data_dir / "stage_rankings.parquet")
        selected_path = data_dir / "selected_candidates.parquet"
        selected = read_parquet(selected_path) if selected_path.exists() else None
        return _plot_p5(grouped, summary=summary, rankings=rankings, selected=selected, outdir=outdir, plt=plt, plot_style=plot_style)
    if stage_kind.startswith("p2"):
        return _plot_p2(grouped, outdir=outdir, plt=plt, plot_style=plot_style)
    return _plot_p3(grouped, summary=summary, outdir=outdir, plt=plt, plot_style=plot_style)


def _plot_p2(grouped: Any, *, outdir: Path, plt: Any, plot_style: PlotStyle) -> dict[str, Path]:
    require_columns(
        grouped,
        ["spring_k_n_per_m", "alpha_p_deg", "success_probability", "f_t_lim_n_mean", "r_sat_n_mean", "f_t_lim_over_w_total_mean"],
        dataset_name="stage_grouped_statistics.parquet",
    )
    outputs = {
        "k_alpha_success_heatmap": plot_style.path(outdir, "k_alpha_success_heatmap"),
        "k_alpha_f_t_lim_heatmap": plot_style.path(outdir, "k_alpha_f_t_lim_heatmap"),
        "k_alpha_saturation_heatmap": plot_style.path(outdir, "k_alpha_saturation_heatmap"),
        "k_alpha_efficiency_heatmap": plot_style.path(outdir, "k_alpha_efficiency_heatmap"),
    }
    collapsed = (
        grouped.groupby(["spring_k_n_per_m", "alpha_p_deg"], dropna=False)
        .agg(
            success_probability=("success_probability", "mean"),
            f_t_lim_n_mean=("f_t_lim_n_mean", "mean"),
            r_sat_n_mean=("r_sat_n_mean", "mean"),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total_mean", "mean"),
        )
        .reset_index()
    )
    _heatmap(
        collapsed,
        row="spring_k_n_per_m",
        col="alpha_p_deg",
        value="success_probability",
        title="P2 Success Probability",
        path=outputs["k_alpha_success_heatmap"],
        plt=plt,
        plot_style=plot_style,
    )
    _heatmap(
        collapsed,
        row="spring_k_n_per_m",
        col="alpha_p_deg",
        value="f_t_lim_n_mean",
        title="P2 Mean Tangential Limit (N)",
        path=outputs["k_alpha_f_t_lim_heatmap"],
        plt=plt,
        plot_style=plot_style,
    )
    _heatmap(
        collapsed,
        row="spring_k_n_per_m",
        col="alpha_p_deg",
        value="r_sat_n_mean",
        title="P2 Saturation Ratio",
        path=outputs["k_alpha_saturation_heatmap"],
        plt=plt,
        plot_style=plot_style,
    )
    _heatmap(
        collapsed,
        row="spring_k_n_per_m",
        col="alpha_p_deg",
        value="f_t_lim_over_w_total_mean",
        title="P2 Efficiency F_lim / W",
        path=outputs["k_alpha_efficiency_heatmap"],
        plt=plt,
        plot_style=plot_style,
    )
    return outputs


def _plot_p3(grouped: Any, *, summary: Any, outdir: Path, plt: Any, plot_style: PlotStyle) -> dict[str, Path]:
    require_columns(
        grouped,
        ["alpha_p_deg", "success_probability", "f_t_lim_n_mean", "f_t_lim_over_w_total_mean"],
        dataset_name="stage_grouped_statistics.parquet",
    )
    outputs = {
        "rigid_alpha_success_curve": plot_style.path(outdir, "rigid_alpha_success_curve"),
        "rigid_alpha_f_t_lim_curve": plot_style.path(outdir, "rigid_alpha_f_t_lim_curve"),
        "rigid_alpha_efficiency_curve": plot_style.path(outdir, "rigid_alpha_efficiency_curve"),
        "rigid_alpha_by_surface_boxplot": plot_style.path(outdir, "rigid_alpha_by_surface_boxplot"),
    }
    collapsed = (
        grouped.groupby("alpha_p_deg", dropna=False)
        .agg(
            success_probability=("success_probability", "mean"),
            f_t_lim_n_mean=("f_t_lim_n_mean", "mean"),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total_mean", "mean"),
        )
        .reset_index()
        .sort_values("alpha_p_deg")
    )
    _curve(collapsed, x="alpha_p_deg", y="success_probability", title="P3 Rigid Success", ylabel="success", path=outputs["rigid_alpha_success_curve"], plt=plt, plot_style=plot_style)
    _curve(collapsed, x="alpha_p_deg", y="f_t_lim_n_mean", title="P3 Rigid Mean Tangential Limit", ylabel="F_lim (N)", path=outputs["rigid_alpha_f_t_lim_curve"], plt=plt, plot_style=plot_style)
    _curve(collapsed, x="alpha_p_deg", y="f_t_lim_over_w_total_mean", title="P3 Rigid Efficiency", ylabel="F_lim / W", path=outputs["rigid_alpha_efficiency_curve"], plt=plt, plot_style=plot_style)
    _surface_boxplot(summary, path=outputs["rigid_alpha_by_surface_boxplot"], plt=plt, plot_style=plot_style)
    return outputs


def _plot_p5(grouped: Any, *, summary: Any, rankings: Any, selected: Any, outdir: Path, plt: Any, plot_style: PlotStyle) -> dict[str, Path]:
    require_columns(
        grouped,
        ["rows", "cols", "f_t_lim_n_mean", "success_probability", "n_eff_kish_mean", "eta_max_mean"],
        dataset_name="stage_grouped_statistics.parquet",
    )
    outputs = {
        "array_force_heatmap": plot_style.path(outdir, "array_force_heatmap"),
        "array_success_heatmap": plot_style.path(outdir, "array_success_heatmap"),
        "array_n_eff_kish_heatmap": plot_style.path(outdir, "array_n_eff_kish_heatmap"),
        "array_eta_max_heatmap": plot_style.path(outdir, "array_eta_max_heatmap"),
        "rigid_vs_compliant_ranking": plot_style.path(outdir, "rigid_vs_compliant_ranking"),
        "selected_candidates_overview": plot_style.path(outdir, "selected_candidates_overview"),
    }
    collapsed = (
        grouped.groupby(["rows", "cols"], dropna=False)
        .agg(
            f_t_lim_n_mean=("f_t_lim_n_mean", "mean"),
            success_probability=("success_probability", "mean"),
            n_eff_kish_mean=("n_eff_kish_mean", "mean"),
            eta_max_mean=("eta_max_mean", "mean"),
        )
        .reset_index()
    )
    _heatmap(collapsed, row="rows", col="cols", value="f_t_lim_n_mean", title="P5 Array Force", path=outputs["array_force_heatmap"], plt=plt, plot_style=plot_style)
    _heatmap(collapsed, row="rows", col="cols", value="success_probability", title="P5 Array Success", path=outputs["array_success_heatmap"], plt=plt, plot_style=plot_style)
    _heatmap(collapsed, row="rows", col="cols", value="n_eff_kish_mean", title="P5 Kish Effective Count", path=outputs["array_n_eff_kish_heatmap"], plt=plt, plot_style=plot_style)
    _heatmap(collapsed, row="rows", col="cols", value="eta_max_mean", title="P5 Eta Max", path=outputs["array_eta_max_heatmap"], plt=plt, plot_style=plot_style)
    _plot_p5_ranking(rankings, path=outputs["rigid_vs_compliant_ranking"], plt=plt, plot_style=plot_style)
    _plot_selected_overview(selected if selected is not None else rankings.loc[rankings["selected"] == True], path=outputs["selected_candidates_overview"], plt=plt, plot_style=plot_style)
    return outputs


def _plot_p6(
    summary: Any,
    *,
    grouped: Any,
    rankings: Any,
    convergence: Any,
    outdir: Path,
    plt: Any,
    plot_style: PlotStyle,
) -> dict[str, Path]:
    require_columns(summary, ["candidate_id", "array_type", "surface_kind", "f_t_lim_n", "load_success", "eta_max"], dataset_name="final_summary.parquet")
    require_columns(grouped, ["candidate_id", "w_total_n", "f_t_lim_over_w_total_mean"], dataset_name="final_grouped_statistics.parquet")
    require_columns(rankings, ["rank", "candidate_id", "array_type", "success_probability"], dataset_name="final_rankings.parquet")
    require_columns(convergence, ["candidate_id", "n_surfaces_used", "f_t_lim_n_mean"], dataset_name="convergence_statistics.parquet")
    outputs = {
        "final_force_boxplot_by_candidate": plot_style.path(outdir, "final_force_boxplot_by_candidate"),
        "final_success_probability_by_candidate": plot_style.path(outdir, "final_success_probability_by_candidate"),
        "final_surface_generalization_boxplot": plot_style.path(outdir, "final_surface_generalization_boxplot"),
        "final_preload_efficiency_curve": plot_style.path(outdir, "final_preload_efficiency_curve"),
        "final_rigid_vs_compliant_summary": plot_style.path(outdir, "final_rigid_vs_compliant_summary"),
        "convergence_curve": plot_style.path(outdir, "convergence_curve"),
    }
    _candidate_boxplot(
        summary,
        value="f_t_lim_n",
        title="P6 Final Force by Candidate",
        ylabel="F_lim (N)",
        path=outputs["final_force_boxplot_by_candidate"],
        plt=plt,
        plot_style=plot_style,
    )
    _candidate_success_bar(rankings, path=outputs["final_success_probability_by_candidate"], plt=plt, plot_style=plot_style)
    _surface_kind_boxplot(
        summary,
        value="f_t_lim_n",
        title="P6 Surface Generalization",
        ylabel="F_lim (N)",
        path=outputs["final_surface_generalization_boxplot"],
        plt=plt,
        plot_style=plot_style,
    )
    preload = (
        grouped.groupby(["candidate_id", "w_total_n"], dropna=False)["f_t_lim_over_w_total_mean"]
        .mean()
        .reset_index()
    )
    _multi_curve(
        preload,
        x="w_total_n",
        y="f_t_lim_over_w_total_mean",
        group="candidate_id",
        title="P6 Preload Efficiency",
        ylabel="F_lim / W",
        path=outputs["final_preload_efficiency_curve"],
        plt=plt,
        plot_style=plot_style,
    )
    _array_type_summary(summary, path=outputs["final_rigid_vs_compliant_summary"], plt=plt, plot_style=plot_style)
    _multi_curve(
        convergence,
        x="n_surfaces_used",
        y="f_t_lim_n_mean",
        group="candidate_id",
        title="P6 Convergence",
        ylabel="Mean F_lim (N)",
        path=outputs["convergence_curve"],
        plt=plt,
        plot_style=plot_style,
    )
    return outputs


def _plot_p7(stats: Any, *, rankings: Any, outdir: Path, plt: Any, plot_style: PlotStyle) -> dict[str, Path]:
    require_columns(stats, ["candidate_id", "surface_kind", "success_probability", "f_t_lim_n_mean"], dataset_name="surface_generalization_statistics.parquet")
    require_columns(rankings, ["candidate_id", "surface_kind", "rank_shift"], dataset_name="surface_generalization_rankings.parquet")
    outputs = {
        "surface_generalization_boxplot": plot_style.path(outdir, "surface_generalization_boxplot"),
        "surface_rank_shift_heatmap": plot_style.path(outdir, "surface_rank_shift_heatmap"),
        "candidate_by_surface_success_heatmap": plot_style.path(outdir, "candidate_by_surface_success_heatmap"),
    }
    _surface_kind_boxplot(
        stats,
        value="f_t_lim_n_mean",
        title="P7 Surface Generalization",
        ylabel="Mean F_lim (N)",
        path=outputs["surface_generalization_boxplot"],
        plt=plt,
        plot_style=plot_style,
    )
    _heatmap(
        rankings,
        row="candidate_id",
        col="surface_kind",
        value="rank_shift",
        title="P7 Rank Shift by Surface",
        path=outputs["surface_rank_shift_heatmap"],
        plt=plt,
        plot_style=plot_style,
    )
    _heatmap(
        stats,
        row="candidate_id",
        col="surface_kind",
        value="success_probability",
        title="P7 Candidate Success by Surface",
        path=outputs["candidate_by_surface_success_heatmap"],
        plt=plt,
        plot_style=plot_style,
    )
    return outputs


def _plot_p8(stats: Any, *, outdir: Path, plt: Any, plot_style: PlotStyle) -> dict[str, Path]:
    require_columns(stats, ["candidate_id", "w_total_n", "f_t_lim_n_mean", "f_t_lim_over_w_total_mean", "success_probability"], dataset_name="preload_efficiency_statistics.parquet")
    outputs = {
        "preload_force_curve": plot_style.path(outdir, "preload_force_curve"),
        "preload_efficiency_curve": plot_style.path(outdir, "preload_efficiency_curve"),
        "preload_success_curve": plot_style.path(outdir, "preload_success_curve"),
    }
    _multi_curve(
        stats,
        x="w_total_n",
        y="f_t_lim_n_mean",
        group="candidate_id",
        title="P8 Preload Force",
        ylabel="Mean F_lim (N)",
        path=outputs["preload_force_curve"],
        plt=plt,
        plot_style=plot_style,
    )
    _multi_curve(
        stats,
        x="w_total_n",
        y="f_t_lim_over_w_total_mean",
        group="candidate_id",
        title="P8 Preload Efficiency",
        ylabel="F_lim / W",
        path=outputs["preload_efficiency_curve"],
        plt=plt,
        plot_style=plot_style,
    )
    _multi_curve(
        stats,
        x="w_total_n",
        y="success_probability",
        group="candidate_id",
        title="P8 Preload Success",
        ylabel="Success probability",
        path=outputs["preload_success_curve"],
        plt=plt,
        plot_style=plot_style,
    )
    return outputs


def _plot_p9(stats: Any, *, outdir: Path, plt: Any, plot_style: PlotStyle) -> dict[str, Path]:
    """Generate P9 sensitivity plots from a saved sensitivity statistics table."""
    require_columns(
        stats,
        ["sensitivity_name", "parameter_value", "metric_value"],
        dataset_name="p9_sensitivity_statistics.parquet",
    )
    outputs = {
        "fs_sensitivity_curve": plot_style.path(outdir, "fs_sensitivity_curve"),
        "F_star_sensitivity_curve": plot_style.path(outdir, "F_star_sensitivity_curve"),
        "tip_radius_sensitivity_curve": plot_style.path(outdir, "tip_radius_sensitivity_curve"),
        "search_travel_sensitivity_curve": plot_style.path(outdir, "search_travel_sensitivity_curve"),
        "chi_tn_sensitivity_curve": plot_style.path(outdir, "chi_tn_sensitivity_curve"),
    }
    mapping = {
        "fs": outputs["fs_sensitivity_curve"],
        "F_star": outputs["F_star_sensitivity_curve"],
        "tip_radius": outputs["tip_radius_sensitivity_curve"],
        "search_travel": outputs["search_travel_sensitivity_curve"],
        "chi_tn": outputs["chi_tn_sensitivity_curve"],
    }
    for name, path in mapping.items():
        subset = stats.loc[stats["sensitivity_name"] == name]
        if subset.empty:
            subset = stats.iloc[0:0]
        _p9_curve(
            subset,
            title=f"P9 {name} sensitivity",
            path=path,
            plt=plt,
            plot_style=plot_style,
        )
    return outputs


def _heatmap(data: Any, *, row: str, col: str, value: str, title: str, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(data, [row, col, value], dataset_name=f"plot data for {path.name}")
    pivot = data.pivot(index=row, columns=col, values=value).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=plot_style.colormap)
    ax.set_title(title)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([_fmt(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_fmt(v) for v in pivot.index])
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _p9_curve(data: Any, *, title: str, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    if len(data):
        group_column = "candidate_id" if "candidate_id" in data.columns else None
        if group_column:
            for group_value, subset in data.groupby(group_column, dropna=False):
                ordered = subset.sort_values("parameter_value")
                ax.plot(
                    ordered["parameter_value"],
                    ordered["metric_value"],
                    marker="o",
                    linewidth=plot_style.line_width,
                    markersize=plot_style.marker_size,
                    label=str(group_value),
                )
            if data[group_column].nunique(dropna=False) <= 12:
                ax.legend(fontsize=max(plot_style.font_size - 2, 6), ncols=2)
        else:
            ordered = data.sort_values("parameter_value")
            ax.plot(
                ordered["parameter_value"],
                ordered["metric_value"],
                marker="o",
                linewidth=plot_style.line_width,
                markersize=plot_style.marker_size,
            )
    ax.set_title(title)
    ax.set_xlabel("parameter_value")
    ax.set_ylabel("metric_value")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _curve(data: Any, *, x: str, y: str, title: str, ylabel: str, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(data, [x, y], dataset_name=f"plot data for {path.name}")
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    ax.plot(data[x], data[y], marker="o", linewidth=plot_style.line_width, markersize=plot_style.marker_size)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _surface_boxplot(summary: Any, *, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(summary, ["surface_kind", "f_t_lim_n"], dataset_name="stage_summary.parquet")
    labels = sorted(str(kind) for kind in summary["surface_kind"].unique())
    values = [
        summary.loc[summary["surface_kind"] == kind, "f_t_lim_n"].dropna().to_numpy(dtype=float)
        for kind in labels
    ]
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    try:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_title("P3 Rigid F_lim by Surface Kind")
    ax.set_ylabel("F_lim (N)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _candidate_boxplot(data: Any, *, value: str, title: str, ylabel: str, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(data, ["candidate_id", value], dataset_name=f"plot data for {path.name}")
    labels = [str(item) for item in sorted(data["candidate_id"].unique())]
    values = [
        data.loc[data["candidate_id"] == candidate_id, value].dropna().to_numpy(dtype=float)
        for candidate_id in labels
    ]
    fig, ax = plt.subplots(figsize=plot_style.scaled_size(width_scale=max(1.0, 0.065 * len(labels))), constrained_layout=True)
    try:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=60)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _surface_kind_boxplot(data: Any, *, value: str, title: str, ylabel: str, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(data, ["surface_kind", value], dataset_name=f"plot data for {path.name}")
    labels = [str(item) for item in sorted(data["surface_kind"].unique())]
    values = [
        data.loc[data["surface_kind"] == surface_kind, value].dropna().to_numpy(dtype=float)
        for surface_kind in labels
    ]
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    try:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _candidate_success_bar(rankings: Any, *, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    ordered = rankings.sort_values("rank")
    colors = ["#4c78a8" if item == "rigid" else "#f58518" for item in ordered["array_type"]]
    fig, ax = plt.subplots(figsize=plot_style.scaled_size(width_scale=max(1.0, 0.07 * len(ordered))), constrained_layout=True)
    ax.bar(range(len(ordered)), ordered["success_probability"], color=colors)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([str(item) for item in ordered["candidate_id"]], rotation=60, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Success probability")
    ax.set_title("P6 Success Probability by Candidate")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _multi_curve(
    data: Any,
    *,
    x: str,
    y: str,
    group: str,
    title: str,
    ylabel: str,
    path: Path,
    plt: Any,
    plot_style: PlotStyle,
) -> None:
    require_columns(data, [x, y, group], dataset_name=f"plot data for {path.name}")
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    for group_value, subset in data.groupby(group, dropna=False):
        ordered = subset.sort_values(x)
        ax.plot(ordered[x], ordered[y], marker="o", linewidth=plot_style.line_width, markersize=plot_style.marker_size, label=str(group_value))
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if data[group].nunique(dropna=False) <= 12:
        ax.legend(fontsize=max(plot_style.font_size - 2, 6), ncols=2)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _array_type_summary(summary: Any, *, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(summary, ["array_type", "load_success", "f_t_lim_n", "f_t_lim_over_w_total"], dataset_name="final_summary.parquet")
    collapsed = (
        summary.groupby("array_type", dropna=False)
        .agg(
            success_probability=("load_success", "mean"),
            f_t_lim_n_mean=("f_t_lim_n", "mean"),
            f_t_lim_over_w_total_mean=("f_t_lim_over_w_total", "mean"),
        )
        .reset_index()
    )
    x = range(len(collapsed))
    fig, ax1 = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    ax1.bar(x, collapsed["f_t_lim_n_mean"], color="#4c78a8", alpha=0.75, label="mean F_lim")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(item) for item in collapsed["array_type"]])
    ax1.set_ylabel("Mean F_lim (N)")
    ax2 = ax1.twinx()
    ax2.plot(list(x), collapsed["success_probability"], color="#f58518", marker="o", label="success")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Success probability")
    ax1.set_title("P6 Rigid vs Compliant Summary")
    ax1.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _plot_p5_ranking(rankings: Any, *, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    require_columns(rankings, ["rank", "array_type", "candidate_id", "score_total"], dataset_name="stage_rankings.parquet")
    top = rankings.sort_values("rank").groupby("array_type", dropna=False).head(10)
    labels = [str(item) for item in top["candidate_id"]]
    colors = ["#4c78a8" if item == "rigid" else "#f58518" for item in top["array_type"]]
    fig, ax = plt.subplots(figsize=plot_style.scaled_size(width_scale=max(1.0, 0.055 * len(top))), constrained_layout=True)
    ax.bar(range(len(top)), top["score_total"], color=colors)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel("P5 score")
    ax.set_title("Rigid vs Compliant Ranking")
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _plot_selected_overview(selected: Any, *, path: Path, plt: Any, plot_style: PlotStyle) -> None:
    if selected is None or len(selected) == 0:
        fig, ax = plt.subplots(figsize=plot_style.scaled_size(height_scale=0.7), constrained_layout=True)
        ax.set_title("Selected Candidates Overview")
        ax.text(0.5, 0.5, "No selected candidates", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=plot_style.dpi)
        plt.close(fig)
        return
    require_columns(selected, ["array_type", "rows", "cols", "pitch_t_mm", "pitch_l_mm", "candidate_id"], dataset_name="selected_candidates.parquet")
    colors = ["#4c78a8" if item == "rigid" else "#f58518" for item in selected["array_type"]]
    sizes = selected["rows"].astype(float) * selected["cols"].astype(float) * 16.0
    fig, ax = plt.subplots(figsize=plot_style.figure_size, constrained_layout=True)
    ax.scatter(selected["pitch_t_mm"], selected["pitch_l_mm"], s=sizes, c=colors, alpha=0.75, edgecolors="black")
    for _, row in selected.iterrows():
        ax.text(row["pitch_t_mm"], row["pitch_l_mm"], str(row["candidate_id"]), fontsize=7, ha="center", va="center")
    ax.set_xlabel("pitch_t_mm")
    ax.set_ylabel("pitch_l_mm")
    ax.set_title("Selected Candidates Overview")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=plot_style.dpi)
    plt.close(fig)


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"
