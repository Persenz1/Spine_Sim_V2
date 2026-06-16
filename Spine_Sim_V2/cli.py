"""爪刺阵列仿真工程的命令行入口。"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from Spine_Sim_V2 import __version__


def build_parser(program: str | None = None) -> argparse.ArgumentParser:
    """构建控制台入口共享的顶层解析器。"""
    parser = argparse.ArgumentParser(
        prog=program,
        description=(
            "Spine_Sim_V2: quasi-static simulation tooling for spine arrays "
            "on hard rough surfaces."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command")

    simulate = subparsers.add_parser(
        "simulate",
        help="Run a simulation pipeline placeholder.",
        description="Phase 0 placeholder for future simulation pipelines.",
    )
    _add_simulate_arguments(simulate)
    simulate.set_defaults(handler=_handle_simulate)

    analyze = subparsers.add_parser(
        "analyze",
        help="Run analysis placeholder.",
        description="Phase 0 placeholder for future statistics and rankings.",
    )
    analyze.set_defaults(handler=_handle_analyze)

    plot = subparsers.add_parser(
        "plot",
        help="Run plotting placeholder.",
        description="Phase 0 placeholder for future figure generation.",
    )
    plot.set_defaults(handler=_handle_plot)

    return parser


def build_simulate_parser(program: str | None = None) -> argparse.ArgumentParser:
    """构建 ``scripts/simulate.py`` 使用的解析器。"""
    parser = argparse.ArgumentParser(
        prog=program,
        description="Run simulation and data-generation pipelines.",
    )
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    p0 = subparsers.add_parser(
        "p0-surface-bank",
        help="Generate a reusable proxy surface bank.",
        description="Generate Phase 2 proxy surface bank data products.",
    )
    p0.add_argument("--bank-id", required=True, help="Surface bank identifier.")
    p0.add_argument(
        "--surfaces",
        required=True,
        help="Comma-separated surface kinds, e.g. sandpaper,concrete.",
    )
    p0.add_argument("--n-per-kind", type=int, required=True, help="Number of surfaces per kind.")
    p0.add_argument(
        "--resolution",
        type=int,
        default=5,
        help="Grid resolution in cells per mm. Default: 5.",
    )
    p0.add_argument("--size-x-mm", type=float, default=60.0, help="Surface window size along x.")
    p0.add_argument("--size-y-mm", type=float, default=40.0, help="Surface window size along y.")
    p0.add_argument("--tip-radius-mm", type=float, default=0.05, help="Probe filter tip radius.")
    p0.add_argument("--outdir", required=True, help="Output surface bank directory.")
    p0.add_argument("--base-seed", type=int, default=20260616, help="Base seed for reproducibility.")
    p0.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing generated bank directory.",
    )
    p0.set_defaults(handler=_handle_p0_surface_bank)
    p1 = subparsers.add_parser(
        "p1-single-case",
        help="Run rigid and compliant P1 single-case sanity baselines.",
        description="Generate Phase 5 P1 single-case sanity data products.",
    )
    p1.add_argument(
        "--surface-bank",
        default="data/surface_bank_debug",
        help="Path to an existing surface bank. Default: data/surface_bank_debug.",
    )
    p1.add_argument(
        "--surface-id",
        default="concrete_000000",
        help="Surface identifier to load from the bank. Default: concrete_000000.",
    )
    p1.add_argument(
        "--outdir",
        default="outputs/P1_single_case_sanity",
        help="Output stage directory. Default: outputs/P1_single_case_sanity.",
    )
    p1.add_argument(
        "--compliant-spring-k-n-per-m",
        type=float,
        default=330.0,
        help="Compliant baseline spring stiffness in N/m. Default: 330.",
    )
    p1.set_defaults(handler=_handle_p1_single_case)
    p2 = subparsers.add_parser(
        "p2-compliant-k-alpha",
        help="Run P2 compliant single-spine k-alpha screening.",
    )
    p2.add_argument("--surface-bank", required=True, help="Path to surface bank.")
    p2.add_argument("--n-surfaces-per-kind", type=int, default=100)
    p2.add_argument("--outdir", default="outputs/P2_compliant_k_alpha_screen")
    p2.set_defaults(handler=_handle_p2_compliant_k_alpha)

    p3 = subparsers.add_parser(
        "p3-rigid-alpha",
        help="Run P3 rigid single-spine alpha screening.",
    )
    p3.add_argument("--surface-bank", required=True, help="Path to surface bank.")
    p3.add_argument("--n-surfaces-per-kind", type=int, default=100)
    p3.add_argument("--outdir", default="outputs/P3_rigid_alpha_screen")
    p3.set_defaults(handler=_handle_p3_rigid_alpha)

    p5a = subparsers.add_parser(
        "p5a-array-coarse",
        help="Run P5a array size and pitch coarse screening.",
    )
    p5a.add_argument("--surface-bank", required=True, help="Path to surface bank.")
    p5a.add_argument("--p2-selected", required=True, help="P2 selected_candidates.json path.")
    p5a.add_argument("--p3-selected", required=True, help="P3 selected_candidates.json path.")
    p5a.add_argument("--n-surfaces-per-kind", type=int, default=50)
    p5a.add_argument("--outdir", default="outputs/P5a_array_pitch_coarse_screen")
    p5a.set_defaults(handler=_handle_p5a_array_coarse)

    p5b = subparsers.add_parser(
        "p5b-array-refine",
        help="Run P5b array size and pitch refine screening.",
    )
    p5b.add_argument("--surface-bank", required=True, help="Path to surface bank.")
    p5b.add_argument("--p5a-selected", required=True, help="P5a selected_candidates.json path.")
    p5b.add_argument("--n-surfaces-per-kind", type=int, default=200)
    p5b.add_argument("--outdir", default="outputs/P5b_array_pitch_refine_screen")
    p5b.set_defaults(handler=_handle_p5b_array_refine)

    p6 = subparsers.add_parser(
        "p6-final-mc",
        help="Run P6 final 3D Monte Carlo over P5b selected candidates.",
    )
    p6.add_argument("--surface-bank", required=True, help="Path to surface bank.")
    p6.add_argument(
        "--selected-candidates",
        required=True,
        help="P5b selected_candidates.json path.",
    )
    p6.add_argument("--n-surfaces-per-kind", type=int, default=1000)
    p6.add_argument(
        "--surface-selection",
        default="first_n",
        choices=["first_n", "random_fixed", "explicit_list"],
        help="Surface selection policy. Default: first_n.",
    )
    p6.add_argument(
        "--surface-list",
        default=None,
        help="JSON file or comma-separated surface_id list for explicit_list selection.",
    )
    p6.add_argument("--outdir", default="outputs/P6_final_3d_monte_carlo")
    p6.add_argument("--workers", type=int, default=1, help="Parallel worker threads. Default: 1.")
    p6.add_argument("--random-seed", type=int, default=20260617, help="Seed for random_fixed.")
    p6.set_defaults(handler=_handle_p6_final_mc)
    return parser


def build_analyze_parser(program: str | None = None) -> argparse.ArgumentParser:
    """构建 ``scripts/analyze_results.py`` 使用的解析器。"""
    parser = argparse.ArgumentParser(
        prog=program,
        description="Analyze saved simulation data placeholders.",
    )
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    stage = subparsers.add_parser(
        "stage",
        help="Recompute grouped statistics and rankings for a saved stage.",
    )
    stage.add_argument("--stage-dir", required=True, help="Saved P2/P3/P5 stage directory.")
    stage.set_defaults(handler=_handle_analyze_stage)

    final = subparsers.add_parser(
        "final",
        help="Recompute P6 final grouped statistics, rankings, and reports.",
    )
    final.add_argument("--stage-dir", required=True, help="Saved P6 final Monte Carlo directory.")
    final.set_defaults(handler=_handle_analyze_final)

    p7 = subparsers.add_parser(
        "p7-surface",
        help="Generate P7 surface generalization products from P6 data.",
    )
    p7.add_argument("--p6-dir", required=True, help="Saved P6 final Monte Carlo directory.")
    p7.add_argument("--outdir", default="outputs/P7_surface_generalization")
    p7.set_defaults(handler=_handle_analyze_p7_surface_generalization)

    p8 = subparsers.add_parser(
        "p8-preload",
        help="Generate P8 preload efficiency products from P6 data.",
    )
    p8.add_argument("--p6-dir", required=True, help="Saved P6 final Monte Carlo directory.")
    p8.add_argument("--outdir", default="outputs/P8_preload_efficiency")
    p8.set_defaults(handler=_handle_analyze_p8_preload_efficiency)

    p7_old = subparsers.add_parser(
        "p7-surface-generalization",
        help="Alias for p7-surface.",
    )
    p7_old.add_argument("--p6-dir", required=True, help="Saved P6 final Monte Carlo directory.")
    p7_old.add_argument("--outdir", default="outputs/P7_surface_generalization")
    p7_old.set_defaults(handler=_handle_analyze_p7_surface_generalization)

    p8_old = subparsers.add_parser(
        "p8-preload-efficiency",
        help="Alias for p8-preload.",
    )
    p8_old.add_argument("--p6-dir", required=True, help="Saved P6 final Monte Carlo directory.")
    p8_old.add_argument("--outdir", default="outputs/P8_preload_efficiency")
    p8_old.set_defaults(handler=_handle_analyze_p8_preload_efficiency)
    return parser


def build_plot_parser(program: str | None = None) -> argparse.ArgumentParser:
    """构建 ``scripts/plot_results.py`` 使用的解析器。"""
    parser = argparse.ArgumentParser(
        prog=program,
        description="Generate figures from saved data products.",
    )
    _add_common_arguments(parser)
    subparsers = parser.add_subparsers(dest="command")
    surface_audit = subparsers.add_parser(
        "surface-audit",
        help="Generate Phase 2 surface bank audit figures.",
        description="Generate audit figures from an existing surface bank.",
    )
    surface_audit.add_argument("--surface-bank", required=True, help="Path to a surface bank directory.")
    surface_audit.add_argument(
        "--sample-per-kind",
        type=int,
        default=8,
        help="Number of surfaces to show per kind in gallery plots.",
    )
    surface_audit.add_argument("--outdir", required=True, help="Figure output directory.")
    surface_audit.add_argument("--seed", type=int, default=20260616, help="Audit sampling seed.")
    surface_audit.add_argument("--style", default="report", choices=["debug", "report", "paper"], help="Plot style.")
    surface_audit.set_defaults(handler=_handle_surface_audit)
    p1 = subparsers.add_parser(
        "p1",
        help="Generate Phase 5 P1 single-case sanity debug figures.",
        description="Generate P1 debug figures from a saved stage directory.",
    )
    p1.add_argument(
        "--stage-dir",
        default="outputs/P1_single_case_sanity",
        help="P1 stage directory. Default: outputs/P1_single_case_sanity.",
    )
    p1.add_argument(
        "--style",
        default="debug",
        choices=["debug", "report", "paper"],
        help="Plot style. Default: debug.",
    )
    p1.set_defaults(handler=_handle_plot_p1)
    stage = subparsers.add_parser(
        "stage",
        help="Generate P2/P3/P5 report figures from a saved stage.",
    )
    stage.add_argument("--stage-dir", required=True, help="Saved P2/P3/P5 stage directory.")
    stage.add_argument("--style", default="report", choices=["debug", "report", "paper"], help="Plot style.")
    stage.set_defaults(handler=_handle_plot_stage)

    final = subparsers.add_parser(
        "final",
        help="Generate P6 final report figures from saved data.",
    )
    final.add_argument("--stage-dir", required=True, help="Saved P6 final Monte Carlo directory.")
    final.add_argument("--style", default="report", choices=["debug", "report", "paper"], help="Plot style.")
    final.set_defaults(handler=_handle_plot_stage)

    p7 = subparsers.add_parser(
        "p7",
        help="Generate P7 surface generalization figures.",
    )
    p7.add_argument("--stage-dir", default="outputs/P7_surface_generalization")
    p7.add_argument("--style", default="report", choices=["debug", "report", "paper"], help="Plot style.")
    p7.set_defaults(handler=_handle_plot_stage)

    p8 = subparsers.add_parser(
        "p8",
        help="Generate P8 preload efficiency figures.",
    )
    p8.add_argument("--stage-dir", default="outputs/P8_preload_efficiency")
    p8.add_argument("--style", default="report", choices=["debug", "report", "paper"], help="Plot style.")
    p8.set_defaults(handler=_handle_plot_stage)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )


def _add_simulate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage",
        default="p0",
        choices=["p0", "p1", "p2", "p3", "p5", "p6"],
        help="Pipeline stage to run when implemented.",
    )


def _handle_simulate(args: argparse.Namespace) -> int:
    print(f"Simulation stage {args.stage} is not implemented in Phase 0.")
    return 0


def _handle_p0_surface_bank(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p0_surface_bank import run

    bank = run(
        bank_id=args.bank_id,
        surfaces=args.surfaces,
        n_per_kind=args.n_per_kind,
        resolution=args.resolution,
        size_x_mm=args.size_x_mm,
        size_y_mm=args.size_y_mm,
        tip_radius_mm=args.tip_radius_mm,
        outdir=args.outdir,
        base_seed=args.base_seed,
        overwrite=args.overwrite,
    )
    print(f"Generated surface bank {bank.bank_id} at {bank.root}")
    return 0


def _handle_p1_single_case(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p1_single_case import run_single_case_sanity

    stage_dir = run_single_case_sanity(
        surface_bank=args.surface_bank,
        surface_id=args.surface_id,
        outdir=args.outdir,
        compliant_spring_k_n_per_m=args.compliant_spring_k_n_per_m,
    )
    print(f"Generated P1 single-case sanity data at {stage_dir}")
    return 0


def _handle_p2_compliant_k_alpha(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p2_compliant_k_alpha import run

    stage_dir = run(
        surface_bank=args.surface_bank,
        n_surfaces_per_kind=args.n_surfaces_per_kind,
        outdir=args.outdir,
    )
    print(f"Generated P2 compliant k-alpha screen at {stage_dir}")
    return 0


def _handle_p3_rigid_alpha(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p3_rigid_alpha import run

    stage_dir = run(
        surface_bank=args.surface_bank,
        n_surfaces_per_kind=args.n_surfaces_per_kind,
        outdir=args.outdir,
    )
    print(f"Generated P3 rigid alpha screen at {stage_dir}")
    return 0


def _handle_p5a_array_coarse(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p5_array_screen import run_p5a

    stage_dir = run_p5a(
        surface_bank=args.surface_bank,
        p2_selected=args.p2_selected,
        p3_selected=args.p3_selected,
        n_surfaces_per_kind=args.n_surfaces_per_kind,
        outdir=args.outdir,
    )
    print(f"Generated P5a array coarse screen at {stage_dir}")
    return 0


def _handle_p5b_array_refine(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p5_array_screen import run_p5b

    stage_dir = run_p5b(
        surface_bank=args.surface_bank,
        p5a_selected=args.p5a_selected,
        n_surfaces_per_kind=args.n_surfaces_per_kind,
        outdir=args.outdir,
    )
    print(f"Generated P5b array refine screen at {stage_dir}")
    return 0


def _handle_p6_final_mc(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.pipelines.p6_final_mc import run

    stage_dir = run(
        surface_bank=args.surface_bank,
        selected_candidates=args.selected_candidates,
        n_surfaces_per_kind=args.n_surfaces_per_kind,
        surface_selection=args.surface_selection,
        surface_list=args.surface_list,
        outdir=args.outdir,
        workers=args.workers,
        random_seed=args.random_seed,
    )
    print(f"Generated P6 final 3D Monte Carlo at {stage_dir}")
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    print("Analysis is not implemented in Phase 0.")
    return 0


def _handle_analyze_stage(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.analysis.ranking import analyze_stage

    grouped, rankings = analyze_stage(args.stage_dir)
    print(f"Wrote grouped statistics ({len(grouped)} rows) and rankings ({len(rankings)} rows).")
    return 0


def _handle_analyze_final(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.analysis.final_mc import p6_schema_collection, write_final_analysis
    from Spine_Sim_V2.io.schema_io import write_schema
    from Spine_Sim_V2.pipelines.p6_final_mc import _write_final_report

    grouped, rankings, convergence = write_final_analysis(args.stage_dir)
    write_schema(args.stage_dir, p6_schema_collection(grouped, rankings, convergence))
    _write_final_report(Path(args.stage_dir))
    print(
        "Wrote final grouped statistics "
        f"({len(grouped)} rows), rankings ({len(rankings)} rows), "
        f"and convergence ({len(convergence)} rows)."
    )
    return 0


def _handle_analyze_p7_surface_generalization(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.analysis.final_mc import run_p7_surface_generalization

    stage_dir = run_p7_surface_generalization(p6_dir=args.p6_dir, outdir=args.outdir)
    print(f"Generated P7 surface generalization products at {stage_dir}")
    return 0


def _handle_analyze_p8_preload_efficiency(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.analysis.final_mc import run_p8_preload_efficiency

    stage_dir = run_p8_preload_efficiency(p6_dir=args.p6_dir, outdir=args.outdir)
    print(f"Generated P8 preload efficiency products at {stage_dir}")
    return 0


def _handle_plot(args: argparse.Namespace) -> int:
    print("Plotting is not implemented in Phase 0.")
    return 0


def _handle_surface_audit(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.plotting.plot_surface import plot_surface_audit

    outputs = plot_surface_audit(
        surface_bank=args.surface_bank,
        sample_per_kind=args.sample_per_kind,
        outdir=args.outdir,
        seed=args.seed,
        style=args.style,
    )
    for path in outputs.values():
        print(path)
    return 0


def _handle_plot_p1(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.plotting.plot_case import plot_p1_single_case_sanity

    outputs = plot_p1_single_case_sanity(
        stage_dir=args.stage_dir,
        style=args.style,
    )
    for path in outputs.values():
        print(path)
    return 0


def _handle_plot_stage(args: argparse.Namespace) -> int:
    from Spine_Sim_V2.plotting.plot_stage import plot_stage

    outputs = plot_stage(args.stage_dir, style=args.style)
    for path in outputs.values():
        print(path)
    return 0


def main(argv: Sequence[str] | None = None, program: str | None = None) -> int:
    """运行顶层命令行接口。"""
    parser = build_parser(program=program)
    return _run_parser(parser, argv)


def simulate_main(
    argv: Sequence[str] | None = None,
    program: str | None = None,
) -> int:
    """运行仿真脚本入口。"""
    return _run_parser(build_simulate_parser(program=program), argv)


def analyze_main(
    argv: Sequence[str] | None = None,
    program: str | None = None,
) -> int:
    """运行分析脚本入口。"""
    return _run_parser(build_analyze_parser(program=program), argv)


def plot_main(
    argv: Sequence[str] | None = None,
    program: str | None = None,
) -> int:
    """运行绘图脚本入口。"""
    return _run_parser(build_plot_parser(program=program), argv)


def _run_parser(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> int:
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))
