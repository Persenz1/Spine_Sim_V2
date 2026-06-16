"""P2: compliant single-spine stiffness-angle screening."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Spine_Sim_V2.analysis.ranking import analyze_stage
from Spine_Sim_V2.core.types import SingleCaseInput
from Spine_Sim_V2.io.manifest import create_manifest, write_manifest
from Spine_Sim_V2.io.parquet_io import write_parquet, write_preview_csv
from Spine_Sim_V2.io.schema_io import write_schema
from Spine_Sim_V2.pipelines.p1_single_case import run_single_case
from Spine_Sim_V2.surfaces.bank import SurfaceBank


P2_PROJECT_NAME = "P2_compliant_k_alpha_screen"
P2_SPRING_K_N_PER_M = (100.0, 150.0, 220.0, 330.0, 470.0, 680.0, 1000.0)
P2_ALPHA_P_DEG = (50.0, 60.0, 70.0, 80.0)
P2_W_TOTAL_N = (0.5, 1.0, 1.5, 2.0, 2.5)
P2_SURFACE_KINDS = ("sandpaper", "concrete", "brick", "painted_wall")


def run(
    *,
    surface_bank: str | Path,
    n_surfaces_per_kind: int = 100,
    outdir: str | Path = "outputs/P2_compliant_k_alpha_screen",
    spring_values: tuple[float, ...] = P2_SPRING_K_N_PER_M,
    alpha_values: tuple[float, ...] = P2_ALPHA_P_DEG,
    w_values: tuple[float, ...] = P2_W_TOTAL_N,
    surface_kinds: tuple[str, ...] = P2_SURFACE_KINDS,
) -> Path:
    """Run P2 compliant k-alpha screening."""
    return _run_screen(
        project_name=P2_PROJECT_NAME,
        stage_name="p2_compliant_k_alpha",
        surface_bank=surface_bank,
        n_surfaces_per_kind=n_surfaces_per_kind,
        outdir=outdir,
        array_type="compliant",
        spring_values=spring_values,
        alpha_values=alpha_values,
        w_values=w_values,
        surface_kinds=surface_kinds,
    )


def _run_screen(
    *,
    project_name: str,
    stage_name: str,
    surface_bank: str | Path,
    n_surfaces_per_kind: int,
    outdir: str | Path,
    array_type: str,
    spring_values: tuple[float, ...] | None,
    alpha_values: tuple[float, ...],
    w_values: tuple[float, ...],
    surface_kinds: tuple[str, ...],
) -> Path:
    pd = _require_pandas()
    stage_dir = Path(outdir)
    for path in (stage_dir / "data", stage_dir / "sample_cases", stage_dir / "figures_report", stage_dir / "reports"):
        path.mkdir(parents=True, exist_ok=True)
    bank = SurfaceBank.open(surface_bank)
    selected_surfaces = _select_surface_ids(bank, surface_kinds, n_surfaces_per_kind)

    summary_records: list[dict[str, Any]] = []
    spine_records: list[dict[str, Any]] = []
    case_count = 0
    for surface_kind in surface_kinds:
        for surface_id in selected_surfaces[surface_kind]:
            for alpha in alpha_values:
                for w_total_n in w_values:
                    springs = spring_values if spring_values is not None else (None,)
                    for spring_k in springs:
                        candidate_id = (
                            f"k{int(spring_k)}_alpha{int(alpha)}"
                            if spring_k is not None
                            else f"alpha{int(alpha)}"
                        )
                        case_id = (
                            f"{stage_name}_{candidate_id}_W{str(w_total_n).replace('.', 'p')}_{surface_id}"
                        )
                        result = run_single_case(
                            SingleCaseInput(
                                surface_bank_path=Path(surface_bank),
                                surface_id=surface_id,
                                array_type=array_type,
                                rows=1,
                                cols=1,
                                pitch_t_mm=4.0,
                                pitch_l_mm=4.0,
                                alpha_p_deg=alpha,
                                spring_k_n_per_m=spring_k,
                                tip_radius_mm=0.05,
                                spine_diameter_mm=0.20,
                                search_travel_mm=4.0,
                                w_total_n=w_total_n,
                                f_s=1.0,
                                F_ref_star_n=0.50,
                                trial_force_n=0.05,
                                candidate_id=candidate_id,
                                case_id=case_id,
                            )
                        )
                        summary_records.append(result.case_summary.iloc[0].to_dict())
                        spine_records.extend(result.case_spines.to_dict(orient="records"))
                        case_count += 1

    summary = pd.DataFrame.from_records(summary_records)
    spines = pd.DataFrame.from_records(spine_records)
    data_dir = stage_dir / "data"
    write_parquet(summary, data_dir / "stage_summary.parquet")
    write_preview_csv(summary, data_dir / "stage_summary_preview.csv")
    write_parquet(spines, data_dir / "stage_spines.parquet")
    write_schema(stage_dir)
    manifest = create_manifest(
        project_name="Spine_Sim_V2",
        model_version=project_name,
        surface_bank_id=bank.bank_id,
        random_seed_policy="deterministic scan over a pre-generated surface bank",
        parameter_grid={
            "array_type": array_type,
            "rows": 1,
            "cols": 1,
            "spring_k_n_per_m": list(spring_values) if spring_values is not None else [None],
            "alpha_p_deg": list(alpha_values),
            "w_total_n": list(w_values),
            "surface_kinds": list(surface_kinds),
            "n_surfaces_per_kind": n_surfaces_per_kind,
        },
        n_cases_expected=case_count,
        n_cases_completed=len(summary),
        failed_cases=[],
        notes="Single-spine screening stage; no surface arrays are duplicated in stage tables.",
    )
    write_manifest(manifest, stage_dir)
    analyze_stage(stage_dir)
    _write_stage_report(stage_dir, project_name)
    return stage_dir


def _select_surface_ids(
    bank: SurfaceBank,
    surface_kinds: tuple[str, ...],
    n_surfaces_per_kind: int,
) -> dict[str, list[str]]:
    if n_surfaces_per_kind <= 0:
        raise ValueError("n_surfaces_per_kind must be positive.")
    stats = bank.load_statistics()
    selected: dict[str, list[str]] = {}
    for kind in surface_kinds:
        ids = stats.loc[stats["surface_kind"] == kind, "surface_id"].sort_values().tolist()
        if len(ids) < n_surfaces_per_kind:
            raise ValueError(
                f"Surface bank {bank.root} has {len(ids)} {kind!r} surfaces, "
                f"but {n_surfaces_per_kind} are required."
            )
        selected[kind] = [str(item) for item in ids[:n_surfaces_per_kind]]
    return selected


def _write_stage_report(stage_dir: Path, project_name: str) -> None:
    from Spine_Sim_V2.io.parquet_io import read_parquet

    summary = read_parquet(stage_dir / "data" / "stage_summary.parquet")
    rankings = read_parquet(stage_dir / "data" / "stage_rankings.parquet")
    lines = [
        f"# {project_name}",
        "",
        f"Cases completed: {len(summary)}",
        "",
        "## Selected Candidates",
        "",
        "| rank | candidate_id | score_total |",
        "|---:|---|---:|",
    ]
    for _, row in rankings.loc[rankings["selected"] == True].sort_values("rank").iterrows():  # noqa: E712
        lines.append(f"| {int(row['rank'])} | {row['candidate_id']} | {float(row['score_total']):.4f} |")
    (stage_dir / "reports" / "stage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("P2/P3 screening requires pandas.") from exc
    return pd
