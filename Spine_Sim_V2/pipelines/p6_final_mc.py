"""P6：对 P5b 最终候选执行正式 3D Monte Carlo。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from Spine_Sim_V2.analysis.final_mc import p6_schema_collection, write_final_analysis
from Spine_Sim_V2.core.parallel import map_tasks_unordered, resolve_worker_count
from Spine_Sim_V2.core.progress import ProgressReporter
from Spine_Sim_V2.core.types import SingleCaseInput, stage_spines_schema, stage_summary_schema
from Spine_Sim_V2.io.manifest import create_manifest, write_manifest
from Spine_Sim_V2.io.parquet_io import BatchedParquetWriter, write_preview_csv
from Spine_Sim_V2.io.schema_io import write_schema
from Spine_Sim_V2.pipelines.case_tasks import CaseJob, run_case_job
from Spine_Sim_V2.surfaces.bank import SurfaceBank


P6_PROJECT_NAME = "P6_final_3d_monte_carlo"
P6_STAGE = "p6_final_3d_monte_carlo"
P6_SURFACE_KINDS = ("sandpaper", "concrete", "brick", "painted_wall")
P6_W_TOTAL_N = (0.5, 1.0, 1.5, 2.0, 2.5)
P6_RANDOM_SEED = 20260617


@dataclass(frozen=True)
class FinalCandidate:
    """从 P5b 传入 P6 的最终候选参数。"""

    candidate_id: str
    array_type: str
    rows: int
    cols: int
    pitch_t_mm: float
    pitch_l_mm: float
    alpha_p_deg: float
    spring_k_n_per_m: float | None
    source_stage: str | None = None
    selection_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "array_type": self.array_type,
            "rows": self.rows,
            "cols": self.cols,
            "pitch_t_mm": self.pitch_t_mm,
            "pitch_l_mm": self.pitch_l_mm,
            "alpha_p_deg": self.alpha_p_deg,
            "spring_k_n_per_m": self.spring_k_n_per_m,
            "source_stage": self.source_stage,
            "selection_reason": self.selection_reason,
        }


def run(
    *,
    surface_bank: str | Path,
    selected_candidates: str | Path,
    n_surfaces_per_kind: int = 1000,
    surface_selection: str = "first_n",
    outdir: str | Path = "outputs/P6_final_3d_monte_carlo",
    workers: int | None = None,
    surface_list: str | Path | None = None,
    random_seed: int = P6_RANDOM_SEED,
    w_values: tuple[float, ...] = P6_W_TOTAL_N,
) -> Path:
    """运行 P6 最终 Monte Carlo，并保存 ``final_*`` 数据产品。"""
    pd = _require_pandas()
    stage_dir = Path(outdir)
    data_dir = stage_dir / "data"
    for path in (data_dir, stage_dir / "reports", stage_dir / "figures_report", stage_dir / "sample_cases"):
        path.mkdir(parents=True, exist_ok=True)

    bank = SurfaceBank.open(surface_bank)
    candidates = _load_final_candidates(selected_candidates)
    selected_surfaces = select_surface_ids(
        bank=bank,
        surface_kinds=P6_SURFACE_KINDS,
        n_surfaces_per_kind=n_surfaces_per_kind,
        surface_selection=surface_selection,
        surface_list=surface_list,
        random_seed=random_seed,
    )
    n_jobs = len(candidates) * len(P6_SURFACE_KINDS) * n_surfaces_per_kind * len(w_values)
    # P6 是正式统计源：候选 × 表面类别 × 表面 × 预载全组合，懒展开成 case 任务。
    jobs = _iter_p6_jobs(
        candidates=candidates,
        surface_bank_path=Path(surface_bank),
        bank_id=bank.bank_id,
        selected_surfaces=selected_surfaces,
        w_values=w_values,
    )

    summary_writer = BatchedParquetWriter(data_dir / "final_summary.parquet", schema=stage_summary_schema)
    spines_writer = BatchedParquetWriter(data_dir / "final_spines.parquet", schema=stage_spines_schema)
    preview_rows: list[dict[str, Any]] = []
    case_count = 0
    failed_cases: list[str] = []
    # 正式样本量大：有界并行 + 批量 row group 流式写入，全程内存占用恒定。
    with summary_writer, spines_writer, ProgressReporter(n_jobs, label=P6_PROJECT_NAME) as bar:
        for summary_record, spine_records, failed_id in map_tasks_unordered(
            run_case_job, jobs, workers=workers
        ):
            summary_writer.add_record(summary_record)
            spines_writer.add_records(spine_records)
            if len(preview_rows) < 5000:
                preview_rows.append(summary_record)
            if failed_id is not None:
                failed_cases.append(failed_id)
            case_count += 1
            bar.update()

    write_preview_csv(pd.DataFrame.from_records(preview_rows), data_dir / "final_summary_preview.csv")
    write_manifest(
        create_manifest(
            project_name="Spine_Sim_V2",
            model_version=P6_PROJECT_NAME,
            surface_bank_id=bank.bank_id,
            random_seed_policy=(
                "first_n is deterministic; random_fixed uses a fixed numpy seed; "
                "explicit_list is deterministic from user-provided ids"
            ),
            parameter_grid={
                "selected_candidates": str(selected_candidates),
                "n_candidates": len(candidates),
                "surface_kinds": list(P6_SURFACE_KINDS),
                "n_surfaces_per_kind": n_surfaces_per_kind,
                "surface_selection": surface_selection,
                "random_seed": random_seed,
                "w_total_n": list(w_values),
                "workers": resolve_worker_count(workers),
            },
            n_cases_expected=n_jobs,
            n_cases_completed=case_count - len(failed_cases),
            failed_cases=failed_cases,
            notes="P6 final Monte Carlo. Full 2D arrays are not saved; cases reference surface_id.",
        ),
        stage_dir,
    )
    grouped, rankings, convergence = write_final_analysis(stage_dir)
    write_schema(stage_dir, p6_schema_collection(grouped, rankings, convergence))
    _write_sample_cases(stage_dir)
    _write_final_report(stage_dir)
    return stage_dir


def _iter_p6_jobs(
    *,
    candidates: list[FinalCandidate],
    surface_bank_path: Path,
    bank_id: str,
    selected_surfaces: dict[str, list[str]],
    w_values: tuple[float, ...],
) -> Iterator[CaseJob]:
    """展开 P6 的候选、表面和预载组合为可并行的 case 任务。"""
    for candidate in candidates:
        for surface_kind in P6_SURFACE_KINDS:
            for surface_index, surface_id in enumerate(selected_surfaces[surface_kind]):
                for w_total_n in w_values:
                    case_id = _case_id(candidate.candidate_id, w_total_n, surface_id)
                    yield CaseJob(
                        case=SingleCaseInput(
                            surface_bank_path=surface_bank_path,
                            surface_id=surface_id,
                            array_type=candidate.array_type,
                            rows=candidate.rows,
                            cols=candidate.cols,
                            pitch_t_mm=candidate.pitch_t_mm,
                            pitch_l_mm=candidate.pitch_l_mm,
                            alpha_p_deg=candidate.alpha_p_deg,
                            spring_k_n_per_m=candidate.spring_k_n_per_m,
                            tip_radius_mm=0.05,
                            spine_diameter_mm=0.20,
                            search_travel_mm=4.0,
                            w_total_n=w_total_n,
                            f_s=1.0,
                            F_ref_star_n=0.50,
                            trial_force_n=0.50,
                            candidate_id=candidate.candidate_id,
                            case_id=case_id,
                        ),
                        surface_bank_id=bank_id,
                        stage=P6_STAGE,
                        surface_kind=surface_kind,
                        surface_index_within_kind=surface_index,
                    )


def select_surface_ids(
    *,
    bank: SurfaceBank,
    surface_kinds: tuple[str, ...],
    n_surfaces_per_kind: int,
    surface_selection: str,
    surface_list: str | Path | None = None,
    random_seed: int = P6_RANDOM_SEED,
) -> dict[str, list[str]]:
    """按 ``first_n``、``random_fixed`` 或 ``explicit_list`` 策略选择 P6 表面。"""
    if n_surfaces_per_kind <= 0:
        raise ValueError("n_surfaces_per_kind must be positive.")
    if surface_selection not in {"first_n", "random_fixed", "explicit_list"}:
        raise ValueError("surface_selection must be first_n, random_fixed, or explicit_list.")

    stats = bank.load_statistics()
    by_kind = {
        kind: [str(item) for item in stats.loc[stats["surface_kind"] == kind, "surface_id"].sort_values()]
        for kind in surface_kinds
    }
    if surface_selection == "explicit_list":
        explicit = _load_explicit_surface_list(surface_list, stats)
        return _trim_surface_selection(explicit, by_kind, n_surfaces_per_kind)
    if surface_selection == "random_fixed":
        import numpy as np

        rng = np.random.default_rng(random_seed)
        selected: dict[str, list[str]] = {}
        for kind, ids in by_kind.items():
            _require_enough_surfaces(kind, ids, n_surfaces_per_kind)
            selected[kind] = sorted(str(item) for item in rng.choice(ids, size=n_surfaces_per_kind, replace=False))
        return selected
    return _trim_surface_selection(by_kind, by_kind, n_surfaces_per_kind)


def _load_final_candidates(path: str | Path) -> list[FinalCandidate]:
    """读取 P5b 产生的最终候选列表。"""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Selected candidate file must contain a list: {path}")
    candidates = [_candidate_from_record(record) for record in records]
    if not candidates:
        raise ValueError("selected_candidates.json does not contain any candidates.")
    return candidates


def _candidate_from_record(record: dict[str, Any]) -> FinalCandidate:
    """从 JSON 记录恢复 P6 最终候选对象。"""
    return FinalCandidate(
        candidate_id=str(record["candidate_id"]),
        array_type=str(record["array_type"]),
        rows=int(record["rows"]),
        cols=int(record["cols"]),
        pitch_t_mm=float(record["pitch_t_mm"]),
        pitch_l_mm=float(record["pitch_l_mm"]),
        alpha_p_deg=float(record["alpha_p_deg"]),
        spring_k_n_per_m=None
        if record.get("spring_k_n_per_m") is None
        else float(record["spring_k_n_per_m"]),
        source_stage=record.get("source_stage"),
        selection_reason=record.get("selection_reason"),
    )


def _write_sample_cases(stage_dir: Path) -> Path:
    """从 P6 summary 中挑选代表性 case，供后续人工复查。"""
    from Spine_Sim_V2.io.parquet_io import read_parquet

    summary = read_parquet(stage_dir / "data" / "final_summary.parquet")
    records: list[dict[str, Any]] = []
    for candidate_id, subset in summary.groupby("candidate_id", dropna=False):
        success = subset.loc[subset["load_success"] == True]  # noqa: E712
        failure = subset.loc[subset["load_success"] == False]  # noqa: E712
        if not success.empty:
            records.append(_sample_record("candidate_success", success.iloc[len(success) // 2]))
        if not failure.empty:
            records.append(_sample_record("candidate_failure", failure.iloc[0]))
    for surface_kind, subset in summary.groupby("surface_kind", dropna=False):
        if not subset.empty:
            median_force = subset["f_t_lim_n"].median()
            ordered = subset.assign(_distance=(subset["f_t_lim_n"] - median_force).abs()).sort_values("_distance")
            records.append(_sample_record("surface_typical", ordered.iloc[0]))
    if not summary.empty:
        records.append(_sample_record("highest_force", summary.sort_values("f_t_lim_n", ascending=False).iloc[0]))
        records.append(_sample_record("lowest_force", summary.sort_values("f_t_lim_n", ascending=True).iloc[0]))
        records.append(_sample_record("highest_eta", summary.sort_values("eta_max", ascending=False).iloc[0]))
    path = stage_dir / "sample_cases" / "sample_cases.json"
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sample_record(reason: str, row: Any) -> dict[str, Any]:
    """将一行 summary 转成 sample_cases.json 的精简记录。"""
    keys = [
        "case_id",
        "candidate_id",
        "array_type",
        "surface_bank_id",
        "surface_id",
        "surface_kind",
        "w_total_n",
        "rows",
        "cols",
        "pitch_t_mm",
        "pitch_l_mm",
        "alpha_p_deg",
        "spring_k_n_per_m",
        "f_t_lim_n",
        "load_success",
        "failure_mode",
        "eta_max",
    ]
    payload = {"sample_reason": reason}
    for key in keys:
        payload[key] = _json_value(row.get(key))
    return payload


def _write_final_report(stage_dir: Path) -> Path:
    """写出 P6 最终排名简报。"""
    from Spine_Sim_V2.io.parquet_io import read_parquet

    rankings = read_parquet(stage_dir / "data" / "final_rankings.parquet")
    summary = read_parquet(stage_dir / "data" / "final_summary.parquet")
    lines = [
        "# P6 Final 3D Monte Carlo Report",
        "",
        f"Cases: {len(summary)}",
        "",
        "## Final Candidate Ranking",
        "",
        "| rank | candidate_id | array_type | success_probability | f_t_lim_n_mean | score_total |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, row in rankings.sort_values("rank").iterrows():
        lines.append(
            "| {rank} | {candidate_id} | {array_type} | {success:.4f} | {force:.6g} | {score:.4f} |".format(
                rank=int(row["rank"]),
                candidate_id=row["candidate_id"],
                array_type=row["array_type"],
                success=float(row["success_probability"]),
                force=float(row["f_t_lim_n_mean"]),
                score=float(row["score_total"]),
            )
        )
    lines.extend(
        [
            "",
            "Figures are generated separately by `python scripts/plot_results.py stage ...`.",
            "",
        ]
    )
    path = stage_dir / "reports" / "final_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_explicit_surface_list(surface_list: str | Path | None, stats: Any) -> dict[str, list[str]]:
    """读取用户显式指定的 surface_id 列表，并按表面类别归组。"""
    if surface_list is None:
        raise ValueError("surface_selection=explicit_list requires --surface-list.")
    path = Path(surface_list)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = [item.strip() for item in str(surface_list).split(",") if item.strip()]

    if isinstance(payload, dict):
        return {str(kind): [str(item) for item in ids] for kind, ids in payload.items()}
    if isinstance(payload, list):
        ids: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                ids.append(str(item["surface_id"]))
            else:
                ids.append(str(item))
        result: dict[str, list[str]] = {kind: [] for kind in P6_SURFACE_KINDS}
        for surface_id in ids:
            matches = stats.loc[stats["surface_id"] == surface_id]
            if matches.empty:
                raise ValueError(f"Explicit surface_id is not present in bank: {surface_id}")
            kind = str(matches.iloc[0]["surface_kind"])
            result.setdefault(kind, []).append(surface_id)
        return result
    raise ValueError("explicit surface list must be a JSON dict/list or a comma-separated id list.")


def _trim_surface_selection(
    selected: dict[str, list[str]],
    available: dict[str, list[str]],
    n_surfaces_per_kind: int,
) -> dict[str, list[str]]:
    """裁剪并校验每类表面的选择结果。"""
    trimmed: dict[str, list[str]] = {}
    for kind, ids_available in available.items():
        ids = [str(item) for item in selected.get(kind, [])]
        _require_enough_surfaces(kind, ids, n_surfaces_per_kind)
        unknown = sorted(set(ids[:n_surfaces_per_kind]) - set(ids_available))
        if unknown:
            raise ValueError(f"Surface ids are not present for {kind}: {unknown}")
        trimmed[kind] = ids[:n_surfaces_per_kind]
    return trimmed


def _require_enough_surfaces(kind: str, ids: list[str], n_surfaces_per_kind: int) -> None:
    """确认某类表面数量满足 P6 要求。"""
    if len(ids) < n_surfaces_per_kind:
        raise ValueError(
            f"Surface bank has {len(ids)} {kind!r} surfaces, but {n_surfaces_per_kind} are required."
        )


def _case_id(candidate_id: str, w_total_n: float, surface_id: str) -> str:
    """生成 P6 case_id。"""
    return f"p6_{candidate_id}_W{str(w_total_n).replace('.', 'p')}_{surface_id}"


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("P6 final Monte Carlo requires pandas.") from exc
    return pd


def _json_value(value: Any) -> Any:
    try:
        import pandas as pd

        if pd.isna(value):
            return None
    except (ModuleNotFoundError, TypeError, ValueError):
        pass
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value
