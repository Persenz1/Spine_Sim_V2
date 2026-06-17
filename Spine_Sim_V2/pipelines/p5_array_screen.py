"""P5：阵列规模与间距筛选管线。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from Spine_Sim_V2.analysis.ranking import analyze_stage
from Spine_Sim_V2.core.parallel import map_tasks_unordered, resolve_worker_count
from Spine_Sim_V2.core.progress import ProgressReporter
from Spine_Sim_V2.core.types import SingleCaseInput, stage_spines_schema, stage_summary_schema
from Spine_Sim_V2.io.manifest import create_manifest, write_manifest
from Spine_Sim_V2.io.parquet_io import BatchedParquetWriter, write_preview_csv
from Spine_Sim_V2.io.schema_io import write_schema
from Spine_Sim_V2.pipelines.case_tasks import CaseJob, run_case_job
from Spine_Sim_V2.surfaces.bank import SurfaceBank


P5A_PROJECT_NAME = "P5a_array_pitch_coarse_screen"
P5B_PROJECT_NAME = "P5b_array_pitch_refine_screen"
P5_ROWS = (2, 3, 4, 5)
P5_COLS = (2, 3, 4, 5, 6, 7)
P5_PITCH_T_MM = (4.0, 5.0, 6.0)
P5_PITCH_L_MM = (4.0, 5.0, 6.0)
P5_W_TOTAL_N = (0.5, 1.0, 1.5, 2.0, 2.5)
P5_SURFACE_KINDS = ("sandpaper", "concrete", "brick", "painted_wall")


@dataclass(frozen=True)
class ArrayCandidate:
    """完整阵列候选参数。"""

    candidate_id: str
    array_type: str
    rows: int
    cols: int
    pitch_t_mm: float
    pitch_l_mm: float
    alpha_p_deg: float
    spring_k_n_per_m: float | None
    source_candidate_id: str | None = None
    source_stage: str | None = None

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
            "source_candidate_id": self.source_candidate_id,
            "source_stage": self.source_stage,
        }


def run_p5a(
    *,
    surface_bank: str | Path,
    p2_selected: str | Path,
    p3_selected: str | Path,
    n_surfaces_per_kind: int = 50,
    outdir: str | Path = "outputs/P5a_array_pitch_coarse_screen",
    rows: tuple[int, ...] = P5_ROWS,
    cols: tuple[int, ...] = P5_COLS,
    pitch_t_values: tuple[float, ...] = P5_PITCH_T_MM,
    pitch_l_values: tuple[float, ...] = P5_PITCH_L_MM,
    w_values: tuple[float, ...] = P5_W_TOTAL_N,
    max_p2_selected: int = 6,
    max_p3_selected: int = 2,
    workers: int | None = None,
) -> Path:
    """运行 P5a 阵列规模与间距粗筛。"""
    base_candidates = _build_p5a_candidates(
        p2_selected=p2_selected,
        p3_selected=p3_selected,
        rows=rows,
        cols=cols,
        pitch_t_values=pitch_t_values,
        pitch_l_values=pitch_l_values,
        max_p2_selected=max_p2_selected,
        max_p3_selected=max_p3_selected,
    )
    return _run_p5_stage(
        project_name=P5A_PROJECT_NAME,
        stage_name="p5a_array_pitch_coarse",
        surface_bank=surface_bank,
        candidates=base_candidates,
        n_surfaces_per_kind=n_surfaces_per_kind,
        w_values=w_values,
        outdir=outdir,
        workers=workers,
    )


def run_p5b(
    *,
    surface_bank: str | Path,
    p5a_selected: str | Path,
    n_surfaces_per_kind: int = 200,
    outdir: str | Path = "outputs/P5b_array_pitch_refine_screen",
    w_values: tuple[float, ...] = P5_W_TOTAL_N,
    workers: int | None = None,
) -> Path:
    """基于 P5a 入围候选运行 P5b 复筛。"""
    candidates = [
        _candidate_from_record(record)
        for record in _load_selected_records(p5a_selected)
    ]
    return _run_p5_stage(
        project_name=P5B_PROJECT_NAME,
        stage_name="p5b_array_pitch_refine",
        surface_bank=surface_bank,
        candidates=candidates,
        n_surfaces_per_kind=n_surfaces_per_kind,
        w_values=w_values,
        outdir=outdir,
        workers=workers,
    )


def run() -> None:
    """已废弃的 P5 通用入口；请使用 ``run_p5a`` 或 ``run_p5b``。"""
    raise NotImplementedError("Use run_p5a or run_p5b.")


def _run_p5_stage(
    *,
    project_name: str,
    stage_name: str,
    surface_bank: str | Path,
    candidates: list[ArrayCandidate],
    n_surfaces_per_kind: int,
    w_values: tuple[float, ...],
    outdir: str | Path,
    workers: int | None = None,
) -> Path:
    """运行 P5a/P5b 共用的大规模阵列筛选流程（并行 + 流式落盘 + 进度条）。"""
    pd = _require_pandas()
    stage_dir = Path(outdir)
    data_dir = stage_dir / "data"
    for path in (data_dir, stage_dir / "reports", stage_dir / "figures_report", stage_dir / "sample_cases"):
        path.mkdir(parents=True, exist_ok=True)
    bank = SurfaceBank.open(surface_bank)
    selected_surfaces = _select_surface_ids(bank, P5_SURFACE_KINDS, n_surfaces_per_kind)
    n_jobs = len(candidates) * len(P5_SURFACE_KINDS) * n_surfaces_per_kind * len(w_values)
    jobs = _iter_p5_jobs(
        surface_bank=Path(surface_bank),
        bank_id=bank.bank_id,
        stage_name=stage_name,
        candidates=candidates,
        selected_surfaces=selected_surfaces,
        w_values=w_values,
    )

    preview_rows: list[dict[str, Any]] = []
    failed_cases: list[str] = []
    case_count = 0
    summary_writer = BatchedParquetWriter(data_dir / "stage_summary.parquet", schema=stage_summary_schema)
    spines_writer = BatchedParquetWriter(data_dir / "stage_spines.parquet", schema=stage_spines_schema)
    # P5 case 数可能很大：有界并行 + 批量 row group 写盘，全程内存占用恒定。
    with summary_writer, spines_writer, ProgressReporter(n_jobs, label=project_name) as bar:
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

    write_preview_csv(pd.DataFrame.from_records(preview_rows), data_dir / "stage_summary_preview.csv")
    write_schema(stage_dir)
    write_manifest(
        create_manifest(
            project_name="Spine_Sim_V2",
            model_version=project_name,
            surface_bank_id=bank.bank_id,
            random_seed_policy="deterministic scan over selected candidates and pre-generated surfaces",
            parameter_grid={
                "stage_name": stage_name,
                "n_candidates": len(candidates),
                "n_surfaces_per_kind": n_surfaces_per_kind,
                "w_total_n": list(w_values),
                "surface_kinds": list(P5_SURFACE_KINDS),
                "workers": resolve_worker_count(workers),
            },
            n_cases_expected=n_jobs,
            n_cases_completed=case_count - len(failed_cases),
            failed_cases=failed_cases,
            notes="P5 array pitch screen; full 2D arrays are not saved.",
        ),
        stage_dir,
    )
    grouped, rankings = analyze_stage(stage_dir)
    _write_stage_report(stage_dir, project_name, rankings)
    return stage_dir


def _iter_p5_jobs(
    *,
    surface_bank: Path,
    bank_id: str,
    stage_name: str,
    candidates: list[ArrayCandidate],
    selected_surfaces: dict[str, list[str]],
    w_values: tuple[float, ...],
) -> Iterator[CaseJob]:
    """展开 P5 的候选 × 表面类别 × 表面 × 预载组合。"""
    for candidate in candidates:
        for surface_kind in P5_SURFACE_KINDS:
            for surface_index, surface_id in enumerate(selected_surfaces[surface_kind]):
                for w_total_n in w_values:
                    case_id = _case_id(stage_name, candidate.candidate_id, w_total_n, surface_id)
                    yield CaseJob(
                        case=SingleCaseInput(
                            surface_bank_path=surface_bank,
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
                        stage=stage_name,
                        surface_kind=surface_kind,
                        surface_index_within_kind=surface_index,
                    )


def _build_p5a_candidates(
    *,
    p2_selected: str | Path,
    p3_selected: str | Path,
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    pitch_t_values: tuple[float, ...],
    pitch_l_values: tuple[float, ...],
    max_p2_selected: int,
    max_p3_selected: int,
) -> list[ArrayCandidate]:
    """把 P2/P3 入围单刺候选扩展成 P5a 阵列几何候选。"""
    p2_records = _load_selected_records(p2_selected)[:max_p2_selected]
    p3_records = _load_selected_records(p3_selected)[:max_p3_selected]
    candidates: list[ArrayCandidate] = []
    rigid_counter = 1
    compliant_counter = 1
    for record in p3_records:
        for geometry in _geometry_grid(rows, cols, pitch_t_values, pitch_l_values):
            candidates.append(
                ArrayCandidate(
                    candidate_id=f"R{rigid_counter:03d}",
                    array_type="rigid",
                    alpha_p_deg=float(record["alpha_p_deg"]),
                    spring_k_n_per_m=None,
                    source_candidate_id=str(record.get("candidate_id")),
                    source_stage=str(record.get("source_stage")),
                    **geometry,
                )
            )
            rigid_counter += 1
    for record in p2_records:
        for geometry in _geometry_grid(rows, cols, pitch_t_values, pitch_l_values):
            candidates.append(
                ArrayCandidate(
                    candidate_id=f"C{compliant_counter:03d}",
                    array_type="compliant",
                    alpha_p_deg=float(record["alpha_p_deg"]),
                    spring_k_n_per_m=float(record["spring_k_n_per_m"]),
                    source_candidate_id=str(record.get("candidate_id")),
                    source_stage=str(record.get("source_stage")),
                    **geometry,
                )
            )
            compliant_counter += 1
    return candidates


def _geometry_grid(
    rows: tuple[int, ...],
    cols: tuple[int, ...],
    pitch_t_values: tuple[float, ...],
    pitch_l_values: tuple[float, ...],
) -> Iterable[dict[str, Any]]:
    """枚举 P5 的 rows/cols/pitch_t/pitch_l 几何网格。"""
    for row_count in rows:
        for col_count in cols:
            for pitch_t_mm in pitch_t_values:
                for pitch_l_mm in pitch_l_values:
                    yield {
                        "rows": int(row_count),
                        "cols": int(col_count),
                        "pitch_t_mm": float(pitch_t_mm),
                        "pitch_l_mm": float(pitch_l_mm),
                    }


def _candidate_from_record(record: dict[str, Any]) -> ArrayCandidate:
    """从 selected_candidates 记录恢复阵列候选对象。"""
    return ArrayCandidate(
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
        source_candidate_id=record.get("source_candidate_id") or record.get("candidate_id"),
        source_stage=record.get("source_stage"),
    )


def _load_selected_records(path: str | Path) -> list[dict[str, Any]]:
    """读取上游阶段的 selected_candidates.json。"""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Selected candidate file must contain a list: {path}")
    return records


def _select_surface_ids(
    bank: SurfaceBank,
    surface_kinds: tuple[str, ...],
    n_surfaces_per_kind: int,
) -> dict[str, list[str]]:
    """按表面类别选取固定数量的 surface_id。"""
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


def _case_id(stage_name: str, candidate_id: str, w_total_n: float, surface_id: str) -> str:
    """生成阶段内可追溯的 case_id。"""
    return f"{stage_name}_{candidate_id}_W{str(w_total_n).replace('.', 'p')}_{surface_id}"


def _write_stage_report(stage_dir: Path, project_name: str, rankings: Any) -> None:
    """写出 P5 入围候选简报。"""
    lines = [
        f"# {project_name}",
        "",
        "## Selected Candidates",
        "",
        "| rank | candidate_id | array_type | score_total | reason |",
        "|---:|---|---|---:|---|",
    ]
    selected = rankings.loc[rankings["selected"] == True].sort_values("rank")  # noqa: E712
    for _, row in selected.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['candidate_id']} | {row['array_type']} | {float(row['score_total']):.4f} | {row['selection_reason']} |"
        )
    (stage_dir / "reports" / "stage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("P5 screening requires pandas.") from exc
    return pd
