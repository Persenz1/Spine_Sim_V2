"""P6：对 P5b 最终候选执行正式 3D Monte Carlo。"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from Spine_Sim_V2.analysis.final_mc import p6_schema_collection, write_final_analysis
from Spine_Sim_V2.core.types import SingleCaseInput, SchemaField, stage_spines_schema, stage_summary_schema
from Spine_Sim_V2.io.manifest import create_manifest, write_manifest
from Spine_Sim_V2.io.parquet_io import write_preview_csv
from Spine_Sim_V2.io.schema_io import write_schema
from Spine_Sim_V2.pipelines.p1_single_case import run_single_case
from Spine_Sim_V2.surfaces.bank import SurfaceBank


P6_PROJECT_NAME = "P6_final_3d_monte_carlo"
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


@dataclass(frozen=True)
class CaseTask:
    """一个 P6 Monte Carlo case 的执行请求。"""

    case_id: str
    candidate: FinalCandidate
    surface_id: str
    surface_kind: str
    surface_index_within_kind: int
    w_total_n: float
    surface_bank_path: Path


def run(
    *,
    surface_bank: str | Path,
    selected_candidates: str | Path,
    n_surfaces_per_kind: int = 1000,
    surface_selection: str = "first_n",
    outdir: str | Path = "outputs/P6_final_3d_monte_carlo",
    workers: int = 1,
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
    # P6 是正式统计源：候选 × 表面类别 × 预载 × surface_id 全组合展开。
    tasks = _iter_case_tasks(
        candidates=candidates,
        surface_bank_path=Path(surface_bank),
        selected_surfaces=selected_surfaces,
        w_values=w_values,
    )

    summary_writer = _ParquetStreamWriter(data_dir / "final_summary.parquet", schema=stage_summary_schema)
    spines_writer = _ParquetStreamWriter(data_dir / "final_spines.parquet", schema=stage_spines_schema)
    preview_rows: list[dict[str, Any]] = []
    case_count = 0
    failed_cases: list[str] = []
    try:
        # 正式样本量较大，summary/spines 均流式写入，避免一次性占用大量内存。
        for summary_df, spines_df, failed_case_id in _run_tasks(tasks, workers=max(1, int(workers))):
            summary_writer.write(summary_df)
            spines_writer.write(spines_df)
            if len(preview_rows) < 5000 and len(summary_df):
                preview_rows.append(summary_df.iloc[0].to_dict())
            if failed_case_id is not None:
                failed_cases.append(failed_case_id)
            case_count += len(summary_df)
    finally:
        summary_writer.close()
        spines_writer.close()

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
                "workers": int(workers),
            },
            n_cases_expected=len(candidates) * len(P6_SURFACE_KINDS) * n_surfaces_per_kind * len(w_values),
            n_cases_completed=case_count,
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


def _iter_case_tasks(
    *,
    candidates: list[FinalCandidate],
    surface_bank_path: Path,
    selected_surfaces: dict[str, list[str]],
    w_values: tuple[float, ...],
) -> Iterable[CaseTask]:
    """展开 P6 的候选、表面和预载组合。"""
    for candidate in candidates:
        for surface_kind in P6_SURFACE_KINDS:
            for index, surface_id in enumerate(selected_surfaces[surface_kind]):
                for w_total_n in w_values:
                    case_id = _case_id(candidate.candidate_id, w_total_n, surface_id)
                    yield CaseTask(
                        case_id=case_id,
                        candidate=candidate,
                        surface_id=surface_id,
                        surface_kind=surface_kind,
                        surface_index_within_kind=index,
                        w_total_n=w_total_n,
                        surface_bank_path=surface_bank_path,
                    )


def _run_tasks(tasks: Iterable[CaseTask], *, workers: int) -> Iterable[tuple[Any, Any, str | None]]:
    """顺序或线程池执行 P6 case 任务。"""
    if workers <= 1:
        for task in tasks:
            yield _run_one_task(task)
        return

    pending: set[Future[tuple[Any, Any, str | None]]] = set()
    iterator = iter(tasks)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(max(1, workers * 4)):
            try:
                pending.add(executor.submit(_run_one_task, next(iterator)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            for _ in range(len(done)):
                try:
                    pending.add(executor.submit(_run_one_task, next(iterator)))
                except StopIteration:
                    break


def _run_one_task(task: CaseTask) -> tuple[Any, Any, str | None]:
    """执行单个 P6 case；异常 case 转成失败 summary，不静默丢弃。"""
    try:
        result = run_single_case(
            SingleCaseInput(
                surface_bank_path=task.surface_bank_path,
                surface_id=task.surface_id,
                array_type=task.candidate.array_type,
                rows=task.candidate.rows,
                cols=task.candidate.cols,
                pitch_t_mm=task.candidate.pitch_t_mm,
                pitch_l_mm=task.candidate.pitch_l_mm,
                alpha_p_deg=task.candidate.alpha_p_deg,
                spring_k_n_per_m=task.candidate.spring_k_n_per_m,
                tip_radius_mm=0.05,
                spine_diameter_mm=0.20,
                search_travel_mm=4.0,
                w_total_n=task.w_total_n,
                f_s=1.0,
                F_ref_star_n=0.50,
                trial_force_n=0.05,
                candidate_id=task.candidate.candidate_id,
                case_id=task.case_id,
            )
        )
        result.case_summary.loc[:, "stage"] = "p6_final_3d_monte_carlo"
        result.case_summary.loc[:, "surface_index_within_kind"] = task.surface_index_within_kind
        result.case_spines.loc[:, "stage"] = "p6_final_3d_monte_carlo"
        return result.case_summary, result.case_spines, None
    except Exception as exc:  # pragma: no cover - hard to trigger deterministically.
        pd = _require_pandas()
        summary = pd.DataFrame.from_records([_failed_summary_record(task, exc)])
        spines = pd.DataFrame(columns=[field.name for field in stage_spines_schema])
        return summary, spines, task.case_id


def _failed_summary_record(task: CaseTask, exc: Exception) -> dict[str, Any]:
    """为失败 case 构造保留字段完整性的 summary 行。"""
    record = {field.name: None for field in stage_summary_schema}
    candidate = task.candidate
    record.update(
        {
            "case_id": task.case_id,
            "stage": "p6_final_3d_monte_carlo",
            "case_status": "failed",
            "error_code": type(exc).__name__,
            "warning_flags": [str(exc)],
            "surface_bank_id": None,
            "surface_id": task.surface_id,
            "surface_kind": task.surface_kind,
            "surface_seed": None,
            "candidate_id": candidate.candidate_id,
            "array_type": candidate.array_type,
            "rows": candidate.rows,
            "cols": candidate.cols,
            "n_nom": candidate.rows * candidate.cols,
            "pitch_t_mm": candidate.pitch_t_mm,
            "pitch_l_mm": candidate.pitch_l_mm,
            "alpha_p_deg": candidate.alpha_p_deg,
            "spring_k_n_per_m": candidate.spring_k_n_per_m,
            "spring_k_n_per_mm": None if candidate.spring_k_n_per_m is None else candidate.spring_k_n_per_m / 1000.0,
            "tip_radius_mm": 0.05,
            "spine_diameter_mm": 0.20,
            "search_travel_mm": 4.0,
            "w_total_n": task.w_total_n,
            "f_s": 1.0,
            "phi_s_deg": 45.0,
            "F_ref_star_n": 0.50,
            "trial_force_n": 0.05,
            "n_con": 0,
            "n_eng": 0,
            "n_eff_count": 0,
            "n_eff_kish": 0.0,
            "r_con": 0.0,
            "r_eng": 0.0,
            "r_fail_search": 1.0,
            "normal_range_insufficient": False,
            "f_t_lim_n": 0.0,
            "f_t_lim_over_w_total": 0.0,
            "f_t_lim_per_nom_n": 0.0,
            "f_t_lim_per_eff_n": None,
            "limit_displacement_mm": 0.0,
            "eta_max": 0.0,
            "engagement_success": False,
            "load_success": False,
            "failure_mode": "case_failed",
            "cascade_failure": False,
            "r_slip": 0.0,
            "r_overload": 0.0,
            "r_side_contact_risk": 0.0,
            "surface_index_within_kind": task.surface_index_within_kind,
        }
    )
    return record


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


class _ParquetStreamWriter:
    """按固定 schema 将 DataFrame 追加写为 Parquet row group。"""

    def __init__(self, path: Path, *, schema: tuple[SchemaField, ...]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Any = None
        self._schema_fields = schema
        self._arrow_schema: Any = None
        self._has_written = False

    def write(self, df: Any) -> None:
        if df is None:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        normalized = _normalize_for_arrow_schema(df, self._schema_fields)
        if self._arrow_schema is None:
            self._arrow_schema = _arrow_schema(self._schema_fields)
        table = pa.Table.from_pandas(normalized, schema=self._arrow_schema, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        self._has_written = True

    def close(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self._has_written:
            schema = self._arrow_schema or _arrow_schema(self._schema_fields)
            arrays = [pa.array([], type=field.type) for field in schema]
            pq.write_table(pa.Table.from_arrays(arrays, schema=schema), self.path)


def _normalize_for_arrow_schema(df: Any, schema: tuple[SchemaField, ...]) -> Any:
    pd = _require_pandas()
    normalized = df.copy()
    for field in schema:
        if field.name not in normalized.columns:
            normalized[field.name] = None
        if field.dtype == "float64":
            normalized[field.name] = pd.to_numeric(normalized[field.name], errors="coerce").astype("float64")
        elif field.dtype == "int64":
            normalized[field.name] = pd.to_numeric(normalized[field.name], errors="coerce").astype("Int64")
        elif field.dtype == "bool":
            normalized[field.name] = normalized[field.name].astype("boolean")
        elif field.dtype == "string":
            normalized[field.name] = normalized[field.name].astype("string")
        elif field.dtype == "list[string]":
            normalized[field.name] = normalized[field.name].apply(_string_list)
    return normalized[[field.name for field in schema]]


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if value != value:
            return []
    except ValueError:
        pass
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _arrow_schema(schema: tuple[SchemaField, ...]) -> Any:
    import pyarrow as pa

    return pa.schema([pa.field(field.name, _arrow_type(field.dtype), nullable=field.nullable) for field in schema])


def _arrow_type(dtype: str) -> Any:
    import pyarrow as pa

    if dtype == "float64":
        return pa.float64()
    if dtype == "int64":
        return pa.int64()
    if dtype == "bool":
        return pa.bool_()
    if dtype == "list[string]":
        return pa.list_(pa.string())
    return pa.string()
