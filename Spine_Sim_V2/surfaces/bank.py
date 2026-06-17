"""surface bank 的创建、读取和表面数组查询。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from Spine_Sim_V2.core.parallel import map_tasks_unordered, resolve_worker_count
from Spine_Sim_V2.core.progress import ProgressReporter
from Spine_Sim_V2.io.manifest import create_manifest, read_manifest, write_manifest
from Spine_Sim_V2.io.npz_io import load_npz_arrays, save_npz_arrays
from Spine_Sim_V2.io.parquet_io import read_parquet, write_parquet, write_preview_csv
from Spine_Sim_V2.io.schema_io import write_schema
from Spine_Sim_V2.surfaces.generator import (
    PROBE_FILTER_VERSION,
    SURFACE_GENERATOR_VERSION,
    generate_surface,
    make_surface_id,
    stable_surface_seed,
)
from Spine_Sim_V2.surfaces.profiles import parse_surface_kinds


@dataclass(frozen=True)
class _SurfaceGenJob:
    """单张代理表面的生成任务（可跨进程 pickle）。"""

    bank_id: str
    surface_kind: str
    index: int
    seed: int
    resolution_cells_per_mm: int
    size_x_mm: float
    size_y_mm: float
    tip_radius_mm: float
    surfaces_dir: str


def _generate_surface_job(job: _SurfaceGenJob) -> tuple[dict[str, object], str | None]:
    """在 worker 内生成一张表面、写出 NPZ，并只回传小体积的统计记录。

    只回传统计 dict（不回传大数组），把数组直接落盘，避免跨进程 IPC 传输大数组
    以及在主进程堆积内存。失败时回传带 ``valid=False`` 的记录，不静默丢弃。
    """
    surface_id = make_surface_id(job.surface_kind, job.index)
    try:
        generated = generate_surface(
            surface_kind=job.surface_kind,
            surface_id=surface_id,
            seed=job.seed,
            resolution_cells_per_mm=job.resolution_cells_per_mm,
            size_x_mm=job.size_x_mm,
            size_y_mm=job.size_y_mm,
            tip_radius_mm=job.tip_radius_mm,
        )
        save_npz_arrays(
            Path(job.surfaces_dir) / f"{surface_id}.npz",
            height_raw=generated.height_raw.astype(np.float32),
            height_filtered=generated.height_filtered.astype(np.float32),
        )
        record: dict[str, object] = {
            "surface_bank_id": job.bank_id,
            "surface_id": surface_id,
            "surface_kind": job.surface_kind,
            "seed": job.seed,
            **generated.statistics,
        }
        return record, None
    except Exception as exc:  # 失败表面保留为一行记录，标记 invalid，不静默跳过。
        record = {
            "surface_bank_id": job.bank_id,
            "surface_id": surface_id,
            "surface_kind": job.surface_kind,
            "seed": job.seed,
            "dx_mm": 1.0 / job.resolution_cells_per_mm,
            "dy_mm": 1.0 / job.resolution_cells_per_mm,
            "size_x_mm": job.size_x_mm,
            "size_y_mm": job.size_y_mm,
            "tip_radius_mm": job.tip_radius_mm,
            "rq_raw_mm": np.nan,
            "ra_raw_mm": np.nan,
            "hpv_raw_mm": np.nan,
            "rq_eff_mm": np.nan,
            "ra_eff_mm": np.nan,
            "hpv_eff_mm": np.nan,
            "slope_mean_deg": np.nan,
            "slope_p50_deg": np.nan,
            "slope_p95_deg": np.nan,
            "slope_max_deg": np.nan,
            "candidate_density_preload_free": np.nan,
            "valid": False,
            "reject_reason": f"{type(exc).__name__}: {exc}",
        }
        return record, surface_id


@dataclass(frozen=True)
class SurfaceBank:
    """对一个 surface bank 目录及其持久化元数据的引用。"""

    bank_id: str
    root: Path

    @classmethod
    def open(cls, root: str | Path) -> "SurfaceBank":
        """按目录打开已有 surface bank。"""
        root_path = Path(root)
        manifest = read_manifest(root_path)
        bank_id = manifest.get("surface_bank_id") or root_path.name
        return cls(bank_id=str(bank_id), root=root_path)

    @property
    def surfaces_dir(self) -> Path:
        """保存逐表面 NPZ 数组的目录。"""
        return self.root / "surfaces"

    @property
    def statistics_path(self) -> Path:
        """表面统计 Parquet 的规范路径。"""
        return self.root / "surface_statistics.parquet"

    def load_statistics(self) -> Any:
        """以 pandas DataFrame 读取 ``surface_statistics.parquet``。"""
        return _read_parquet_cached(str(self.statistics_path.resolve())).copy()

    def surface_path(self, surface_id: str) -> Path:
        """返回某个 ``surface_id`` 对应的 NPZ 路径。"""
        return self.surfaces_dir / f"{surface_id}.npz"

    def load_surface_arrays(self, surface_id: str) -> dict[str, np.ndarray]:
        """读取某个表面的 ``height_raw`` 和 ``height_filtered`` 数组。"""
        path = self.surface_path(surface_id)
        if not path.exists():
            raise FileNotFoundError(f"Surface array file not found for {surface_id!r}: {path}")
        arrays = {
            key: value.copy()
            for key, value in _load_npz_arrays_cached(str(path.resolve())).items()
        }
        expected = {"height_raw", "height_filtered"}
        missing = expected - set(arrays)
        if missing:
            raise KeyError(f"Surface {surface_id!r} is missing arrays: {sorted(missing)}")
        return arrays

    def get_surface_record(self, surface_id: str) -> dict[str, Any]:
        """返回某个 ``surface_id`` 的统计记录。"""
        stats = self.load_statistics()
        matches = stats.loc[stats["surface_id"] == surface_id]
        if matches.empty:
            raise KeyError(f"surface_id {surface_id!r} is not present in {self.statistics_path}")
        return matches.iloc[0].to_dict()


@lru_cache(maxsize=16)
def _read_parquet_cached(path: str) -> Any:
    return read_parquet(path)


@lru_cache(maxsize=128)
def _load_npz_arrays_cached(path: str) -> dict[str, np.ndarray]:
    return load_npz_arrays(path)


def create_surface_bank(
    *,
    bank_id: str,
    surface_kinds: str | list[str] | tuple[str, ...],
    n_per_kind: int,
    resolution_cells_per_mm: int = 5,
    size_x_mm: float = 60.0,
    size_y_mm: float = 40.0,
    tip_radius_mm: float = 0.05,
    outdir: str | Path | None = None,
    base_seed: int = 20260616,
    overwrite: bool = False,
    workers: int | None = None,
) -> SurfaceBank:
    """生成并持久化一个 surface bank（按 CPU 线程数并行 + 进度条）。"""
    if n_per_kind <= 0:
        raise ValueError("n_per_kind must be positive.")
    kinds = parse_surface_kinds(surface_kinds)
    root = Path(outdir) if outdir is not None else Path("data") / bank_id
    surfaces_dir = root / "surfaces"
    if root.exists() and not overwrite:
        raise FileExistsError(f"Surface bank already exists: {root}. Pass overwrite=True to replace it.")
    if root.exists() and overwrite:
        _clear_generated_bank(root)
    surfaces_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        _SurfaceGenJob(
            bank_id=bank_id,
            surface_kind=surface_kind,
            index=index,
            seed=stable_surface_seed(bank_id, surface_kind, index, base_seed),
            resolution_cells_per_mm=resolution_cells_per_mm,
            size_x_mm=size_x_mm,
            size_y_mm=size_y_mm,
            tip_radius_mm=tip_radius_mm,
            surfaces_dir=str(surfaces_dir),
        )
        for surface_kind in kinds
        for index in range(n_per_kind)
    ]

    records: list[dict[str, object]] = []
    failed_cases: list[str] = []
    # 表面数组直接由 worker 落盘；主进程只收集小体积统计记录，内存占用恒定。
    with ProgressReporter(len(jobs), label=f"surface_bank {bank_id}") as bar:
        for record, failed_id in map_tasks_unordered(_generate_surface_job, jobs, workers=workers):
            records.append(record)
            if failed_id is not None:
                failed_cases.append(failed_id)
            bar.update()

    # 完成顺序与 worker 无关，按 (kind, surface_id) 排序保证 statistics 表稳定可复现。
    records.sort(key=lambda item: (str(item["surface_kind"]), str(item["surface_id"])))

    pd = _require_pandas_for_bank()
    stats_df = pd.DataFrame.from_records(records)
    write_parquet(stats_df, root / "surface_statistics.parquet")
    write_preview_csv(stats_df, root / "surface_statistics_preview.csv")
    write_schema(root)
    manifest = create_manifest(
        project_name="Spine_Sim_V2",
        model_version="phase2_surface_bank",
        surface_bank_id=bank_id,
        surface_generator_version=SURFACE_GENERATOR_VERSION,
        probe_filter_version=PROBE_FILTER_VERSION,
        random_seed_policy=(
            "stable sha256-derived per-surface seeds from bank_id, kind, index, and base_seed"
        ),
        parameter_grid={
            "surface_kinds": kinds,
            "n_per_kind": n_per_kind,
            "resolution_cells_per_mm": resolution_cells_per_mm,
            "size_x_mm": size_x_mm,
            "size_y_mm": size_y_mm,
            "tip_radius_mm": tip_radius_mm,
            "base_seed": base_seed,
            "workers": resolve_worker_count(workers),
        },
        n_cases_expected=len(kinds) * n_per_kind,
        n_cases_completed=int(stats_df["valid"].sum()),
        failed_cases=failed_cases,
        notes="Phase 2 proxy surface bank. Synthetic roughness parameters are not material truth.",
    )
    write_manifest(manifest, root)
    return SurfaceBank(bank_id=bank_id, root=root)


def _clear_generated_bank(root: Path) -> None:
    """清理已生成的 bank 内容，但不越界触碰父目录。"""
    if root.name in {"", ".", "/"}:
        raise ValueError(f"Refusing to clear unsafe bank root: {root}")
    for child in root.iterdir():
        if child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested_dir in sorted((p for p in child.rglob("*") if p.is_dir()), reverse=True):
                nested_dir.rmdir()
            child.rmdir()
        else:
            child.unlink()


def _require_pandas_for_bank() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Surface bank generation requires pandas and pyarrow for Parquet output. "
            "Install dependencies with `python3 -m pip install -e .`."
        ) from exc
    return pd
