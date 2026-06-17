"""Parquet 权威表和 CSV 预览表的读写辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from Spine_Sim_V2.core.types import SchemaField


def write_parquet(df: Any, path: str | Path) -> Path:
    """使用 pyarrow 将 DataFrame 写为 Parquet。"""
    _require_parquet_dependencies()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)
    return output_path


def read_parquet(path: str | Path) -> Any:
    """读取 Parquet 文件为 pandas DataFrame。"""
    pd = _require_parquet_dependencies()
    return pd.read_parquet(Path(path), engine="pyarrow")


def write_preview_csv(df: Any, path: str | Path, max_rows: int = 5000) -> Path:
    """写出小规模 CSV 预览表。

    CSV 只供人工快速查看；仿真数据产品的权威表格式仍是 Parquet。
    """
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = df.head(max_rows) if hasattr(df, "head") else df
    preview.to_csv(output_path, index=False)
    return output_path


def _require_parquet_dependencies() -> Any:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet IO requires pandas and pyarrow. Install project "
            "dependencies with `python3 -m pip install -e .`."
        ) from exc

    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet IO requires pyarrow. Install project dependencies with "
            "`python3 -m pip install -e .`."
        ) from exc

    return pd


class BatchedParquetWriter:
    """按固定 schema 累积记录字典，达到 ``batch_size`` 后作为一个 row group 落盘。

    用途与设计：
    - 大规模阶段（P5/P6）逐 case 产出记录，本类只在内存里保留**一个批次**，
      满批即写一个 Parquet row group，写完即释放，避免把全部 case 堆在内存里导致内存爆炸；
    - 用批次（而非逐行）写 row group，避免出现成千上万个 1 行 row group 拖慢回读；
    - 强制 arrow schema，使刚性阵列的 ``spring_k_n_per_m`` 等字段保持 null 语义、
      失败 case 的可空字段也能正常写入。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        schema: Sequence[SchemaField],
        batch_size: int = 2000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._schema_fields = tuple(schema)
        self.batch_size = int(batch_size)
        self._batch: list[dict[str, Any]] = []
        self._writer: Any = None
        self._arrow_schema: Any = None
        self._has_written = False

    def add_record(self, record: dict[str, Any]) -> None:
        """追加一条记录；满批自动落盘。"""
        self._batch.append(record)
        if len(self._batch) >= self.batch_size:
            self._flush()

    def add_records(self, records: Iterable[dict[str, Any]]) -> None:
        """批量追加多条记录。"""
        for record in records:
            self.add_record(record)

    def _flush(self) -> None:
        if not self._batch:
            return
        pd = _require_parquet_dependencies()
        import pyarrow as pa
        import pyarrow.parquet as pq

        frame = pd.DataFrame.from_records(self._batch)
        normalized = _normalize_for_arrow_schema(frame, self._schema_fields)
        if self._arrow_schema is None:
            self._arrow_schema = _arrow_schema(self._schema_fields)
        table = pa.Table.from_pandas(normalized, schema=self._arrow_schema, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        self._has_written = True
        self._batch = []

    def close(self) -> None:
        """落盘剩余批次；若一条都没有，则写出一个带 schema 的空表。"""
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif not self._has_written:
            schema = self._arrow_schema or _arrow_schema(self._schema_fields)
            arrays = [pa.array([], type=field.type) for field in schema]
            pq.write_table(pa.Table.from_arrays(arrays, schema=schema), self.path)
            self._has_written = True

    def __enter__(self) -> "BatchedParquetWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _normalize_for_arrow_schema(df: Any, schema: Sequence[SchemaField]) -> Any:
    pd = _require_parquet_dependencies()
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
        if value != value:  # NaN
            return []
    except ValueError:
        pass
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _arrow_schema(schema: Sequence[SchemaField]) -> Any:
    import pyarrow as pa

    # 物理写盘 schema 一律放宽为可空：失败 / 部分完成的 case 行可能在任意字段留空，
    # 不能因为非空约束让整轮仿真崩溃。字段的"应然可空性"仍由 schema.json 记录。
    return pa.schema(
        [pa.field(field.name, _arrow_type(field.dtype), nullable=True) for field in schema]
    )


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
