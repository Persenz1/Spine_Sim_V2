"""读写 ``schema.json`` 表结构说明文件。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from Spine_Sim_V2.config.schema import SCHEMA_VERSION
from Spine_Sim_V2.core.types import SCHEMA_REGISTRY, SchemaField, schema_to_dicts


SchemaLike = tuple[SchemaField, ...] | list[SchemaField]
SchemaCollection = Mapping[str, SchemaLike]


def build_schema_document(
    schemas: SchemaCollection | None = None,
    *,
    data_schema_version: str = SCHEMA_VERSION,
) -> dict[str, Any]:
    """构造要保存到 ``schema.json`` 的 JSON 文档。"""
    selected_schemas = schemas or SCHEMA_REGISTRY
    return {
        "data_schema_version": data_schema_version,
        "schemas": {
            name: schema_to_dicts(tuple(fields))
            for name, fields in selected_schemas.items()
        },
    }


def write_schema(
    path: str | Path,
    schemas: SchemaCollection | None = None,
    *,
    data_schema_version: str = SCHEMA_VERSION,
) -> Path:
    """将 schema 元数据写到指定文件或目录下的 ``schema.json``。"""
    output_path = _schema_file_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_schema_document(
        schemas=schemas,
        data_schema_version=data_schema_version,
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def dataframe_schema(df: Any) -> tuple[SchemaField, ...]:
    """从实际 DataFrame 列推断轻量 schema，用于分析阶段的派生表。"""
    return tuple(
        SchemaField(
            name=str(column),
            dtype=_schema_dtype(df[column]),
            unit=_unit_from_field_name(str(column)),
            nullable=bool(df[column].isna().any()) if len(df) else True,
            description=str(column).replace("_", " "),
        )
        for column in df.columns
    )


def read_schema(path: str | Path) -> dict[str, Any]:
    """从指定文件或目录下的 ``schema.json`` 读取 schema 元数据。"""
    input_path = _schema_file_path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _schema_file_path(path: str | Path) -> Path:
    path_ = Path(path)
    if path_.suffix:
        return path_
    return path_ / "schema.json"


def _schema_dtype(series: Any) -> str:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - schema inference is pandas-backed.
        raise RuntimeError("DataFrame schema inference requires pandas.") from exc

    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "float64"
    if series.apply(lambda value: isinstance(value, (list, tuple))).any():
        return "list[string]"
    return "string"


def _unit_from_field_name(name: str) -> str | None:
    if name.endswith("_n_per_m"):
        return "N/m"
    if name.endswith("_n_per_mm") or name.endswith("_n_per_mm2"):
        return "N/mm^2" if name.endswith("_n_per_mm2") else "N/mm"
    if name.endswith("_mm"):
        return "mm"
    if name.endswith("_n") or "_n_" in name:
        return "N"
    if name.endswith("_deg"):
        return "deg"
    return None
