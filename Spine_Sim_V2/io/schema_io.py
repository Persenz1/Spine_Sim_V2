"""Read and write schema.json files."""

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
    """Build the JSON document saved as schema.json."""
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
    """Write schema metadata to ``path`` or ``path/schema.json``."""
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


def read_schema(path: str | Path) -> dict[str, Any]:
    """Read schema metadata from ``path`` or ``path/schema.json``."""
    input_path = _schema_file_path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _schema_file_path(path: str | Path) -> Path:
    path_ = Path(path)
    if path_.suffix:
        return path_
    return path_ / "schema.json"
