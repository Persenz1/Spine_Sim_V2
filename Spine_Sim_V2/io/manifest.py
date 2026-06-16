"""Read and write project-level manifest.json files."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Spine_Sim_V2 import __version__
from Spine_Sim_V2.config.schema import SCHEMA_VERSION


@dataclass(frozen=True)
class Manifest:
    """Project metadata saved with every simulation stage directory."""

    project_name: str
    created_time: str
    code_version: str
    model_version: str
    data_schema_version: str
    surface_bank_id: str | None
    surface_generator_version: str | None
    probe_filter_version: str | None
    random_seed_policy: str
    parameter_grid: dict[str, Any]
    n_cases_expected: int
    n_cases_completed: int
    failed_cases: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest dictionary."""
        return asdict(self)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_code_version() -> str:
    """Return the current git commit hash, or ``unknown`` outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def create_manifest(
    *,
    project_name: str = "Spine_Sim_V2",
    model_version: str = "phase1",
    data_schema_version: str = SCHEMA_VERSION,
    surface_bank_id: str | None = None,
    surface_generator_version: str | None = None,
    probe_filter_version: str | None = None,
    random_seed_policy: str = "all stochastic processes must use explicit seeds",
    parameter_grid: dict[str, Any] | None = None,
    n_cases_expected: int = 0,
    n_cases_completed: int = 0,
    failed_cases: list[str] | None = None,
    notes: str = "",
    code_version: str | None = None,
    created_time: str | None = None,
) -> Manifest:
    """Create a manifest with all required fields present."""
    return Manifest(
        project_name=project_name,
        created_time=created_time or utc_now_iso(),
        code_version=code_version or get_code_version() or __version__ or "unknown",
        model_version=model_version,
        data_schema_version=data_schema_version,
        surface_bank_id=surface_bank_id,
        surface_generator_version=surface_generator_version,
        probe_filter_version=probe_filter_version,
        random_seed_policy=random_seed_policy,
        parameter_grid=parameter_grid or {},
        n_cases_expected=n_cases_expected,
        n_cases_completed=n_cases_completed,
        failed_cases=failed_cases or [],
        notes=notes,
    )


def write_manifest(manifest: Manifest | dict[str, Any], path: str | Path) -> Path:
    """Write a manifest to ``path`` or ``path/manifest.json``.

    If ``path`` is a directory, the file name ``manifest.json`` is appended.
    """
    output_path = _manifest_file_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict() if isinstance(manifest, Manifest) else dict(manifest)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def read_manifest(path: str | Path) -> dict[str, Any]:
    """Read a manifest from ``path`` or ``path/manifest.json``."""
    input_path = _manifest_file_path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def _manifest_file_path(path: str | Path) -> Path:
    path_ = Path(path)
    if path_.suffix:
        return path_
    return path_ / "manifest.json"
