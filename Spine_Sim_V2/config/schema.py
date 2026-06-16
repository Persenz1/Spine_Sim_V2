"""仿真落盘数据的 schema 版本与必需字段配置。"""

from __future__ import annotations

SCHEMA_VERSION = "0.1.0"

REQUIRED_CASE_STATUS_FIELDS = (
    "case_status",
    "error_code",
    "warning_flags",
)
