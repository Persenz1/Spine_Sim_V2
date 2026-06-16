"""单位换算辅助函数。

长度统一保存为 mm，力统一保存为 N，外部输入/输出角度统一为 deg。
弹簧刚度输入为 N/m，进入内部计算时换算为 N/mm。
"""

from __future__ import annotations

from math import degrees, radians


def spring_k_n_per_m_to_n_per_mm(value_n_per_m: float | None) -> float | None:
    """将弹簧刚度从 N/m 换算为 N/mm。

    刚性阵列的弹簧刚度必须用 ``None``，不能用 ``0``；这里原样透传
    ``None``，避免把刚性误读成零刚度柔顺结构。
    """
    if value_n_per_m is None:
        return None
    return value_n_per_m / 1000.0


def spring_k_n_per_mm_to_n_per_m(value_n_per_mm: float | None) -> float | None:
    """将弹簧刚度从 N/mm 换回 N/m。"""
    if value_n_per_mm is None:
        return None
    return value_n_per_mm * 1000.0


def deg_to_rad(value_deg: float) -> float:
    """将角度从度转换为弧度，供内部计算使用。"""
    return radians(value_deg)


def rad_to_deg(value_rad: float) -> float:
    """将弧度转换为度，供落盘输出使用。"""
    return degrees(value_rad)
