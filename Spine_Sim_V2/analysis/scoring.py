"""筛选阶段使用的 Q05-Q95 归一化评分工具。"""

from __future__ import annotations

from typing import Any


def score_high(series: Any) -> Any:
    """越大越好的 Q05-Q95 归一化分数。"""
    return _score(series, larger_is_better=True)


def score_low(series: Any) -> Any:
    """越小越好的 Q05-Q95 归一化分数。"""
    return _score(series, larger_is_better=False)


def _score(series: Any, *, larger_is_better: bool) -> Any:
    """执行带裁剪的分位数归一化；退化分布统一给 0.5。"""
    q05 = series.quantile(0.05)
    q95 = series.quantile(0.95)
    if q95 == q05:
        return series * 0.0 + 0.5
    clipped = ((series - q05) / (q95 - q05)).clip(lower=0.0, upper=1.0)
    if larger_is_better:
        return clipped
    return 1.0 - clipped
