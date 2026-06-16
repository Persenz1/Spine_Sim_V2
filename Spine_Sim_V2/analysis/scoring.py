"""Q05-Q95 scoring helpers for screening stages."""

from __future__ import annotations

from typing import Any


def score_high(series: Any) -> Any:
    """Q05-Q95 normalized score where larger is better."""
    return _score(series, larger_is_better=True)


def score_low(series: Any) -> Any:
    """Q05-Q95 normalized score where smaller is better."""
    return _score(series, larger_is_better=False)


def _score(series: Any, *, larger_is_better: bool) -> Any:
    q05 = series.quantile(0.05)
    q95 = series.quantile(0.95)
    if q95 == q05:
        return series * 0.0 + 0.5
    clipped = ((series - q05) / (q95 - q05)).clip(lower=0.0, upper=1.0)
    if larger_is_better:
        return clipped
    return 1.0 - clipped
