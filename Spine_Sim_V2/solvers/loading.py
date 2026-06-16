"""切向位移控制加载与事件驱动承载上限求解。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LoadingResult:
    """单个 case 的整体加载结果。"""

    f_t_lim_n: float
    limit_displacement_mm: float | None
    load_at_limit_n: NDArray[np.float64]
    failed: NDArray[np.bool_]
    failure_mode: list[str | None]
    failure_order: list[int | None]
    cascade_failure: bool
    event_displacements_mm: tuple[float, ...] = ()
    event_total_force_n: tuple[float, ...] = ()


def run_loading_sequence(
    *,
    engaged: NDArray[np.bool_],
    search_distance_mm: NDArray[np.floating],
    cap_n: NDArray[np.floating],
    k_tt_n_per_mm: float | None,
) -> LoadingResult:
    """执行简化的位移控制事件序列。

    已接合刺在搜索距离 ``X_i`` 之后开始承载，载荷按
    ``k_share * (s - X_i)`` 增长；达到容量的刺在事件点移除。刚性阵列使用
    数值共享刚度，因为第一版不把刚性切向弹性作为真实物理参数标定。
    """
    engaged_arr = np.asarray(engaged, dtype=bool)
    search = np.asarray(search_distance_mm, dtype=float)
    caps = np.asarray(cap_n, dtype=float)
    n = len(caps)
    k_share = float(k_tt_n_per_mm) if k_tt_n_per_mm is not None and k_tt_n_per_mm > 0.0 else 1.0
    loads_at_best = np.zeros(n, dtype=float)
    failed = np.zeros(n, dtype=bool)
    failure_mode: list[str | None] = [None] * n
    failure_order: list[int | None] = [None] * n
    event_order = 0
    cascade_failure = False

    remaining = engaged_arr & np.isfinite(search) & np.isfinite(caps) & (caps > 0.0)
    if not np.any(remaining):
        return LoadingResult(0.0, None, loads_at_best, failed, failure_mode, failure_order, False)

    best_total = -1.0
    best_s: float | None = None
    event_displacements: list[float] = []
    event_forces: list[float] = []
    event_tol = 1e-10
    while np.any(remaining):
        # 每轮直接计算所有剩余刺的失效位移，整体极限载荷只从事件点产生。
        s_fail = np.full(n, np.inf, dtype=float)
        s_fail[remaining] = search[remaining] + caps[remaining] / k_share
        event_s = float(np.min(s_fail))
        if not np.isfinite(event_s):
            break

        loads = np.zeros(n, dtype=float)
        active = remaining & (event_s >= search)
        loads[active] = k_share * (event_s - search[active])
        loads = np.minimum(loads, caps)
        total = float(np.sum(loads))
        event_displacements.append(event_s)
        event_forces.append(total)
        if total > best_total:
            best_total = total
            best_s = float(event_s)
            loads_at_best = loads.copy()

        failing_now = np.where(remaining & (np.abs(s_fail - event_s) <= event_tol))[0]
        if len(failing_now) > 1:
            # 同一位移多个刺同时失效，记录为潜在级联失效。
            cascade_failure = True
        for idx in sorted(int(item) for item in failing_now):
            if not failed[idx]:
                event_order += 1
                failed[idx] = True
                failure_mode[idx] = "overload"
                failure_order[idx] = event_order
                remaining[idx] = False

    return LoadingResult(
        f_t_lim_n=float(max(best_total, 0.0)),
        limit_displacement_mm=best_s,
        load_at_limit_n=loads_at_best,
        failed=failed,
        failure_mode=failure_mode,
        failure_order=failure_order,
        cascade_failure=cascade_failure,
        event_displacements_mm=tuple(event_displacements),
        event_total_force_n=tuple(event_forces),
    )
