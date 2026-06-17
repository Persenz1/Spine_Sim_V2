"""按 CPU 线程数动态调整的任务并行工具。

仿真主链条是 CPU 密集的纯 Python（逐刺搜索/加载），受 GIL 限制，多线程无法真正并行，
因此用 ``ProcessPoolExecutor`` 跨进程并行。为应对未来更大规模仿真：

- ``resolve_worker_count`` 在未显式指定时取 ``os.cpu_count()``，自动吃满核数；
- ``map_tasks_unordered`` 维持有界的在途任务数（``max_pending``），边完成边产出结果，
  使主进程可以即时落盘，**不把全部结果堆在内存里**；
- ``workers <= 1`` 时退化为顺序执行，便于调试，也避免小规模运行的进程开销。
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def _safe_start_context() -> mp.context.BaseContext:
    """选择不受父进程多线程影响的进程启动方式。

    主进程常已加载 numpy/openblas 等多线程库，直接 ``fork`` 可能在子进程死锁
    （Python 3.12 起也会告警）。优先 ``forkserver``（从干净的服务进程派生），
    其次 ``spawn``，最后回退默认。入口脚本均有 ``__main__`` 守卫，spawn 安全。
    """
    available = set(mp.get_all_start_methods())
    for method in ("forkserver", "spawn", "fork"):
        if method in available:
            return mp.get_context(method)
    return mp.get_context()


def resolve_worker_count(workers: int | None) -> int:
    """把 ``workers`` 解析为实际进程数。

    ``None`` 或 ``<=0`` 表示自动按 CPU 线程数选择；显式正整数原样使用。
    """
    if workers is None or workers <= 0:
        return max(1, os.cpu_count() or 1)
    return int(workers)


def map_tasks_unordered(
    func: Callable[[T], R],
    tasks: Iterable[T],
    *,
    workers: int | None,
    max_pending: int | None = None,
) -> Iterator[R]:
    """并行执行 ``func(task)``，按完成顺序产出结果。

    ``func`` 与 ``task`` 必须可被 pickle（模块级函数 + dataclass 参数）。
    """
    n_workers = resolve_worker_count(workers)
    if n_workers <= 1:
        for task in tasks:
            yield func(task)
        return

    pending_cap = max_pending if max_pending and max_pending > 0 else n_workers * 4
    iterator = iter(tasks)
    pending: set[Future[R]] = set()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=_safe_start_context()) as executor:
        for _ in range(pending_cap):
            try:
                pending.add(executor.submit(func, next(iterator)))
            except StopIteration:
                break
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            # 每完成一个就补一个，保持在途任务数恒定，内存占用有界。
            for _ in range(len(done)):
                try:
                    pending.add(executor.submit(func, next(iterator)))
                except StopIteration:
                    break
