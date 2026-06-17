"""终端原位进度条工具。

设计目标：
- 在交互式终端（TTY）中用回车 ``\r`` 原位刷新单行进度条，而不是不断换行；
- 在非 TTY（如 pytest、重定向到文件）下保持安静，只在结束时打印一行汇总，避免日志刷屏；
- 通过最小刷新间隔节流，避免百万级 case 时频繁写终端拖慢仿真。
"""

from __future__ import annotations

import sys
import time
from typing import Any, TextIO


def _format_seconds(seconds: float) -> str:
    """把秒数格式化为紧凑的 ``MM:SS`` / ``HH:MM:SS``。"""
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:4.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes:02d}:{sec:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{sec:02d}"


class ProgressReporter:
    """原位刷新的进度条。

    用法::

        with ProgressReporter(total, label="P6") as bar:
            for _ in work():
                bar.update()
    """

    def __init__(
        self,
        total: int,
        *,
        label: str = "",
        stream: TextIO | None = None,
        enabled: bool = True,
        min_interval_s: float = 0.2,
        width: int = 28,
    ) -> None:
        self.total = int(total) if total and total > 0 else 0
        self.label = label
        self.stream: TextIO = stream if stream is not None else sys.stderr
        self.count = 0
        self.min_interval_s = float(min_interval_s)
        self.width = int(width)
        self._start = time.monotonic()
        self._last_render = 0.0
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.enabled = bool(enabled) and self.total > 0
        self._finished = False

    def update(self, n: int = 1) -> None:
        """推进进度；按时间节流刷新。"""
        self.count += int(n)
        if not self.enabled:
            return
        now = time.monotonic()
        if self.count >= self.total or (now - self._last_render) >= self.min_interval_s:
            self._render(now)

    def _render(self, now: float) -> None:
        if not (self.enabled and self._is_tty):
            return
        self._last_render = now
        frac = min(1.0, self.count / self.total) if self.total else 1.0
        filled = int(round(self.width * frac))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = now - self._start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.count) / rate if rate > 0 else 0.0
        msg = (
            f"{self.label} [{bar}] {frac * 100:5.1f}% "
            f"({self.count}/{self.total}) "
            f"{_format_seconds(elapsed)}<{_format_seconds(eta)} {rate:6.0f}/s"
        )
        # \r 回到行首原位覆盖，\x1b[K 清除行尾残留字符。
        self.stream.write("\r" + msg + "\x1b[K")
        self.stream.flush()

    def close(self) -> None:
        """收尾：TTY 下补一个换行，非 TTY 下打印一行汇总。"""
        if self._finished:
            return
        self._finished = True
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = now - self._start
        if self._is_tty:
            self._render(now)
            self.stream.write("\n")
        else:
            self.stream.write(
                f"{self.label} done: {self.count}/{self.total} in {_format_seconds(elapsed)}\n"
            )
        self.stream.flush()

    def __enter__(self) -> "ProgressReporter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
