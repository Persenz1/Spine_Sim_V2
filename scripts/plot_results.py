#!/usr/bin/env python
"""从已保存数据生成图片。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Spine_Sim_V2.cli import plot_main


if __name__ == "__main__":
    raise SystemExit(plot_main(program="plot_results.py"))
