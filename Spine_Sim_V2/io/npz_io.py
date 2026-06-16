"""NPZ 数组读写辅助函数。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_npz_arrays(path: str | Path, **arrays: object) -> Path:
    """将命名数组保存为压缩 NPZ 文件。"""
    if not arrays:
        raise ValueError("At least one named array is required.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def load_npz_arrays(path: str | Path) -> dict[str, np.ndarray]:
    """将 NPZ 文件中的全部数组读入内存。"""
    with np.load(Path(path), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}
