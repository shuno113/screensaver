#!/usr/bin/env python3
"""フレーム処理モジュール"""
from typing import Tuple

import numpy as np

from lib.config import LOGICAL_WIDTH, LOGICAL_HEIGHT, CORNER_LOGICAL_COORDS


def precompute_data_indices() -> Tuple[np.ndarray, np.ndarray]:
    """
    全論理ピクセルのフラットインデックスを作り、
    四隅を除いた「データ格納用インデックス」と、
    四隅のインデックスを返す。
    """
    h, w = LOGICAL_HEIGHT, LOGICAL_WIDTH
    total = h * w
    all_indices = np.arange(total, dtype=np.int32)

    corner_indices = []
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        idx = cy * LOGICAL_WIDTH + cx
        corner_indices.append(idx)
    corner_indices = np.array(corner_indices, dtype=np.int32)

    mask = np.ones(total, dtype=bool)
    mask[corner_indices] = False
    data_indices = all_indices[mask]

    return data_indices, corner_indices
