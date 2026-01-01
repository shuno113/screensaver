#!/usr/bin/env python3
"""共通設定モジュール"""
from typing import Tuple, List

# ============================================================
#  解像度設定
# ============================================================

# 論理解像度（2x2 ブロック単位）
LOGICAL_WIDTH: int = 960
LOGICAL_HEIGHT: int = 540
BLOCK_SIZE: int = 2

# 物理解像度（動画の実サイズ）
PHYSICAL_WIDTH: int = LOGICAL_WIDTH * BLOCK_SIZE   # 1920
PHYSICAL_HEIGHT: int = LOGICAL_HEIGHT * BLOCK_SIZE  # 1080

# 四隅の予約論理ピクセル（2x2 ブロック単位で白固定）
CORNER_LOGICAL_COORDS: List[Tuple[int, int]] = [
    (0, 0),
    (LOGICAL_WIDTH - 1, 0),
    (0, LOGICAL_HEIGHT - 1),
    (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
]

# ============================================================
#  nibble 設定（16色 → 4bit）
# ============================================================

NIBBLES_PER_BYTE: int = 2  # hi, lo
REPETITIONS: int = 2      # 各 nibble を2回繰り返す
LOGICAL_PIXELS_PER_BYTE: int = NIBBLES_PER_BYTE * REPETITIONS  # 4

# フレームあたりのデータ量
DATA_PIXELS_PER_FRAME: int = LOGICAL_WIDTH * LOGICAL_HEIGHT - len(CORNER_LOGICAL_COORDS)
BYTES_PER_FRAME: int = DATA_PIXELS_PER_FRAME // LOGICAL_PIXELS_PER_BYTE

# ============================================================
#  PBKDF2 / AES 設定
# ============================================================

PBKDF2_SALT: bytes = b"FV-ENC-1"
PBKDF2_ITERATIONS: int = 200_000
PBKDF2_OUTPUT_LEN: int = 48  # 32 bytes key + 16 bytes IV

# ============================================================
#  デコード設定
# ============================================================

# コーナーマーカー（白）のしきい値
CORNER_WHITE_THRESHOLD: int = 128
