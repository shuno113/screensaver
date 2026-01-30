#!/usr/bin/env python3
"""
Bakery Common - 共通モジュール

screensaver.py と decode.py で共有される設定値、
ヘッダ処理、nibble変換、フレーム処理を提供する。
"""
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np


# ============================================================
#  設定値
# ============================================================

# 物理解像度（固定）
PHYSICAL_WIDTH: int = 1920
PHYSICAL_HEIGHT: int = 1080

# ブロックサイズ（デフォルト: 1）
BLOCK_SIZE: int = 1

# 論理解像度（BLOCK_SIZEに依存、get_logical_dimensions()で取得）
LOGICAL_WIDTH: int = PHYSICAL_WIDTH // BLOCK_SIZE   # 1920 (BLOCK_SIZE=1)
LOGICAL_HEIGHT: int = PHYSICAL_HEIGHT // BLOCK_SIZE  # 1080 (BLOCK_SIZE=1)

# コーナー座標はget_corner_coords()で動的に取得
# （BLOCK_SIZE変更時に再計算が必要なため）

NIBBLES_PER_BYTE: int = 2
REPETITIONS: int = 2
LOGICAL_PIXELS_PER_BYTE: int = NIBBLES_PER_BYTE * REPETITIONS

# DATA_PIXELS_PER_FRAME, BYTES_PER_FRAMEはget_frame_params()で動的に取得
CORNER_WHITE_THRESHOLD: int = 128
GROUP_SIZE: int = 2 * REPETITIONS


def configure_block_size(block_size: int) -> None:
    """ブロックサイズを設定し、依存する値を再計算する。"""
    global BLOCK_SIZE, LOGICAL_WIDTH, LOGICAL_HEIGHT
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if PHYSICAL_WIDTH % block_size != 0 or PHYSICAL_HEIGHT % block_size != 0:
        raise ValueError(f"block_size must evenly divide {PHYSICAL_WIDTH}x{PHYSICAL_HEIGHT}")
    BLOCK_SIZE = block_size
    LOGICAL_WIDTH = PHYSICAL_WIDTH // block_size
    LOGICAL_HEIGHT = PHYSICAL_HEIGHT // block_size


def get_logical_dimensions() -> Tuple[int, int]:
    """現在のBLOCK_SIZEに基づいた論理解像度を取得する。"""
    return LOGICAL_WIDTH, LOGICAL_HEIGHT


def get_physical_dimensions() -> Tuple[int, int]:
    """物理解像度を取得する（固定値）。"""
    return PHYSICAL_WIDTH, PHYSICAL_HEIGHT


def get_corner_coords() -> List[Tuple[int, int]]:
    """現在の論理解像度に基づいたコーナー座標を取得する。"""
    return [
        (0, 0),
        (LOGICAL_WIDTH - 1, 0),
        (0, LOGICAL_HEIGHT - 1),
        (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
    ]


def get_frame_params() -> Tuple[int, int]:
    """現在の論理解像度に基づいたDATA_PIXELS_PER_FRAME, BYTES_PER_FRAMEを取得する。"""
    data_pixels = LOGICAL_WIDTH * LOGICAL_HEIGHT - 4  # 4 corners
    bytes_per_frame = data_pixels // LOGICAL_PIXELS_PER_BYTE
    return data_pixels, bytes_per_frame


# ============================================================
#  ヘッダ処理
# ============================================================

# ヘッダフォーマット FVD2 (256バイト):
#   0-3:     マジック "FVD2"
#   4-7:     ヘッダサイズ (256)
#   8-15:    ファイルサイズ (8バイト, big-endian)
#   16-47:   SHA-256ハッシュ (32バイト)
#   48:      ファイル名長 (最大180)
#   49-228:  ファイル名 (UTF-8, 180バイト)
#   229-232: データフレーム数 (4バイト, big-endian)
#   233-236: nibble数 (4バイト, big-endian)
#   237:     Repetitions (1バイト)
#   238-239: Block size (2バイト, big-endian)
#   240-241: FPS (2バイト, big-endian)
#   242-255: 予約 (ゼロ埋め)

MAGIC_FVD2 = b"FVD2"
HEADER_SIZE = 256
MAX_FILENAME_LEN = 180


@dataclass
class HeaderInfo:
    """ヘッダ情報を格納するデータクラス。"""
    file_size: int
    file_hash: bytes
    filename: str
    frame_count: int
    nibble_count: int
    repetitions: int
    block_size: int
    fps: int


def build_inner_header(
    file_path: Path,
    file_bytes: bytes,
    frame_count: int,
    nibble_count: int,
    repetitions: int,
    block_size: int,
    fps: int,
) -> bytes:
    """256バイトのFVD2ヘッダを構築する。"""
    header = bytearray(HEADER_SIZE)
    
    # マジック & ヘッダサイズ
    header[0:4] = MAGIC_FVD2
    header[4:8] = HEADER_SIZE.to_bytes(4, "big")
    
    # ファイル情報
    header[8:16] = len(file_bytes).to_bytes(8, "big")
    header[16:48] = hashlib.sha256(file_bytes).digest()
    
    # ファイル名 (最大180バイト)
    name = file_path.name.encode("utf-8")[:MAX_FILENAME_LEN]
    header[48] = len(name)
    header[49:49 + len(name)] = name
    
    # エンコーディングパラメータ
    header[229:233] = frame_count.to_bytes(4, "big")
    header[233:237] = nibble_count.to_bytes(4, "big")
    header[237] = repetitions
    header[238:240] = block_size.to_bytes(2, "big")
    header[240:242] = fps.to_bytes(2, "big")
    
    return bytes(header)


def parse_inner_header(header: bytes) -> HeaderInfo:
    """256バイトのFVD2ヘッダを解析する。"""
    if len(header) != HEADER_SIZE:
        raise ValueError(f"Inner header must be exactly {HEADER_SIZE} bytes")
    if header[0:4] != MAGIC_FVD2:
        raise ValueError(f"Invalid magic: {header[0:4]!r} (expected {MAGIC_FVD2!r})")
    if int.from_bytes(header[4:8], "big") != HEADER_SIZE:
        raise ValueError("Invalid header size")
    
    # ファイル情報
    file_size = int.from_bytes(header[8:16], "big")
    file_hash = header[16:48]
    
    # ファイル名
    name_len = header[48]
    if name_len > MAX_FILENAME_LEN:
        raise ValueError(f"Invalid name_len: {name_len}")
    filename = header[49:49 + name_len].decode("utf-8", errors="replace")
    
    # エンコーディングパラメータ
    frame_count = int.from_bytes(header[229:233], "big")
    nibble_count = int.from_bytes(header[233:237], "big")
    repetitions = header[237]
    block_size = int.from_bytes(header[238:240], "big")
    fps = int.from_bytes(header[240:242], "big")
    
    return HeaderInfo(
        file_size=file_size,
        file_hash=file_hash,
        filename=filename,
        frame_count=frame_count,
        nibble_count=nibble_count,
        repetitions=repetitions,
        block_size=block_size,
        fps=fps,
    )


def write_unique_file(base_dir: Path, filename: str, data: bytes) -> Path:
    """既存ファイルがある場合は連番を付けて保存する。"""
    target = base_dir / filename
    if not target.exists():
        target.write_bytes(data)
        return target
    stem, suffix = Path(filename).stem, Path(filename).suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            candidate.write_bytes(data)
            return candidate
        counter += 1


# ============================================================
#  nibble変換
# ============================================================

def nibble_to_gray(nibble: int) -> int:
    """0..15 の nibble を 0..255 のグレースケールにマップする。"""
    return round(max(0, min(15, nibble)) * 255 / 15)


def gray_to_nibble(gray: np.ndarray) -> np.ndarray:
    """グレースケール配列を nibble 配列に変換する。"""
    return np.clip(np.round(gray.astype(np.float32) * 15.0 / 255.0), 0, 15).astype(np.uint8)


def nibbles_from_bytes(ciphertext: bytes) -> np.ndarray:
    """ciphertext から nibble 配列を生成する。"""
    ct = np.frombuffer(ciphertext, dtype=np.uint8)
    hi = (ct >> 4) & 0x0F
    lo = ct & 0x0F
    hi_rep = np.repeat(hi[:, None], REPETITIONS, axis=1)
    lo_rep = np.repeat(lo[:, None], REPETITIONS, axis=1)
    return np.concatenate([hi_rep, lo_rep], axis=1).reshape(-1).astype(np.uint8)


def nibbles_to_gray(nibbles: np.ndarray) -> np.ndarray:
    """nibble 配列をグレースケール配列に変換する。"""
    return np.round(nibbles.astype(np.float32) * 255.0 / 15.0).astype(np.uint8)


# ============================================================
#  フレーム処理
# ============================================================

def precompute_data_indices() -> Tuple[np.ndarray, np.ndarray]:
    """データ格納用インデックスと四隅インデックスを返す。"""
    total = LOGICAL_HEIGHT * LOGICAL_WIDTH
    all_indices = np.arange(total, dtype=np.int32)
    corner_coords = get_corner_coords()
    corner_indices = np.array([cy * LOGICAL_WIDTH + cx for cx, cy in corner_coords], dtype=np.int32)
    mask = np.ones(total, dtype=bool)
    mask[corner_indices] = False
    return all_indices[mask], corner_indices


def generate_frame_array(gray_stream: np.ndarray, offset: int,
                         data_indices: np.ndarray, corner_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    """グレースケールフレーム配列を生成する。"""
    data_pixels_per_frame, _ = get_frame_params()
    total_pixels = LOGICAL_HEIGHT * LOGICAL_WIDTH
    remain = len(gray_stream) - offset
    need = min(data_pixels_per_frame, remain) if remain > 0 else 0
    frame_flat = np.zeros(total_pixels, dtype=np.uint8)
    if need > 0:
        frame_flat[data_indices[:need]] = gray_stream[offset:offset + need]
    frame_flat[corner_indices] = 255
    return frame_flat.reshape(LOGICAL_HEIGHT, LOGICAL_WIDTH), offset + need


def calculate_frame_count(data_length: int) -> int:
    """データ長からフレーム数を計算する。"""
    _, bytes_per_frame = get_frame_params()
    return math.ceil(data_length / bytes_per_frame)
