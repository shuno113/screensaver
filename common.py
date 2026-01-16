#!/usr/bin/env python3
"""
Bakery Common - 共通モジュール

screensaver.py と decode.py で共有される設定値、
ヘッダ処理、nibble変換、フレーム処理を提供する。
"""
import hashlib
import math
from pathlib import Path
from typing import Tuple, List

import numpy as np


# ============================================================
#  設定値
# ============================================================

LOGICAL_WIDTH: int = 960
LOGICAL_HEIGHT: int = 540
BLOCK_SIZE: int = 2

PHYSICAL_WIDTH: int = LOGICAL_WIDTH * BLOCK_SIZE   # 1920
PHYSICAL_HEIGHT: int = LOGICAL_HEIGHT * BLOCK_SIZE  # 1080

CORNER_LOGICAL_COORDS: List[Tuple[int, int]] = [
    (0, 0),
    (LOGICAL_WIDTH - 1, 0),
    (0, LOGICAL_HEIGHT - 1),
    (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
]

NIBBLES_PER_BYTE: int = 2
REPETITIONS: int = 2
LOGICAL_PIXELS_PER_BYTE: int = NIBBLES_PER_BYTE * REPETITIONS

DATA_PIXELS_PER_FRAME: int = LOGICAL_WIDTH * LOGICAL_HEIGHT - len(CORNER_LOGICAL_COORDS)
BYTES_PER_FRAME: int = DATA_PIXELS_PER_FRAME // LOGICAL_PIXELS_PER_BYTE

CORNER_WHITE_THRESHOLD: int = 128
GROUP_SIZE: int = 2 * REPETITIONS


# ============================================================
#  ヘッダ処理
# ============================================================

def build_inner_header(file_path: Path, file_bytes: bytes) -> bytes:
    """256 バイトの内側ヘッダを構築する。"""
    header = bytearray(256)
    header[0:4] = b"FVD1"
    header[4:8] = (256).to_bytes(4, "big")
    header[8:16] = len(file_bytes).to_bytes(8, "big")
    header[16:48] = hashlib.sha256(file_bytes).digest()
    name = file_path.name.encode("utf-8")[:200]
    header[48] = len(name)
    header[49:49+len(name)] = name
    return bytes(header)


def build_plaintext_block(file_path: Path) -> bytes:
    """平文ブロック = [内側ヘッダ][元ファイル] を構築する。"""
    data = file_path.read_bytes()
    return build_inner_header(file_path, data) + data


def parse_inner_header(header: bytes) -> Tuple[int, bytes, str]:
    """256 バイトの内側ヘッダを解析する。"""
    if len(header) != 256:
        raise ValueError("Inner header must be exactly 256 bytes")
    if header[0:4] != b"FVD1":
        raise ValueError(f"Invalid magic: {header[0:4]!r}")
    if int.from_bytes(header[4:8], "big") != 256:
        raise ValueError("Invalid header size")
    file_size = int.from_bytes(header[8:16], "big")
    file_hash = header[16:48]
    name_len = header[48]
    if name_len > 200:
        raise ValueError(f"Invalid name_len: {name_len}")
    filename = header[49:49 + name_len].decode("utf-8", errors="replace")
    return file_size, file_hash, filename


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
    corner_indices = np.array([cy * LOGICAL_WIDTH + cx for cx, cy in CORNER_LOGICAL_COORDS], dtype=np.int32)
    mask = np.ones(total, dtype=bool)
    mask[corner_indices] = False
    return all_indices[mask], corner_indices


def generate_frame_array(gray_stream: np.ndarray, offset: int,
                         data_indices: np.ndarray, corner_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    """グレースケールフレーム配列を生成する。"""
    total_pixels = LOGICAL_HEIGHT * LOGICAL_WIDTH
    remain = len(gray_stream) - offset
    need = min(DATA_PIXELS_PER_FRAME, remain) if remain > 0 else 0
    frame_flat = np.zeros(total_pixels, dtype=np.uint8)
    if need > 0:
        frame_flat[data_indices[:need]] = gray_stream[offset:offset + need]
    frame_flat[corner_indices] = 255
    return frame_flat.reshape(LOGICAL_HEIGHT, LOGICAL_WIDTH), offset + need


def calculate_frame_count(data_length: int) -> int:
    """データ長からフレーム数を計算する。"""
    return math.ceil(data_length / BYTES_PER_FRAME)
