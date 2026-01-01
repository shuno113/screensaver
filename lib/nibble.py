#!/usr/bin/env python3
"""nibble変換モジュール"""
import numpy as np

from lib.config import REPETITIONS


def nibble_to_gray(nibble: int) -> int:
    """
    0..15 の nibble を 0..255 のグレースケールにマップする。
    16段階の等間隔レベル。
    """
    nibble = max(0, min(15, nibble))
    return round(nibble * 255 / 15)


def gray_to_nibble(gray: np.ndarray) -> np.ndarray:
    """
    0..255 のグレースケール配列を 0..15 の nibble 配列に変換する。
    nibble = round(gray * 15 / 255)
    """
    gray_f = gray.astype(np.float32)
    level = np.round(gray_f * (15.0 / 255.0)).astype(np.int16)
    level = np.clip(level, 0, 15)
    return level.astype(np.uint8)


def nibbles_from_bytes(ciphertext: bytes) -> np.ndarray:
    """
    ciphertext (uint8) から hi/lo nibble を取り出し、
    hi, lo を REPETITIONS 回ずつ繰り返した nibble 配列を返す。
    REPETITIONS=2 の場合、1 byte→[hi,hi,lo,lo]（4 nibbles）。
    """
    ct = np.frombuffer(ciphertext, dtype=np.uint8)
    hi = (ct >> 4) & 0x0F
    lo = ct & 0x0F

    hi_rep = np.repeat(hi[:, None], REPETITIONS, axis=1)  # (N, REP)
    lo_rep = np.repeat(lo[:, None], REPETITIONS, axis=1)  # (N, REP)

    nibbles = np.concatenate([hi_rep, lo_rep], axis=1).reshape(-1)
    return nibbles.astype(np.uint8)


def nibbles_to_gray(nibbles: np.ndarray) -> np.ndarray:
    """
    nibble(0..15) の配列を 0..255 グレースケール配列に変換する。
    gray = round(n * 255 / 15)
    """
    n = nibbles.astype(np.float32)
    gray = np.round(n * (255.0 / 15.0)).astype(np.uint8)
    return gray
