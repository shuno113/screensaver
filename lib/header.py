#!/usr/bin/env python3
"""ヘッダ構築・解析モジュール"""
import hashlib
from pathlib import Path
from typing import Tuple


def build_inner_header(file_path: Path, file_bytes: bytes) -> bytes:
    """256 バイトの内側ヘッダを構築する。"""
    header = bytearray(256)
    # Magic "FVD1"
    header[0:4] = b"FVD1"
    # Header size = 256 (uint32 BE)
    header[4:8] = (256).to_bytes(4, "big")
    # File size (uint64 BE)
    file_size = len(file_bytes)
    header[8:16] = file_size.to_bytes(8, "big")
    # SHA-256 hash
    sha = hashlib.sha256(file_bytes).digest()
    header[16:48] = sha

    # File name (no directory)
    name = file_path.name.encode("utf-8")
    if len(name) > 200:
        name = name[:200]
    header[48] = len(name)  # name_len
    header[49:49+len(name)] = name
    # 残りは 0 のまま
    return bytes(header)


def build_plaintext_block(file_path: Path) -> bytes:
    """平文ブロック = [内側ヘッダ][元ファイル] を構築する。"""
    data = file_path.read_bytes()
    header = build_inner_header(file_path, data)
    plaintext = header + data
    return plaintext


def parse_inner_header(header: bytes) -> Tuple[int, bytes, str]:
    """
    256 バイトの内側ヘッダを解析し、
    (file_size, file_hash, filename) を返す。
    """
    if len(header) != 256:
        raise ValueError("Inner header must be exactly 256 bytes")

    magic = header[0:4]
    if magic != b"FVD1":
        raise ValueError(f"Invalid magic: {magic!r}, expected b'FVD1'")

    header_size = int.from_bytes(header[4:8], "big")
    if header_size != 256:
        raise ValueError(f"Invalid header size: {header_size}, expected 256")

    file_size = int.from_bytes(header[8:16], "big")
    file_hash = header[16:48]

    name_len = header[48]
    if name_len > 200:
        raise ValueError(f"Invalid name_len: {name_len}, must be <= 200")
    name_bytes = header[49:49 + name_len]
    filename = name_bytes.decode("utf-8", errors="replace")

    return file_size, file_hash, filename


def write_unique_file(base_dir: Path, filename: str, data: bytes) -> Path:
    """既存ファイルがある場合は (1), (2) ... を付けて保存する。"""
    target = base_dir / filename
    if not target.exists():
        target.write_bytes(data)
        return target

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            candidate.write_bytes(data)
            return candidate
        counter += 1
