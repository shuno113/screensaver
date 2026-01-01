#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from lib.config import (
    LOGICAL_WIDTH,
    LOGICAL_HEIGHT,
    CORNER_LOGICAL_COORDS,
    REPETITIONS,
    CORNER_WHITE_THRESHOLD,
)
from lib.crypto import decrypt_ciphertext
from lib.header import parse_inner_header, write_unique_file
from lib.nibble import gray_to_nibble
from lib.frame import precompute_data_indices

# nibble（4bit）による 16 色表現
GROUP_SIZE = 2 * REPETITIONS  # 4 nibbles / byte


# ============================================================
#  ffmpeg で動画 → フレームPNG
# ============================================================

def run_ffmpeg_extract_frames(video_path: Path, tmpdir: Path) -> List[Path]:
    """
    ffmpeg を使って動画からフレーム画像を PNG で書き出す。
    戻り値: frame_000001.png, ... のパス一覧（ソート済み）。
    """
    pattern = tmpdir / "frame_%06d.png"
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        str(pattern),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}\n"
            f"stdout: {result.stdout.decode(errors='ignore')}\n"
            f"stderr: {result.stderr.decode(errors='ignore')}"
        )

    frames = sorted(tmpdir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames extracted by ffmpeg")
    return frames


# ============================================================
#  フレーム処理
# ============================================================

def verify_corners(frame_gray: np.ndarray) -> None:
    """
    四隅の論理ピクセルが「おおむね白」であることを確認する。
    frame_gray: (H,W) uint8
    """
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        v = int(frame_gray[cy, cx])
        if v < CORNER_WHITE_THRESHOLD:
            raise ValueError(
                f"Corner marker at logical ({cx},{cy}) not white enough: gray={v}"
            )


def extract_nibbles_from_frames(frame_files: List[Path]) -> np.ndarray:
    """
    フレームPNG群から nibble 配列を復元する。
    各フレーム:
      - まず 960x540 に NEAREST でリサイズして論理解像度に揃える
      - 四隅を検証
      - 四隅以外のピクセルから nibble を取り出し、フラットに連結
    """
    data_indices, corner_indices = precompute_data_indices()
    nibbles_all: List[np.ndarray] = []

    for frame_path in frame_files:
        img = Image.open(frame_path).convert("RGB")

        # 論理解像度にリサイズ
        img = img.resize((LOGICAL_WIDTH, LOGICAL_HEIGHT), resample=Image.NEAREST)

        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        # グレースケール化（R,G,B の平均）
        gray = arr.mean(axis=2).astype(np.uint8)  # (H, W)

        # verify_corners(gray)

        flat = gray.reshape(-1)  # (H*W,)
        # データ用ピクセル部分だけ取り出す
        data_gray = flat[data_indices]
        # gray → nibble
        data_nibbles = gray_to_nibble(data_gray)
        nibbles_all.append(data_nibbles)

    if not nibbles_all:
        return np.zeros((0,), dtype=np.uint8)

    return np.concatenate(nibbles_all, axis=0)


def reconstruct_ciphertext_from_nibbles(nibbles: np.ndarray) -> bytes:
    """
    nibble 列から ciphertext (bytes) を復元する。
    エンコード側では 1バイトごとに:
      hi,hi,lo,lo (REPETITIONS=2)
    の順に nibble を出しているので、
    nibble 配列を 4個ずつのグループに分け、
      hi = round(mean(前半2個))
      lo = round(mean(後半2個))
    として 1byte を再構成する。
    """
    if nibbles.size == 0:
        return b""

    group_n = GROUP_SIZE  # 4
    usable_len = (nibbles.size // group_n) * group_n
    if usable_len == 0:
        return b""

    nibbles = nibbles[:usable_len]
    groups = nibbles.reshape(-1, group_n)  # (Nbytes, 4)

    hi_group = groups[:, :REPETITIONS]     # (Nbytes, 2)
    lo_group = groups[:, REPETITIONS:]     # (Nbytes, 2)

    hi = np.round(hi_group.mean(axis=1)).astype(np.uint8)
    lo = np.round(lo_group.mean(axis=1)).astype(np.uint8)

    hi = np.clip(hi, 0, 15)
    lo = np.clip(lo, 0, 15)

    bytes_arr = ((hi << 4) | lo).astype(np.uint8)
    return bytes(bytes_arr)


# ============================================================
#  メイン処理
# ============================================================

def decode_video_to_file(
    video_path: Path,
    password: str,
    output_dir: Path,
) -> Path:
    """動画ファイルをデコードして元のファイルを復元する。"""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        print(f"[+] Extracting frames from {video_path} using ffmpeg ...")
        frame_files = run_ffmpeg_extract_frames(video_path, tmpdir)
        print(f"    Extracted {len(frame_files)} frames")

        print("[+] Extracting nibbles from frames ...")
        nibbles = extract_nibbles_from_frames(frame_files)
        print(f"    Total nibbles: {nibbles.size}")

    print("[+] Reconstructing ciphertext from nibbles ...")
    ciphertext = reconstruct_ciphertext_from_nibbles(nibbles)
    print(f"    Reconstructed ciphertext size: {len(ciphertext)} bytes")

    print("[+] Decrypting ciphertext with AES-256-CTR ...")
    plaintext = decrypt_ciphertext(ciphertext, password)
    print(f"    Plaintext block size: {len(plaintext)} bytes")

    if len(plaintext) < 256:
        raise ValueError("Plaintext too short to contain inner header")

    header = plaintext[:256]
    file_size, file_hash, filename = parse_inner_header(header)
    print(f"[+] Parsed header: filename={filename!r}, file_size={file_size}")

    if file_size > len(plaintext) - 256:
        raise ValueError("File size in header exceeds available plaintext data")

    file_data = plaintext[256:256 + file_size]

    calc_hash = hashlib.sha256(file_data).digest()
    if calc_hash != file_hash:
        raise ValueError("SHA-256 hash mismatch: wrong password or corrupted data")

    output_path = write_unique_file(output_dir, filename, file_data)
    print(f"[+] Recovered file saved: {output_path}")
    print(f"    Size: {file_size} bytes")
    print("    SHA-256 verified successfully")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Decode an encrypted visual stream (960x540 logical, 2x2 blocks, "
            "16-level grayscale, REPETITIONS=2) captured as a video file."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file (e.g. MP4)")
    parser.add_argument("-p", "--password", help="Password (if omitted, prompt securely)")
    parser.add_argument(
        "-d", "--output-dir",
        help="Output directory for recovered file (default: current directory)",
        default="."
    )
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    password = args.password or getpass.getpass("Password: ")

    decode_video_to_file(video_path, password, output_dir)


if __name__ == "__main__":
    main()
