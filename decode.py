#!/usr/bin/env python3
"""
Bakery Decode - デコードツール

動画からファイルを復元する。

使用方法:
    python decode.py -i output.mp4
    python decode.py -i output.mp4 -d ./output
"""
import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from common import (
    LOGICAL_WIDTH, LOGICAL_HEIGHT, GROUP_SIZE, REPETITIONS,
    precompute_data_indices, gray_to_nibble,
    parse_inner_header, write_unique_file,
)


# ============================================================
#  キーフレーム検出
# ============================================================

# キーフレーム色閾値
KEY_FRAME_THRESHOLD_HIGH = 200
KEY_FRAME_THRESHOLD_LOW = 50


def detect_key_frames(frame_files: List[Path]) -> Tuple[Optional[int], Optional[int]]:
    """開始/終了キーフレームのインデックスを検出する。
    
    Returns:
        (start_index, end_index): データ開始/終了フレームのインデックス。
        キーフレーム自体は含まない。検出失敗時はNone。
    """
    start_idx = None
    end_idx = None
    
    for i, frame_path in enumerate(frame_files):
        img = Image.open(frame_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        r_mean = arr[:, :, 0].mean()
        g_mean = arr[:, :, 1].mean()
        b_mean = arr[:, :, 2].mean()
        
        # 緑キーフレーム検出 (開始)
        if (g_mean > KEY_FRAME_THRESHOLD_HIGH and 
            r_mean < KEY_FRAME_THRESHOLD_LOW and 
            b_mean < KEY_FRAME_THRESHOLD_LOW):
            start_idx = i  # 最後の緑フレームを記録
        
        # 赤キーフレーム検出 (終了)
        elif (r_mean > KEY_FRAME_THRESHOLD_HIGH and 
              g_mean < KEY_FRAME_THRESHOLD_LOW and 
              b_mean < KEY_FRAME_THRESHOLD_LOW):
            if start_idx is not None and end_idx is None:
                end_idx = i  # 最初の赤フレームを記録
                break
    
    # データ範囲を返す（キーフレーム自体は除外）
    if start_idx is not None:
        start_idx += 1
    
    return start_idx, end_idx


# ============================================================
#  デコード機能
# ============================================================

def run_ffmpeg_extract_frames(video_path: Path, tmpdir: Path) -> List[Path]:
    """ffmpeg で動画からフレームを抽出する。"""
    cmd = ["ffmpeg", "-i", str(video_path), str(tmpdir / "frame_%06d.png")]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode(errors='ignore')}")
    frames = sorted(tmpdir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames extracted")
    return frames


def extract_nibbles_from_frames(frame_files: List[Path]) -> np.ndarray:
    """フレームから nibble 配列を復元する。"""
    data_indices, _ = precompute_data_indices()
    nibbles_all = []
    for frame_path in frame_files:
        img = Image.open(frame_path).convert("RGB")
        img = img.resize((LOGICAL_WIDTH, LOGICAL_HEIGHT), resample=Image.NEAREST)
        gray = np.array(img, dtype=np.uint8).mean(axis=2).astype(np.uint8)
        nibbles_all.append(gray_to_nibble(gray.reshape(-1)[data_indices]))
    return np.concatenate(nibbles_all) if nibbles_all else np.zeros((0,), dtype=np.uint8)


def reconstruct_ciphertext_from_nibbles(nibbles: np.ndarray) -> bytes:
    """nibble 列から暗号文を復元する。"""
    if nibbles.size == 0:
        return b""
    usable_len = (nibbles.size // GROUP_SIZE) * GROUP_SIZE
    if usable_len == 0:
        return b""
    groups = nibbles[:usable_len].reshape(-1, GROUP_SIZE)
    hi = np.clip(np.round(groups[:, :REPETITIONS].mean(axis=1)), 0, 15).astype(np.uint8)
    lo = np.clip(np.round(groups[:, REPETITIONS:].mean(axis=1)), 0, 15).astype(np.uint8)
    return bytes(((hi << 4) | lo).astype(np.uint8))


def decode_video_to_file(video_path: Path, output_dir: Path) -> Path:
    """動画をデコードしてファイルを復元する。"""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        print(f"[+] Extracting frames from {video_path} ...")
        frame_files = run_ffmpeg_extract_frames(video_path, tmpdir)
        print(f"    Extracted {len(frame_files)} frames")
        
        print("[+] Detecting key frames ...")
        start_idx, end_idx = detect_key_frames(frame_files)
        if start_idx is None or end_idx is None:
            raise ValueError("Key frames not detected. Start or end marker is missing.")
        print(f"    Data frames: {start_idx} to {end_idx - 1} ({end_idx - start_idx} frames)")
        
        # キーフレーム間のみ処理
        data_frames = frame_files[start_idx:end_idx]
        
        print("[+] Extracting nibbles ...")
        nibbles = extract_nibbles_from_frames(data_frames)
        print(f"    Total nibbles: {nibbles.size}")

    print("[+] Reconstructing data ...")
    data = reconstruct_ciphertext_from_nibbles(nibbles)
    print(f"    Data size: {len(data)} bytes")

    if len(data) < 256:
        raise ValueError("Data too short")

    file_size, file_hash, filename = parse_inner_header(data[:256])
    print(f"[+] Header: filename={filename!r}, size={file_size}")
    file_data = data[256:256 + file_size]

    if hashlib.sha256(file_data).digest() != file_hash:
        raise ValueError("SHA-256 hash mismatch")

    output_path = write_unique_file(output_dir, filename, file_data)
    print(f"[+] Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        prog="decode",
        description="動画からファイルを復元する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python decode.py -i output.mp4
  python decode.py -i output.mp4 -d ./recovered
"""
    )
    parser.add_argument("-i", "--input", required=True, help="入力動画ファイル")
    parser.add_argument("-d", "--output-dir", default=".", help="出力ディレクトリ (デフォルト: .)")

    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decode_video_to_file(video_path, output_dir)


if __name__ == "__main__":
    main()
