#!/usr/bin/env python3
"""
Bakery Decode - デコードツール（最適化版）

動画からファイルを復元する。

最適化:
- シングルパス・ストリーミング処理（メモリ削減・高速化）
- Numba JIT（ホットパス高速化）
- GPU デコード対応 (videotoolbox)

使用方法:
    python decode.py -i output.mp4
    python decode.py -i output.mp4 -d ./output --gpu
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

from common import (
    LOGICAL_WIDTH, LOGICAL_HEIGHT,
    precompute_data_indices, gray_to_nibble,
    parse_inner_header, write_unique_file,
)


# ============================================================
#  定数
# ============================================================

KEY_FRAME_THRESHOLD_HIGH = 200
KEY_FRAME_THRESHOLD_LOW = 50
MIN_CORNER_THRESHOLD = 160.0


# ============================================================
#  Numba JIT 高速化関数
# ============================================================

@jit(nopython=True, cache=True)
def rgb_to_gray_numba(rgb: np.ndarray) -> np.ndarray:
    """RGB配列をグレースケールに変換（Numba版）。"""
    h, w, _ = rgb.shape
    gray = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            gray[y, x] = (float(rgb[y, x, 0]) + float(rgb[y, x, 1]) + float(rgb[y, x, 2])) / 3.0
    return gray


@jit(nopython=True, cache=True, parallel=True)
def resize_block_average_numba(gray: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """ブロック平均によるリサイズ（Numba版）。"""
    src_h, src_w = gray.shape
    block_h = src_h // target_h
    block_w = src_w // target_w
    
    result = np.zeros((target_h, target_w), dtype=np.float32)
    
    for ty in prange(target_h):
        for tx in range(target_w):
            total = 0.0
            for by in range(block_h):
                for bx in range(block_w):
                    total += gray[ty * block_h + by, tx * block_w + bx]
            result[ty, tx] = total / (block_h * block_w)
    
    return result


@jit(nopython=True, cache=True)
def extract_nibbles_numba(gray_corrected: np.ndarray, data_indices: np.ndarray) -> np.ndarray:
    """グレースケール配列からnibbleを抽出（Numba版）。"""
    flat = gray_corrected.ravel()
    nibbles = np.zeros(len(data_indices), dtype=np.uint8)
    for i in range(len(data_indices)):
        val = flat[data_indices[i]]
        nibbles[i] = np.uint8(min(15, max(0, int(round(float(val) * 15.0 / 255.0)))))
    return nibbles


def resize_with_block_average(arr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """ブロック平均によるリサイズ。"""
    if HAS_NUMBA:
        gray = rgb_to_gray_numba(arr)
        return resize_block_average_numba(gray, target_height, target_width)
    else:
        gray = arr.astype(np.float32).mean(axis=2)
        src_h, src_w = gray.shape
        block_h = src_h // target_height
        block_w = src_w // target_width
        reshaped = gray[:target_height * block_h, :target_width * block_w]
        reshaped = reshaped.reshape(target_height, block_h, target_width, block_w)
        return reshaped.mean(axis=(1, 3))


def calibrate_brightness(gray: np.ndarray, corner_indices: np.ndarray) -> np.ndarray:
    """コーナーマーカーを基準に輝度を補正する。"""
    flat = gray.ravel()
    measured_white = flat[corner_indices].max()
    if measured_white < 50:
        return np.clip(gray, 0, 255).astype(np.uint8)
    scale = 255.0 / measured_white
    return np.clip(gray * scale, 0, 255).astype(np.uint8)


# ============================================================
#  ffmpeg ストリーミング
# ============================================================

def get_video_info(video_path: Path) -> Tuple[int, int]:
    """動画の情報を取得する。"""
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path)
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    width, height = map(int, result.stdout.strip().split(','))
    return width, height


# ============================================================
#  シングルパス・ストリーミング処理
# ============================================================

def decode_video_streaming(video_path: Path, output_dir: Path, 
                           repetitions: int = 1, use_gpu: bool = False) -> Path:
    """シングルパスでストリーミングデコード。
    
    - 緑キーフレームを検出するまでスキップ
    - データフレームを処理しながらnibbleを蓄積
    - 赤キーフレームを検出したら終了
    """
    width, height = get_video_info(video_path)
    frame_size = width * height * 3
    
    data_indices, corner_indices = precompute_data_indices()
    nibbles_per_frame = len(data_indices)
    
    # ffmpeg起動
    if use_gpu:
        cmd = [
            "ffmpeg", "-hwaccel", "videotoolbox",
            "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-v", "error", "pipe:1"
        ]
    else:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-v", "error", "pipe:1"
        ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 状態管理
    state = "SEEKING_GREEN"  # SEEKING_GREEN -> TRANSITION -> DATA -> DONE
    frame_idx = 0
    data_start_idx = None
    data_end_idx = None
    
    # nibble蓄積（動的拡張）
    nibbles_chunks: List[np.ndarray] = []
    processed_frames = 0
    first_logged = False
    
    print(f"[+] Streaming decode: {video_path}")
    print(f"    Frame size: {width}x{height}")
    
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            
            # RGB平均値（高速計算）
            r_mean = arr[:, :, 0].mean()
            g_mean = arr[:, :, 1].mean()
            b_mean = arr[:, :, 2].mean()
            
            if state == "SEEKING_GREEN":
                # 緑キーフレーム検出
                if (g_mean > KEY_FRAME_THRESHOLD_HIGH and 
                    r_mean < KEY_FRAME_THRESHOLD_LOW and 
                    b_mean < KEY_FRAME_THRESHOLD_LOW):
                    # 緑を見つけた、次フレームから遷移開始
                    pass
                elif frame_idx > 0:
                    # 緑から離れた = 遷移開始
                    state = "TRANSITION"
            
            elif state == "TRANSITION":
                # 赤キーフレーム検出（早期終了）
                if (r_mean > KEY_FRAME_THRESHOLD_HIGH and 
                    g_mean < KEY_FRAME_THRESHOLD_LOW and 
                    b_mean < KEY_FRAME_THRESHOLD_LOW):
                    data_end_idx = frame_idx
                    state = "DONE"
                    break
                
                # コーナー輝度チェック
                gray = resize_with_block_average(arr, LOGICAL_WIDTH, LOGICAL_HEIGHT)
                corner_min = gray.ravel()[corner_indices].min()
                
                if corner_min >= MIN_CORNER_THRESHOLD:
                    # 安定したデータフレーム開始
                    state = "DATA"
                    data_start_idx = frame_idx
                    print(f"    Data start: frame {frame_idx}")
                    
                    # このフレームも処理
                    gray_corrected = calibrate_brightness(gray, corner_indices)
                    if HAS_NUMBA:
                        nibbles = extract_nibbles_numba(gray_corrected, data_indices)
                    else:
                        nibbles = gray_to_nibble(gray_corrected.ravel()[data_indices])
                    nibbles_chunks.append(nibbles)
                    processed_frames += 1
                    first_logged = True
            
            elif state == "DATA":
                # 赤キーフレーム検出（終了）
                if (r_mean > KEY_FRAME_THRESHOLD_HIGH and 
                    g_mean < KEY_FRAME_THRESHOLD_LOW and 
                    b_mean < KEY_FRAME_THRESHOLD_LOW):
                    data_end_idx = frame_idx
                    state = "DONE"
                    break
                
                # データフレーム処理
                gray = resize_with_block_average(arr, LOGICAL_WIDTH, LOGICAL_HEIGHT)
                corner_min = gray.ravel()[corner_indices].min()
                
                if corner_min < MIN_CORNER_THRESHOLD:
                    # 品質が低いフレームはスキップ
                    frame_idx += 1
                    continue
                
                gray_corrected = calibrate_brightness(gray, corner_indices)
                if HAS_NUMBA:
                    nibbles = extract_nibbles_numba(gray_corrected, data_indices)
                else:
                    nibbles = gray_to_nibble(gray_corrected.ravel()[data_indices])
                nibbles_chunks.append(nibbles)
                processed_frames += 1
                
                # 進捗表示
                if processed_frames % 1000 == 0:
                    print(f"    Processed {processed_frames} frames...")
            
            frame_idx += 1
    
    finally:
        proc.stdout.close()
        proc.stderr.close()
        proc.terminate()
        proc.wait()
    
    if data_start_idx is None:
        raise ValueError("Start key frame not detected")
    
    print(f"    Data end: frame {data_end_idx or frame_idx}")
    print(f"    Total data frames: {processed_frames}")
    
    # nibble結合
    if not nibbles_chunks:
        raise ValueError("No data frames extracted")
    
    all_nibbles = np.concatenate(nibbles_chunks)
    print(f"    Total nibbles: {all_nibbles.size}")
    
    # メモリ解放
    del nibbles_chunks
    
    # バイト復元
    print("[+] Reconstructing data ...")
    group_size = 2 * repetitions
    usable_len = (all_nibbles.size // group_size) * group_size
    if usable_len == 0:
        raise ValueError("No usable data")
    groups = all_nibbles[:usable_len].reshape(-1, group_size)
    hi = np.clip(np.round(groups[:, :repetitions].mean(axis=1) + 0.01), 0, 15).astype(np.uint8)
    lo = np.clip(np.round(groups[:, repetitions:].mean(axis=1) + 0.01), 0, 15).astype(np.uint8)
    data = bytes(((hi << 4) | lo).astype(np.uint8))
    print(f"    Data size: {len(data)} bytes")
    
    # メモリ解放
    del all_nibbles, groups
    
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
        description="動画からファイルを復元する（最適化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python decode.py -i output.mp4
  python decode.py -i output.mp4 --gpu --repetitions 1
"""
    )
    parser.add_argument("-i", "--input", required=True, help="入力動画ファイル")
    parser.add_argument("-d", "--output-dir", default=".", help="出力ディレクトリ")
    parser.add_argument("--repetitions", type=int, default=1, help="nibbleの繰り返し回数 (デフォルト: 1)")
    parser.add_argument("--gpu", action="store_true", help="GPUデコード (macOS videotoolbox)")

    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[+] Config: repetitions={args.repetitions}, gpu={args.gpu}, numba={HAS_NUMBA}")
    decode_video_streaming(video_path, output_dir, args.repetitions, args.gpu)


if __name__ == "__main__":
    main()
