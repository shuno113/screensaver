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
import io
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

from common import (
    LOGICAL_WIDTH, LOGICAL_HEIGHT, GROUP_SIZE, REPETITIONS,
    PHYSICAL_WIDTH, PHYSICAL_HEIGHT,
    CORNER_WHITE_THRESHOLD,
    precompute_data_indices, gray_to_nibble,
    parse_inner_header, write_unique_file,
)


# ============================================================
#  定数
# ============================================================

# キーフレーム色閾値
KEY_FRAME_THRESHOLD_HIGH = 200
KEY_FRAME_THRESHOLD_LOW = 50

# コーナー探索範囲（ピクセル）
CORNER_SEARCH_RANGE = 20

# フレーム選択閾値
MIN_CORNER_THRESHOLD = 180.0

# 並列処理のワーカー数（0 = CPUコア数）
MAX_WORKERS = 0


# ============================================================
#  ユーティリティ
# ============================================================

def get_worker_count() -> int:
    """並列処理のワーカー数を取得する。"""
    if MAX_WORKERS > 0:
        return MAX_WORKERS
    return os.cpu_count() or 4


# ============================================================
#  ffmpeg パイプ処理
# ============================================================

def run_ffmpeg_pipe_frames(video_path: Path) -> Tuple[List[np.ndarray], int, int]:
    """ffmpegからパイプで直接フレームを読み込む。
    
    Returns:
        (フレームリスト, width, height)
    """
    # まず動画情報を取得
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path)
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    if probe_result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe_result.stderr}")
    
    width, height = map(int, probe_result.stdout.strip().split(','))
    frame_size = width * height * 3  # RGB24
    
    # ffmpegでrawvideoとして出力
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-v", "error",
        "pipe:1"
    ]
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        frames.append(arr)
    
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode(errors='ignore')
        raise RuntimeError(f"ffmpeg failed: {stderr}")
    
    return frames, width, height


# ============================================================
#  キーフレーム検出
# ============================================================

def detect_key_frames_from_arrays(frames: List[np.ndarray]) -> Tuple[Optional[int], Optional[int]]:
    """開始/終了キーフレームのインデックスを検出する（配列版）。
    
    Returns:
        (start_index, end_index): データ開始/終了フレームのインデックス。
        キーフレーム自体および遷移フレームは含まない。検出失敗時はNone。
    """
    start_idx = None
    end_idx = None
    
    for i, arr in enumerate(frames):
        arr_f = arr.astype(np.float32)
        r_mean = arr_f[:, :, 0].mean()
        g_mean = arr_f[:, :, 1].mean()
        b_mean = arr_f[:, :, 2].mean()
        
        # 緑キーフレーム検出 (開始)
        if (g_mean > KEY_FRAME_THRESHOLD_HIGH and 
            r_mean < KEY_FRAME_THRESHOLD_LOW and 
            b_mean < KEY_FRAME_THRESHOLD_LOW):
            start_idx = i  # 最後の緑フレームを記録
        
        # 赤キーフレーム検出 (終了) - 検出したら即終了
        elif (r_mean > KEY_FRAME_THRESHOLD_HIGH and 
              g_mean < KEY_FRAME_THRESHOLD_LOW and 
              b_mean < KEY_FRAME_THRESHOLD_LOW):
            if start_idx is not None and end_idx is None:
                end_idx = i  # 最初の赤フレームを記録
                break  # 早期終了
    
    # データ範囲を返す（キーフレーム自体は除外）
    if start_idx is not None:
        start_idx += 1
        
        # 遷移フレームをスキップ：コーナーが安定して明るく、かつ内容が安定しているフレームを探す
        data_indices, corner_indices = precompute_data_indices()
        
        def get_frame_info(idx):
            gray = resize_with_block_average_from_array(frames[idx], LOGICAL_WIDTH, LOGICAL_HEIGHT)
            flat = gray.reshape(-1)
            corner_min = flat[corner_indices].min()
            sample = gray[10:50, 10:50].tobytes()
            return corner_min, sample
        
        while start_idx < (end_idx or len(frames)) - 1:
            curr_corner_min, curr_sample = get_frame_info(start_idx)
            next_corner_min, next_sample = get_frame_info(start_idx + 1)
            
            # コーナーが明るく（>=160）、かつ次のフレームと内容が同じなら安定
            if curr_corner_min >= 160 and curr_sample == next_sample:
                break
            start_idx += 1
    
    return start_idx, end_idx


# ============================================================
#  リサイズ・輝度補正
# ============================================================

def resize_with_block_average_from_array(arr: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """ブロック平均によるリサイズ（配列版）。
    
    Args:
        arr: 入力画像配列 (height, width, 3) RGB
        target_width: 目標幅
        target_height: 目標高さ
    
    Returns:
        グレースケール配列 (target_height, target_width) as float32
    """
    gray = arr.astype(np.float32).mean(axis=2)  # RGB to grayscale
    
    src_height, src_width = gray.shape
    block_h = src_height // target_height
    block_w = src_width // target_width
    
    # ブロック単位で平均化
    reshaped = gray[:target_height * block_h, :target_width * block_w]
    reshaped = reshaped.reshape(target_height, block_h, target_width, block_w)
    result = reshaped.mean(axis=(1, 3))
    
    return result  # float32のまま返す


def calibrate_brightness(gray: np.ndarray, corner_indices: np.ndarray) -> np.ndarray:
    """コーナーマーカーを基準に輝度を補正する。"""
    flat = gray.reshape(-1)
    corner_values = flat[corner_indices]
    
    measured_white = corner_values.max()
    
    if measured_white < 50:
        return np.clip(gray, 0, 255).astype(np.uint8)
    
    scale = 255.0 / measured_white
    corrected = gray * scale
    
    return np.clip(corrected, 0, 255).astype(np.uint8)


# ============================================================
#  並列フレーム処理
# ============================================================

def _process_single_frame(args: Tuple[int, np.ndarray, np.ndarray]) -> Tuple[int, float, np.ndarray, bytes]:
    """1フレームを処理する（ワーカー関数）。
    
    Args:
        args: (インデックス, フレーム配列, コーナーインデックス)
    
    Returns:
        (インデックス, 最小コーナー値, グレースケール配列, サンプルハッシュ)
    """
    idx, arr, corner_indices = args
    gray = resize_with_block_average_from_array(arr, LOGICAL_WIDTH, LOGICAL_HEIGHT)
    flat = gray.reshape(-1)
    corner_values = flat[corner_indices]
    min_corner = corner_values.min()
    data_sample = gray[10:50, 10:50].tobytes()
    return idx, min_corner, gray, data_sample


def select_best_frames_parallel(frames: List[np.ndarray], corner_indices: np.ndarray,
                                 min_corner_threshold: float = MIN_CORNER_THRESHOLD) -> List[Tuple[int, np.ndarray]]:
    """録画フレームから最もクリーンなフレームを選択する（並列版）。
    
    Args:
        frames: フレーム配列のリスト
        corner_indices: コーナーピクセルのインデックス
        min_corner_threshold: 使用するフレームの最小コーナー値閾値
    
    Returns:
        選択されたフレームインデックスとグレースケール配列のリスト
    """
    if not frames:
        return []
    
    n_workers = get_worker_count()
    
    # 並列で全フレームを処理
    frame_data = [None] * len(frames)  # (idx, quality, gray, sample)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # ジョブを投入
        futures = {}
        for i, arr in enumerate(frames):
            future = executor.submit(_process_single_frame, (i, arr, corner_indices))
            futures[future] = i
        
        # 結果を収集
        for future in as_completed(futures):
            idx, quality, gray, sample = future.result()
            frame_data[idx] = (idx, quality, gray, sample)
    
    # 類似フレームをグループ化し、各グループのベストを選択
    selected = []
    i = 0
    n = len(frame_data)
    
    while i < n:
        group_start = i
        current_sample = frame_data[i][3]
        
        j = i + 1
        while j < n:
            if frame_data[j][3] == current_sample:
                j += 1
            else:
                break
        
        group_end = j
        
        # グループ内で最も品質が高いフレームを選択
        best_idx = group_start
        best_quality = frame_data[group_start][1]
        
        for k in range(group_start + 1, group_end):
            if frame_data[k][1] > best_quality:
                best_quality = frame_data[k][1]
                best_idx = k
        
        # 閾値以上なら選択
        if best_quality >= min_corner_threshold:
            idx, _, gray, _ = frame_data[best_idx]
            selected.append((idx, gray))
        
        i = group_end
    
    return selected


def extract_nibbles_from_arrays(frames: List[np.ndarray]) -> np.ndarray:
    """フレーム配列から nibble 配列を復元する（並列版）。"""
    data_indices, corner_indices = precompute_data_indices()
    
    # フレーム選択（並列処理）
    selected_frames = select_best_frames_parallel(frames, corner_indices, min_corner_threshold=MIN_CORNER_THRESHOLD)
    
    if not selected_frames:
        print("    Warning: No frames passed quality threshold!")
        return np.zeros((0,), dtype=np.uint8)
    
    print(f"    Selected {len(selected_frames)} clean frames from {len(frames)} total")
    
    nibbles_all = []
    scale_logged = False
    
    for idx, gray in selected_frames:
        # 輝度補正
        gray_corrected = calibrate_brightness(gray, corner_indices)
        
        if not scale_logged:
            flat = gray.reshape(-1)
            corner_values = flat[corner_indices]
            min_corner = corner_values.min()
            mean_corner = corner_values.mean()
            print(f"    First clean frame: min_corner = {min_corner:.1f}, mean = {mean_corner:.1f}")
            if mean_corner < 255:
                print(f"    Brightness calibration: scale = {255.0/mean_corner:.3f}")
            scale_logged = True
        
        nibbles_all.append(gray_to_nibble(gray_corrected.reshape(-1)[data_indices]))
    
    return np.concatenate(nibbles_all) if nibbles_all else np.zeros((0,), dtype=np.uint8)


# ============================================================
#  メイン処理
# ============================================================

def reconstruct_ciphertext_from_nibbles(nibbles: np.ndarray) -> bytes:
    """nibble 列から暗号文を復元する。"""
    if nibbles.size == 0:
        return b""
    usable_len = (nibbles.size // GROUP_SIZE) * GROUP_SIZE
    if usable_len == 0:
        return b""
    groups = nibbles[:usable_len].reshape(-1, GROUP_SIZE)
    # 平均値 + 0.01 で銀行家丸め（0.5 → 0）を回避し、0.5 → 1 にする
    hi = np.clip(np.round(groups[:, :REPETITIONS].mean(axis=1) + 0.01), 0, 15).astype(np.uint8)
    lo = np.clip(np.round(groups[:, REPETITIONS:].mean(axis=1) + 0.01), 0, 15).astype(np.uint8)
    return bytes(((hi << 4) | lo).astype(np.uint8))


def decode_video_to_file(video_path: Path, output_dir: Path) -> Path:
    """動画をデコードしてファイルを復元する。"""
    print(f"[+] Extracting frames from {video_path} ...")
    frames, width, height = run_ffmpeg_pipe_frames(video_path)
    print(f"    Extracted {len(frames)} frames ({width}x{height})")
    
    print("[+] Detecting key frames ...")
    start_idx, end_idx = detect_key_frames_from_arrays(frames)
    if start_idx is None or end_idx is None:
        raise ValueError("Key frames not detected. Start or end marker is missing.")
    print(f"    Data frames: {start_idx} to {end_idx - 1} ({end_idx - start_idx} frames)")
    
    # キーフレーム間のみ処理
    data_frames = frames[start_idx:end_idx]
    
    # メモリ解放（不要なフレームを削除）
    del frames
    
    print("[+] Extracting nibbles ...")
    nibbles = extract_nibbles_from_arrays(data_frames)
    print(f"    Total nibbles: {nibbles.size}")
    
    # メモリ解放
    del data_frames

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
