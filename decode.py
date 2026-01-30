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
    CORNER_WHITE_THRESHOLD,
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
        キーフレーム自体および遷移フレームは含まない。検出失敗時はNone。
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
        
        # 遷移フレームをスキップ：コーナーが安定して明るく、かつ内容が安定しているフレームを探す
        data_indices, corner_indices = precompute_data_indices()
        
        def get_frame_info(idx):
            img = Image.open(frame_files[idx]).convert("RGB")
            gray = resize_with_block_average(img, LOGICAL_WIDTH, LOGICAL_HEIGHT)
            flat = gray.reshape(-1)
            corner_min = flat[corner_indices].min()
            sample = gray[10:50, 10:50].tobytes()
            return corner_min, sample
        
        while start_idx < (end_idx or len(frame_files)) - 1:
            curr_corner_min, curr_sample = get_frame_info(start_idx)
            next_corner_min, next_sample = get_frame_info(start_idx + 1)
            
            # コーナーが明るく（>=160）、かつ次のフレームと内容が同じなら安定
            if curr_corner_min >= 160 and curr_sample == next_sample:
                break
            start_idx += 1
    
    return start_idx, end_idx


# ============================================================
#  位置補正
# ============================================================

CORNER_SEARCH_RANGE = 20  # コーナー探索範囲（ピクセル）


def detect_corner_offset(gray: np.ndarray) -> Tuple[int, int]:
    """左上コーナーマーカー（白ピクセル）を検出し、オフセットを返す。
    
    Args:
        gray: グレースケール画像配列 (height, width)
    
    Returns:
        (dx, dy): 検出されたオフセット。補正不要または検出失敗時は (0, 0)
    """
    # 左上付近で最も白いピクセルを探索
    search_region = gray[:CORNER_SEARCH_RANGE, :CORNER_SEARCH_RANGE]
    max_val = search_region.max()
    
    if max_val < CORNER_WHITE_THRESHOLD:
        return 0, 0  # 白ピクセルが見つからない
    
    max_pos = np.unravel_index(search_region.argmax(), search_region.shape)
    dy, dx = max_pos  # numpy の shape は (row, col) = (y, x)
    
    return dx, dy


def apply_offset_correction(gray: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """オフセット補正を適用する。"""
    if dx == 0 and dy == 0:
        return gray
    
    # 負方向にロール（左上に移動）
    return np.roll(np.roll(gray, -dy, axis=0), -dx, axis=1)


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


def resize_with_block_average(img: Image.Image, target_width: int, target_height: int) -> np.ndarray:
    """ブロック平均によるリサイズ。
    
    Image.NEAREST はブロック内の1ピクセルのみを選択するため、
    OBSのフレームキャプチャタイミングずれで値がばらつく問題を回避する。
    
    Args:
        img: 入力画像 (RGB)
        target_width: 目標幅
        target_height: 目標高さ
    
    Returns:
        グレースケール配列 (target_height, target_width) as float32
    """
    arr = np.array(img, dtype=np.float32)
    gray = arr.mean(axis=2)  # RGB to grayscale
    
    src_height, src_width = gray.shape
    block_h = src_height // target_height
    block_w = src_width // target_width
    
    # ブロック単位で平均化
    reshaped = gray[:target_height * block_h, :target_width * block_w]
    reshaped = reshaped.reshape(target_height, block_h, target_width, block_w)
    result = reshaped.mean(axis=(1, 3))
    
    return result  # float32のまま返す


def calibrate_brightness(gray: np.ndarray, corner_indices: np.ndarray) -> np.ndarray:
    """コーナーマーカーを基準に輝度を補正する。
    
    コーナーは本来255（白）であるべき。最も明るいコーナーを基準に
    スケーリング係数を計算し、全体に適用する。
    
    Args:
        gray: グレースケール配列 (height, width) as float32
        corner_indices: コーナーピクセルのインデックス配列
    
    Returns:
        補正後のグレースケール配列 (uint8)
    """
    flat = gray.reshape(-1)
    corner_values = flat[corner_indices]
    
    # 最も明るいコーナーを白レベルとして使用
    # フレーム選択により、少なくとも1つのコーナーは255に近いはず
    measured_white = corner_values.max()
    
    if measured_white < 50:
        # コーナーが暗すぎる場合は補正を諦める
        return np.clip(gray, 0, 255).astype(np.uint8)
    
    # スケーリング係数を計算
    scale = 255.0 / measured_white
    
    # 補正を適用
    corrected = gray * scale
    
    return np.clip(corrected, 0, 255).astype(np.uint8)


def get_frame_corner_quality(img: Image.Image, corner_indices: np.ndarray) -> Tuple[float, np.ndarray]:
    """フレームのコーナー品質（最小コーナー値）を計算する。
    
    Returns:
        (最小コーナー値, リサイズ後のグレースケール配列)
    """
    gray = resize_with_block_average(img, LOGICAL_WIDTH, LOGICAL_HEIGHT)
    flat = gray.reshape(-1)
    corner_values = flat[corner_indices]
    min_corner = corner_values.min()
    return min_corner, gray


def select_best_frames(frame_files: List[Path], corner_indices: np.ndarray,
                       min_corner_threshold: float = 180.0,
                       similarity_threshold: float = 0.95) -> List[Tuple[Path, np.ndarray]]:
    """録画フレームから最もクリーンなフレームを選択する。
    
    フレーム内容の類似性に基づいてグループ化し、各グループから最もクリーンなフレームを選択する。
    
    Args:
        frame_files: フレームファイルのリスト
        corner_indices: コーナーピクセルのインデックス
        min_corner_threshold: 使用するフレームの最小コーナー値閾値
        similarity_threshold: 同一フレームと判定する相関係数の閾値
    
    Returns:
        選択されたフレームとそのグレースケール配列のリスト
    """
    if not frame_files:
        return []
    
    # 全フレームを読み込んでコーナー品質を計算
    frame_data = []  # (path, gray, min_corner, data_hash)
    for path in frame_files:
        img = Image.open(path).convert("RGB")
        quality, gray = get_frame_corner_quality(img, corner_indices)
        # データ部分のハッシュ（高速な類似性判定用）
        data_sample = gray[10:50, 10:50].tobytes()  # 中央部分のサンプル
        frame_data.append((path, gray, quality, data_sample))
    
    # 類似フレームをグループ化し、各グループのベストを選択
    selected = []
    i = 0
    n = len(frame_data)
    
    while i < n:
        # 現在のフレームと類似しているフレームを集める
        group_start = i
        current_sample = frame_data[i][3]
        
        j = i + 1
        while j < n:
            # サンプルが同じなら同じ論理フレーム
            if frame_data[j][3] == current_sample:
                j += 1
            else:
                break
        
        group_end = j
        
        # グループ内で最も品質が高いフレームを選択
        best_idx = group_start
        best_quality = frame_data[group_start][2]
        
        for k in range(group_start + 1, group_end):
            if frame_data[k][2] > best_quality:
                best_quality = frame_data[k][2]
                best_idx = k
        
        # 閾値以上なら選択
        if best_quality >= min_corner_threshold:
            path, gray, _, _ = frame_data[best_idx]
            selected.append((path, gray))
        
        i = group_end
    
    return selected


def extract_nibbles_from_frames(frame_files: List[Path]) -> np.ndarray:
    """フレームから nibble 配列を復元する（フレーム選択＋ブロック平均＋輝度補正）。
    
    録画FPSがスクリーンセーバーFPSより高い場合、最もクリーンなフレームを選択して使用する。
    """
    data_indices, corner_indices = precompute_data_indices()
    
    # フレーム選択（内容ベースのグループ化で最もクリーンなフレームを選択）
    selected_frames = select_best_frames(frame_files, corner_indices, min_corner_threshold=180.0)
    
    if not selected_frames:
        print("    Warning: No frames passed quality threshold!")
        return np.zeros((0,), dtype=np.uint8)
    
    print(f"    Selected {len(selected_frames)} clean frames from {len(frame_files)} total")
    
    nibbles_all = []
    scale_logged = False
    
    for frame_path, gray in selected_frames:
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
