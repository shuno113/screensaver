#!/usr/bin/env python3
"""
Transporter - Visual Data Transfer System

ファイルを視覚的なグレースケールフレームの動画に変換し、
エアギャップ環境からのデータ転送を可能にするシステム。

使用方法:
    python transporter.py play -i file.txt
    python transporter.py decode -i output.mp4 -d ./output
"""
import argparse
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple, List, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import tkinter as tk
    from PIL import ImageTk


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
        print("[+] Extracting nibbles ...")
        nibbles = extract_nibbles_from_frames(frame_files)
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


# ============================================================
#  スクリーンセーバー機能
# ============================================================

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


class EncoderApp:
    """Tkinter フルスクリーン再生アプリ。"""
    def __init__(self, ciphertext: bytes, fps: int):
        import tkinter as tk
        from PIL import ImageTk
        self._tk = tk
        self._ImageTk = ImageTk
        
        self.fps = fps
        self.interval_ms = int(1000 / fps)
        self.gray_stream = nibbles_to_gray(nibbles_from_bytes(ciphertext))
        self.data_indices, self.corner_indices = precompute_data_indices()
        self.frame_count = math.ceil(len(ciphertext) / BYTES_PER_FRAME)
        self.offset = 0
        self.current_frame = 0

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.photo = None
        self.root.after(0, self.show_next_frame)

    def show_next_frame(self):
        if self.current_frame >= self.frame_count:
            self.root.after(500, self.root.destroy)
            return
        frame_gray, self.offset = generate_frame_array(
            self.gray_stream, self.offset, self.data_indices, self.corner_indices)
        img = Image.fromarray(np.stack([frame_gray] * 3, axis=-1), mode="RGB")
        img = img.resize((PHYSICAL_WIDTH, PHYSICAL_HEIGHT), resample=Image.NEAREST)
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.current_frame += 1
        self.root.after(self.interval_ms, self.show_next_frame)

    def run(self):
        self.root.mainloop()


# ============================================================
#  CLI コマンド
# ============================================================


def cmd_decode(args):
    """decode コマンド"""
    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decode_video_to_file(video_path, output_dir)


def cmd_play(args):
    """play コマンド"""
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[+] Building data from {input_path} ...")
    data = build_plaintext_block(input_path)

    EncoderApp(data, args.fps).run()
    print("[+] Playback finished.")


def main():
    parser = argparse.ArgumentParser(
        prog="transporter",
        description="Visual Data Transfer System - ファイルを視覚的なデータストリームに変換",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python transporter.py play -i file.txt --fps 15
  python transporter.py decode -i output.mp4 -d ./recovered
"""
    )
    subparsers = parser.add_subparsers(dest="command", help="コマンド")

    # play
    p_play = subparsers.add_parser("play", help="フルスクリーン再生")
    p_play.add_argument("-i", "--input", required=True, help="入力ファイル")
    p_play.add_argument("--fps", type=int, default=10, help="フレームレート")

    # decode
    p_dec = subparsers.add_parser("decode", help="MP4動画からファイルを復元")
    p_dec.add_argument("-i", "--input", required=True, help="入力動画ファイル")
    p_dec.add_argument("-d", "--output-dir", default=".", help="出力ディレクトリ")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    {"decode": cmd_decode, "play": cmd_play}[args.command](args)


if __name__ == "__main__":
    main()
