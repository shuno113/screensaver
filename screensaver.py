#!/usr/bin/env python3
"""
Bakery Screensaver - スクリーンセーバー表示ツール

ファイルを視覚的なグレースケールフレームでフルスクリーン再生する。
ストリーミング方式でメモリ使用量を削減し、並列化で高fps再生に対応。

使用方法:
    python screensaver.py -i file.txt
    python screensaver.py -i file.txt --fps 60
"""
import argparse
import hashlib
import queue
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional, BinaryIO

import numpy as np
from PIL import Image

from common import (
    PHYSICAL_WIDTH, PHYSICAL_HEIGHT, BYTES_PER_FRAME, DATA_PIXELS_PER_FRAME,
    LOGICAL_WIDTH, LOGICAL_HEIGHT, REPETITIONS,
    precompute_data_indices,
)

if TYPE_CHECKING:
    import tkinter as tk
    from PIL import ImageTk


# バッファサイズ（フレーム数）
BUFFER_SIZE = 120


# ============================================================
#  ストリーミングヘッダー生成
# ============================================================

def build_header_bytes(file_path: Path, file_size: int, file_hash: bytes) -> bytes:
    """256バイトのヘッダーを生成する。"""
    header = bytearray(256)
    header[0:4] = b"FVD1"
    header[4:8] = (256).to_bytes(4, "big")
    header[8:16] = file_size.to_bytes(8, "big")
    header[16:48] = file_hash
    name = file_path.name.encode("utf-8")[:200]
    header[48] = len(name)
    header[49:49+len(name)] = name
    return bytes(header)


# ============================================================
#  ストリーミングフレーム生成
# ============================================================

def bytes_to_gray_pixels(data: bytes) -> np.ndarray:
    """バイト列をグレースケールピクセル配列に変換する（重複あり）。"""
    if not data:
        return np.zeros((0,), dtype=np.uint8)
    
    arr = np.frombuffer(data, dtype=np.uint8)
    hi = (arr >> 4) & 0x0F
    lo = arr & 0x0F
    
    hi_rep = np.repeat(hi[:, None], REPETITIONS, axis=1)
    lo_rep = np.repeat(lo[:, None], REPETITIONS, axis=1)
    nibbles = np.concatenate([hi_rep, lo_rep], axis=1).reshape(-1)
    
    return np.round(nibbles.astype(np.float32) * 255.0 / 15.0).astype(np.uint8)


def generate_frame_from_gray(gray_pixels: np.ndarray, 
                              data_indices: np.ndarray, 
                              corner_indices: np.ndarray) -> np.ndarray:
    """グレースケールピクセル配列からフレームを生成する。"""
    total_pixels = LOGICAL_HEIGHT * LOGICAL_WIDTH
    frame_flat = np.zeros(total_pixels, dtype=np.uint8)
    
    need = min(len(gray_pixels), DATA_PIXELS_PER_FRAME)
    if need > 0:
        frame_flat[data_indices[:need]] = gray_pixels[:need]
    
    frame_flat[corner_indices] = 255
    return frame_flat.reshape(LOGICAL_HEIGHT, LOGICAL_WIDTH)


class StreamingEncoder:
    """ストリーミング方式でフレームを生成するエンコーダー。"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_size = file_path.stat().st_size
        
        # ファイルハッシュを計算（ストリーミング）
        print(f"    Calculating hash ...")
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        self.file_hash = sha256.digest()
        
        # ヘッダー生成
        self.header = build_header_bytes(file_path, self.file_size, self.file_hash)
        
        # 総データサイズ（ヘッダー + ファイル）
        self.total_size = len(self.header) + self.file_size
        
        # フレーム計算
        self.frame_count = (self.total_size + BYTES_PER_FRAME - 1) // BYTES_PER_FRAME
        
        # インデックス事前計算
        self.data_indices, self.corner_indices = precompute_data_indices()
        
        # ファイルハンドル
        self.file_handle: Optional[BinaryIO] = None
        self.header_offset = 0
        
        print(f"    File size: {self.file_size:,} bytes")
        print(f"    Total frames: {self.frame_count}")
    
    def open(self):
        """ファイルを開く。"""
        self.file_handle = open(self.file_path, 'rb')
        self.header_offset = 0
    
    def close(self):
        """ファイルを閉じる。"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def read_next_chunk(self) -> bytes:
        """次のフレーム用のデータを読み取る。"""
        data = bytearray()
        need = BYTES_PER_FRAME
        
        # まずヘッダーから読む
        if self.header_offset < len(self.header):
            from_header = min(need, len(self.header) - self.header_offset)
            data.extend(self.header[self.header_offset:self.header_offset + from_header])
            self.header_offset += from_header
            need -= from_header
        
        # 残りはファイルから読む
        if need > 0 and self.file_handle:
            chunk = self.file_handle.read(need)
            if chunk:
                data.extend(chunk)
        
        return bytes(data)
    
    def generate_next_frame(self) -> Optional[np.ndarray]:
        """次のフレームを生成する。"""
        data = self.read_next_chunk()
        if not data:
            return None
        
        gray_pixels = bytes_to_gray_pixels(data)
        return generate_frame_from_gray(gray_pixels, self.data_indices, self.corner_indices)


# ============================================================
#  並列フレーム生成
# ============================================================

class FrameProducer(threading.Thread):
    """バックグラウンドでフレームを事前生成するプロデューサースレッド。"""
    
    KEY_FRAME_START_COLOR = (0, 255, 0)
    KEY_FRAME_END_COLOR = (255, 0, 0)
    
    def __init__(self, encoder: StreamingEncoder, fps: int, frame_queue: queue.Queue,
                 screen_w: int, screen_h: int, key_frame_duration_sec: int = 1):
        super().__init__(daemon=True)
        self.encoder = encoder
        self.fps = fps
        self.frame_queue = frame_queue
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.key_frame_count = fps * key_frame_duration_sec
        self._stop_event = threading.Event()
    
    def stop(self):
        """スレッドに停止を要求する。"""
        self._stop_event.set()
    
    def _create_solid_image(self, color: tuple) -> Image.Image:
        """単色ベタ塗りフレームを生成する。"""
        img = Image.new("RGB", (PHYSICAL_WIDTH, PHYSICAL_HEIGHT), color)
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)
        return img
    
    def _create_data_frame(self, frame_gray: np.ndarray) -> Image.Image:
        """データフレームをPIL Imageに変換する。"""
        img = Image.fromarray(np.stack([frame_gray] * 3, axis=-1), mode="RGB")
        img = img.resize((PHYSICAL_WIDTH, PHYSICAL_HEIGHT), resample=Image.NEAREST)
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)
        return img
    
    def run(self):
        """フレームを順次生成してキューに追加する。"""
        try:
            # 開始キーフレーム（緑）
            start_frame = self._create_solid_image(self.KEY_FRAME_START_COLOR)
            for _ in range(self.key_frame_count):
                if self._stop_event.is_set():
                    return
                self.frame_queue.put(start_frame)
            
            # データフレーム（ストリーミング）
            self.encoder.open()
            for _ in range(self.encoder.frame_count):
                if self._stop_event.is_set():
                    self.encoder.close()
                    return
                frame_gray = self.encoder.generate_next_frame()
                if frame_gray is None:
                    break
                img = self._create_data_frame(frame_gray)
                self.frame_queue.put(img)
            self.encoder.close()
            
            # 終了キーフレーム（赤）
            end_frame = self._create_solid_image(self.KEY_FRAME_END_COLOR)
            for _ in range(self.key_frame_count):
                if self._stop_event.is_set():
                    return
                self.frame_queue.put(end_frame)
            
            # 終了シグナル
            self.frame_queue.put(None)
        except Exception as e:
            print(f"Error in FrameProducer: {e}", file=sys.stderr)
            self.frame_queue.put(None)


# ============================================================
#  Tkinter アプリケーション
# ============================================================

class EncoderApp:
    """Tkinter フルスクリーン再生アプリ。"""
    
    def __init__(self, encoder: StreamingEncoder, fps: int):
        import tkinter as tk
        from PIL import ImageTk
        self._tk = tk
        self._ImageTk = ImageTk
        
        self.fps = fps
        self.interval_ms = int(1000 / fps)
        
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.bind("<Escape>", self._on_escape)
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.photo: Optional[ImageTk.PhotoImage] = None
        
        # フレームキューとプロデューサーを初期化
        self.frame_queue: queue.Queue[Optional[Image.Image]] = queue.Queue(maxsize=BUFFER_SIZE)
        self.producer = FrameProducer(
            encoder, fps, self.frame_queue, self.screen_w, self.screen_h)
        self.producer.start()
        
        self.root.after(0, self._show_frame)
    
    def _on_escape(self, event):
        """ESCキー押下時の処理。"""
        self.producer.stop()
        self.root.destroy()
    
    def _show_frame(self):
        """キューからフレームを取得して表示する。"""
        try:
            img = self.frame_queue.get_nowait()
        except queue.Empty:
            self.root.after(1, self._show_frame)
            return
        
        if img is None:
            self.root.after(500, self.root.destroy)
            return
        
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.root.after(self.interval_ms, self._show_frame)

    def run(self):
        self.root.mainloop()
        self.producer.stop()
        self.producer.join(timeout=1.0)


# ============================================================
#  メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="screensaver",
        description="ファイルを視覚的なグレースケールフレームでフルスクリーン再生する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python screensaver.py -i file.txt
  python screensaver.py -i file.txt --fps 60
"""
    )
    parser.add_argument("-i", "--input", required=True, help="入力ファイル")
    parser.add_argument("--fps", type=int, default=10, help="フレームレート (デフォルト: 10)")
    parser.add_argument("--countdown", type=int, default=5, help="開始前カウントダウン秒数 (デフォルト: 5)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Preparing {input_path} ...")
    encoder = StreamingEncoder(input_path)

    # カウントダウン表示
    if args.countdown > 0:
        print(f"[+] Starting in {args.countdown} seconds ...")
        for i in range(args.countdown, 0, -1):
            print(f"    {i}...", flush=True)
            time.sleep(1)

    print(f"[+] Starting playback at {args.fps} fps ...")
    EncoderApp(encoder, args.fps).run()
    print("[+] Playback finished.")


if __name__ == "__main__":
    main()
