#!/usr/bin/env python3
"""
Bakery Screensaver - スクリーンセーバー表示ツール

ファイルを視覚的なグレースケールフレームでフルスクリーン再生する。

使用方法:
    python screensaver.py -i file.txt
    python screensaver.py -i file.txt --fps 60
"""
import argparse
import queue
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image

from common import (
    PHYSICAL_WIDTH, PHYSICAL_HEIGHT, BYTES_PER_FRAME,
    build_plaintext_block, precompute_data_indices,
    nibbles_from_bytes, nibbles_to_gray, generate_frame_array,
    calculate_frame_count,
)

if TYPE_CHECKING:
    import tkinter as tk
    from PIL import ImageTk


# バッファサイズ（フレーム数）
BUFFER_SIZE = 120


class FrameProducer(threading.Thread):
    """バックグラウンドでフレームを事前生成するプロデューサースレッド。"""
    
    # キーフレーム色 (RGB)
    KEY_FRAME_START_COLOR = (0, 255, 0)   # 緑: 開始
    KEY_FRAME_END_COLOR = (255, 0, 0)     # 赤: 終了
    
    def __init__(self, ciphertext: bytes, fps: int, frame_queue: queue.Queue,
                 screen_w: int, screen_h: int, key_frame_duration_sec: int = 1):
        super().__init__(daemon=True)
        self.fps = fps
        self.gray_stream = nibbles_to_gray(nibbles_from_bytes(ciphertext))
        self.data_indices, self.corner_indices = precompute_data_indices()
        self.frame_count = calculate_frame_count(len(ciphertext))
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
    
    def _create_data_frame(self, offset: int) -> tuple[Image.Image, int]:
        """データフレームを生成する。"""
        frame_gray, new_offset = generate_frame_array(
            self.gray_stream, offset, self.data_indices, self.corner_indices)
        img = Image.fromarray(np.stack([frame_gray] * 3, axis=-1), mode="RGB")
        img = img.resize((PHYSICAL_WIDTH, PHYSICAL_HEIGHT), resample=Image.NEAREST)
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)
        return img, new_offset
    
    def run(self):
        """フレームを順次生成してキューに追加する。"""
        # 開始キーフレーム（緑）
        start_frame = self._create_solid_image(self.KEY_FRAME_START_COLOR)
        for _ in range(self.key_frame_count):
            if self._stop_event.is_set():
                return
            self.frame_queue.put(start_frame)
        
        # データフレーム
        offset = 0
        for _ in range(self.frame_count):
            if self._stop_event.is_set():
                return
            img, offset = self._create_data_frame(offset)
            self.frame_queue.put(img)
        
        # 終了キーフレーム（赤）
        end_frame = self._create_solid_image(self.KEY_FRAME_END_COLOR)
        for _ in range(self.key_frame_count):
            if self._stop_event.is_set():
                return
            self.frame_queue.put(end_frame)
        
        # 終了シグナル
        self.frame_queue.put(None)


class EncoderApp:
    """Tkinter フルスクリーン再生アプリ。"""
    
    def __init__(self, ciphertext: bytes, fps: int):
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
            ciphertext, fps, self.frame_queue, self.screen_w, self.screen_h)
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
            # キューが空の場合、短い待機後に再試行
            self.root.after(1, self._show_frame)
            return
        
        if img is None:
            # 終了シグナル
            self.root.after(500, self.root.destroy)
            return
        
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.root.after(self.interval_ms, self._show_frame)

    def run(self):
        self.root.mainloop()
        # クリーンアップ
        self.producer.stop()
        self.producer.join(timeout=1.0)


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

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[+] Baking {input_path} ...")
    data = build_plaintext_block(input_path)

    EncoderApp(data, args.fps).run()
    print("[+] Baking finished.")


if __name__ == "__main__":
    main()
