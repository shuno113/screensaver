#!/usr/bin/env python3
import argparse
import getpass
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from lib.config import (
    LOGICAL_WIDTH,
    LOGICAL_HEIGHT,
    PHYSICAL_WIDTH,
    PHYSICAL_HEIGHT,
    DATA_PIXELS_PER_FRAME,
    BYTES_PER_FRAME,
)
from lib.crypto import encrypt_plaintext
from lib.header import build_plaintext_block
from lib.nibble import nibbles_from_bytes, nibbles_to_gray
from lib.frame import precompute_data_indices


# ============================================================
#  フレーム生成（numpyベース）
# ============================================================

def generate_frame_array(gray_stream: np.ndarray,
                         offset: int,
                         data_indices: np.ndarray,
                         corner_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    gray_stream から offset 以降の値を使い、
    1フレーム分の論理解像度グレースケール画像 (H,W) を numpy で生成する。
    戻り値: (frame_gray(H,W), new_offset)
    """
    h, w = LOGICAL_HEIGHT, LOGICAL_WIDTH
    total_pixels = h * w

    # このフレームで使う data ピクセル数
    remain = len(gray_stream) - offset
    need = min(DATA_PIXELS_PER_FRAME, remain) if remain > 0 else 0

    frame_flat = np.zeros(total_pixels, dtype=np.uint8)

    if need > 0:
        frame_flat[data_indices[:need]] = gray_stream[offset: offset + need]

    # 四隅は白(255)にしてマーカー
    frame_flat[corner_indices] = 255

    frame_gray = frame_flat.reshape(h, w)
    return frame_gray, offset + need


# ============================================================
#  Tkinter フルスクリーンアプリ
# ============================================================

class EncoderApp:
    def __init__(self, ciphertext: bytes, fps: int):
        self.ciphertext = ciphertext
        self.fps = fps
        self.interval_ms = int(1000 / fps)

        # 前処理：ciphertext → gray_stream
        nibbles = nibbles_from_bytes(ciphertext)
        self.gray_stream = nibbles_to_gray(nibbles)

        self.data_indices, self.corner_indices = precompute_data_indices()

        total_bytes = len(ciphertext)
        self.frame_count = math.ceil(total_bytes / BYTES_PER_FRAME)
        print(f"[+] bytes={total_bytes}, bytes/frame={BYTES_PER_FRAME}, frames={self.frame_count}, fps={self.fps}")
        print(f"[+] nibbles={len(nibbles)}, gray_stream={len(self.gray_stream)}")

        self.offset = 0
        self.current_frame = 0

        # Tkinter セットアップ
        self.root = tk.Tk()
        self.root.title("Encrypted Visual Stream (Tkinter, 960x540 logical, 2x2 blocks, REP=2)")

        # フルスクリーン & 最前面
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)

        # ESC で終了
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        # 画面解像度に合わせてスケーリング
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()

        # 画像表示用ラベル
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)

        self.photo = None  # 参照保持用

        # 最初のフレームをスケジュール
        self.root.after(0, self.show_next_frame)

    def show_next_frame(self):
        if self.current_frame >= self.frame_count:
            # 少し待ってから終了
            self.root.after(500, self.root.destroy)
            return

        # 1フレーム分のグレースケール論理画像を生成
        frame_gray, self.offset = generate_frame_array(
            self.gray_stream, self.offset, self.data_indices, self.corner_indices
        )

        # (H,W) → (H,W,3) RGB
        frame_rgb = np.stack([frame_gray] * 3, axis=-1)

        # numpy → PIL.Image
        img = Image.fromarray(frame_rgb, mode="RGB")

        # 物理解像度 (1920x1080) に 2x2 拡大
        img = img.resize((PHYSICAL_WIDTH, PHYSICAL_HEIGHT), resample=Image.NEAREST)

        # さらに実際のスクリーン解像度にフィットさせたい場合はここでリサイズ（任意）
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)

        self.photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)

        self.current_frame += 1
        # 次フレーム
        self.root.after(self.interval_ms, self.show_next_frame)

    def run(self):
        self.root.mainloop()


# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fullscreen encoder (Tkinter, 960x540 logical, 2x2 blocks, 16-level grayscale, REPETITIONS=2, numpy-optimized)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-p", "--password", help="Password (if omitted, prompt securely)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    password = args.password or getpass.getpass("Password: ")

    print(f"[+] Building plaintext block from {input_path} ...")
    plaintext = build_plaintext_block(input_path)
    print(f"    Plaintext size: {len(plaintext)} bytes")

    print("[+] Encrypting plaintext with AES-256-CTR ...")
    ciphertext = encrypt_plaintext(plaintext, password)
    print(f"    Ciphertext size: {len(ciphertext)} bytes")

    app = EncoderApp(ciphertext, args.fps)
    app.run()
    print("[+] Playback finished. Exiting.")


if __name__ == "__main__":
    main()
