#!/usr/bin/env python3
"""
Bakery Screensaver - スクリーンセーバー表示ツール

ファイルを視覚的なグレースケールフレームでフルスクリーン再生する。

使用方法:
    python screensaver.py -i file.txt
    python screensaver.py -i file.txt --fps 15
"""
import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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
        self.frame_count = calculate_frame_count(len(ciphertext))
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


def main():
    parser = argparse.ArgumentParser(
        prog="screensaver",
        description="ファイルを視覚的なグレースケールフレームでフルスクリーン再生する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python screensaver.py -i file.txt
  python screensaver.py -i file.txt --fps 15
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
