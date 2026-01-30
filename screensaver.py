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
    
    # キーフレーム色 (RGB)
    KEY_FRAME_START_COLOR = (0, 255, 0)   # 緑: 開始
    KEY_FRAME_END_COLOR = (255, 0, 0)     # 赤: 終了
    KEY_FRAME_DURATION_SEC = 1            # キーフレーム表示時間（秒）
    
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
        self.current_data_frame = 0
        
        # キーフレーム用
        self.key_frame_count = fps * self.KEY_FRAME_DURATION_SEC
        self.current_key_frame = 0
        self.phase = "start_key"  # "start_key" -> "data" -> "end_key"

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.photo = None
        self.root.after(0, self._show_frame)
    
    def _create_solid_image(self, color: tuple) -> Image.Image:
        """単色ベタ塗りフレームを生成する。"""
        img = Image.new("RGB", (PHYSICAL_WIDTH, PHYSICAL_HEIGHT), color)
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)
        return img
    
    def _show_frame(self):
        """フェーズに応じてフレームを表示する。"""
        if self.phase == "start_key":
            self._show_key_frame(self.KEY_FRAME_START_COLOR, next_phase="data")
        elif self.phase == "data":
            self._show_data_frame()
        elif self.phase == "end_key":
            self._show_key_frame(self.KEY_FRAME_END_COLOR, next_phase="done")
        else:
            self.root.after(500, self.root.destroy)
    
    def _show_key_frame(self, color: tuple, next_phase: str):
        """キーフレームを表示する。"""
        img = self._create_solid_image(color)
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.current_key_frame += 1
        if self.current_key_frame >= self.key_frame_count:
            self.phase = next_phase
            self.current_key_frame = 0
        self.root.after(self.interval_ms, self._show_frame)
    
    def _show_data_frame(self):
        """データフレームを表示する。"""
        if self.current_data_frame >= self.frame_count:
            self.phase = "end_key"
            self.root.after(0, self._show_frame)
            return
        frame_gray, self.offset = generate_frame_array(
            self.gray_stream, self.offset, self.data_indices, self.corner_indices)
        img = Image.fromarray(np.stack([frame_gray] * 3, axis=-1), mode="RGB")
        img = img.resize((PHYSICAL_WIDTH, PHYSICAL_HEIGHT), resample=Image.NEAREST)
        if self.screen_w != PHYSICAL_WIDTH or self.screen_h != PHYSICAL_HEIGHT:
            img = img.resize((self.screen_w, self.screen_h), resample=Image.NEAREST)
        self.photo = self._ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)
        self.current_data_frame += 1
        self.root.after(self.interval_ms, self._show_frame)

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
