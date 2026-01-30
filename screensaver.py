#!/usr/bin/env python3
"""
Bakery Screensaver - スクリーンセーバー表示ツール

ファイルを視覚的なグレースケールフレームでフルスクリーン再生する。
ストリーミング方式でメモリ使用量を削減し、pygameで高速描画。

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
from typing import Optional, BinaryIO

import numpy as np
import pygame

from common import (
    PHYSICAL_WIDTH, PHYSICAL_HEIGHT, REPETITIONS,
    precompute_data_indices, configure_block_size, get_frame_params,
    get_physical_dimensions,
)


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

def bytes_to_gray_pixels(data: bytes, repetitions: int = 1) -> np.ndarray:
    """バイト列をグレースケールピクセル配列に変換する（重複あり）。"""
    if not data:
        return np.zeros((0,), dtype=np.uint8)
    
    arr = np.frombuffer(data, dtype=np.uint8)
    hi = (arr >> 4) & 0x0F
    lo = arr & 0x0F
    
    hi_rep = np.repeat(hi[:, None], repetitions, axis=1)
    lo_rep = np.repeat(lo[:, None], repetitions, axis=1)
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
    
    def __init__(self, file_path: Path, repetitions: int = 1):
        self.file_path = file_path
        self.file_size = file_path.stat().st_size
        self.repetitions = repetitions
        
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
        
        # 動的なバイト/フレーム計算（repetitionsに依存）
        pixels_per_byte = 2 * self.repetitions  # 2 nibbles × repetitions
        self.bytes_per_frame = DATA_PIXELS_PER_FRAME // pixels_per_byte
        
        # フレーム計算
        self.frame_count = (self.total_size + self.bytes_per_frame - 1) // self.bytes_per_frame
        
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
        need = self.bytes_per_frame
        
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
        
        gray_pixels = bytes_to_gray_pixels(data, self.repetitions)
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
    
    def _create_solid_array(self, color: tuple) -> np.ndarray:
        """単色ベタ塗りフレームを生成する（RGB配列）。"""
        arr = np.zeros((LOGICAL_HEIGHT, LOGICAL_WIDTH, 3), dtype=np.uint8)
        arr[:, :, 0] = color[0]
        arr[:, :, 1] = color[1]
        arr[:, :, 2] = color[2]
        return arr
    
    def _create_data_array(self, frame_gray: np.ndarray) -> np.ndarray:
        """データフレームをRGB配列に変換する。"""
        return np.stack([frame_gray] * 3, axis=-1)
    
    def run(self):
        """フレームを順次生成してキューに追加する。"""
        try:
            # 開始キーフレーム（緑）
            start_frame = self._create_solid_array(self.KEY_FRAME_START_COLOR)
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
                arr = self._create_data_array(frame_gray)
                self.frame_queue.put(arr)
            self.encoder.close()
            
            # 終了キーフレーム（赤）
            end_frame = self._create_solid_array(self.KEY_FRAME_END_COLOR)
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
#  pygame アプリケーション
# ============================================================

class PygameApp:
    """pygame フルスクリーン再生アプリ。"""
    
    def __init__(self, encoder: StreamingEncoder, fps: int, key_duration_sec: int = 3):
        pygame.init()
        
        # フルスクリーン設定
        info = pygame.display.Info()
        self.screen_w = info.current_w
        self.screen_h = info.current_h
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h), pygame.FULLSCREEN)
        pygame.display.set_caption("Bakery Screensaver")
        pygame.mouse.set_visible(False)
        
        self.fps = fps
        self.clock = pygame.time.Clock()
        
        # フレームキューとプロデューサーを初期化
        self.frame_queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=BUFFER_SIZE)
        self.producer = FrameProducer(
            encoder, fps, self.frame_queue, self.screen_w, self.screen_h, key_duration_sec)
        self.producer.start()
        
        # FPSモニタリング用
        self.frame_count = 0
        self.fps_start_time: Optional[float] = None
        self.fps_report_interval = 10.0
        
        self.running = True
    
    def _handle_events(self) -> bool:
        """イベント処理。Falseを返すと終了。"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def _render_frame(self, arr: np.ndarray):
        """フレームを描画する。"""
        # NumPy配列からSurfaceを作成（転置が必要）
        surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        # スケーリングしてフルスクリーン描画
        scaled = pygame.transform.scale(surface, (self.screen_w, self.screen_h))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()
    
    def _update_fps_stats(self):
        """FPS統計を更新・出力する。"""
        now = time.time()
        if self.fps_start_time is None:
            self.fps_start_time = now
        self.frame_count += 1
        
        elapsed = now - self.fps_start_time
        if elapsed >= self.fps_report_interval:
            effective_fps = self.frame_count / elapsed
            print(f"    {self.frame_count} frames in {elapsed:.1f}s = {effective_fps:.1f} fps")
            self.fps_start_time = now
            self.frame_count = 0
    
    def _print_final_fps(self):
        """最終FPS統計を出力する。"""
        if self.fps_start_time is not None and self.frame_count > 0:
            elapsed = time.time() - self.fps_start_time
            effective_fps = self.frame_count / elapsed if elapsed > 0 else 0
            print(f"    Final: {self.frame_count} frames in {elapsed:.1f}s = {effective_fps:.1f} fps")
    
    def run(self):
        """メインループ。"""
        try:
            while self.running:
                if not self._handle_events():
                    break
                
                try:
                    arr = self.frame_queue.get_nowait()
                except queue.Empty:
                    self.clock.tick(self.fps * 2)  # 待機中も適度にスリープ
                    continue
                
                if arr is None:
                    # 終了シグナル - 少し待ってから終了
                    self._print_final_fps()
                    pygame.time.wait(500)
                    break
                
                self._render_frame(arr)
                self._update_fps_stats()
                self.clock.tick(self.fps)
        finally:
            self.producer.stop()
            self.producer.join(timeout=1.0)
            pygame.quit()


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
  python screensaver.py -i file.txt --fps 60 --repetitions 1
"""
    )
    parser.add_argument("-i", "--input", required=True, help="入力ファイル")
    parser.add_argument("--fps", type=int, default=10, help="フレームレート (デフォルト: 10)")
    parser.add_argument("--countdown", type=int, default=5, help="開始前カウントダウン秒数 (デフォルト: 5)")
    parser.add_argument("--key-duration", type=int, default=3, help="キーフレーム表示秒数 (デフォルト: 3)")
    parser.add_argument("--repetitions", type=int, default=1, help="nibbleの繰り返し回数 (デフォルト: 1)")
    parser.add_argument("--block-size", type=int, default=1, help="ブロックサイズ (デフォルト: 1)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # ブロックサイズ設定
    configure_block_size(args.block_size)
    physical_w, physical_h = get_physical_dimensions()

    print(f"[+] Preparing {input_path} ...")
    print(f"    Repetitions: {args.repetitions}")
    print(f"    Block size: {args.block_size} ({physical_w}x{physical_h})")
    encoder = StreamingEncoder(input_path, repetitions=args.repetitions)

    # カウントダウン表示
    if args.countdown > 0:
        print(f"[+] Starting in {args.countdown} seconds ...")
        for i in range(args.countdown, 0, -1):
            print(f"    {i}...", flush=True)
            time.sleep(1)

    print(f"[+] Starting playback at {args.fps} fps (key frames: {args.key_duration}s) ...")
    PygameApp(encoder, args.fps, args.key_duration).run()
    print("[+] Playback finished.")


if __name__ == "__main__":
    main()
