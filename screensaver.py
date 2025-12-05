#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import math
import platform
import ctypes
from pathlib import Path
from typing import Tuple

import numpy as np
import pygame
import pygame.surfarray as surfarray
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ============================================================
#  基本設定
# ============================================================

# 論理解像度（データ格納用グリッド）
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 540

# 各論理ピクセルを 2x2 のブロックとして表示
BLOCK_SIZE = 2

# 実際の描画解像度（フルスクリーン）
PHYSICAL_WIDTH = LOGICAL_WIDTH * BLOCK_SIZE    # 1920
PHYSICAL_HEIGHT = LOGICAL_HEIGHT * BLOCK_SIZE  # 1080

# 四隅マーカー（論理座標）
CORNER_LOGICAL_COORDS = [
    (0, 0),
    (LOGICAL_WIDTH - 1, 0),
    (0, LOGICAL_HEIGHT - 1),
    (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
]

# 16色 → 4bit nibble
NIBBLES_PER_BYTE = 2  # hi, lo
REPETITIONS = 2       # ★ 3→2 に変更：各 nibble を2回繰り返す
LOGICAL_PIXELS_PER_BYTE = NIBBLES_PER_BYTE * REPETITIONS  # 4 論理ピクセルで1バイト

# 1 フレームに使えるデータ用論理ピクセル数
TOTAL_LOGICAL_PIXELS = LOGICAL_WIDTH * LOGICAL_HEIGHT
DATA_PIXELS_PER_FRAME = TOTAL_LOGICAL_PIXELS - len(CORNER_LOGICAL_COORDS)
BYTES_PER_FRAME = DATA_PIXELS_PER_FRAME // LOGICAL_PIXELS_PER_BYTE  # 1フレームで運べるバイト数

# AES / PBKDF2
PBKDF2_SALT = b"FV-ENC-1"
PBKDF2_ITERATIONS = 200_000
PBKDF2_OUTPUT_LEN = 48  # 32 bytes key, 16 bytes IV


# ============================================================
#  暗号処理
# ============================================================

def derive_key_iv(password: str) -> Tuple[bytes, bytes]:
    """パスワードから AES-256 キーと IV を導出する。"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=PBKDF2_OUTPUT_LEN,
        salt=PBKDF2_SALT,
        iterations=PBKDF2_ITERATIONS,
        backend=default_backend(),
    )
    derived = kdf.derive(password.encode("utf-8"))
    key = derived[:32]
    iv = derived[32:]
    return key, iv


def build_inner_header(file_path: Path, file_bytes: bytes) -> bytes:
    """256 バイトの内側ヘッダを構築する。"""
    header = bytearray(256)
    # Magic "FVD1"
    header[0:4] = b"FVD1"
    # Header size = 256 (uint32 BE)
    header[4:8] = (256).to_bytes(4, "big")
    # File size (uint64 BE)
    file_size = len(file_bytes)
    header[8:16] = file_size.to_bytes(8, "big")
    # SHA-256 hash
    sha = hashlib.sha256(file_bytes).digest()
    header[16:48] = sha

    # File name (no directory)
    name = file_path.name.encode("utf-8")
    if len(name) > 200:
        name = name[:200]
    header[48] = len(name)  # name_len
    header[49:49+len(name)] = name
    # 残りは 0 のまま
    return bytes(header)


def build_plaintext_block(file_path: Path) -> bytes:
    """平文ブロック = [内側ヘッダ][元ファイル] を構築する。"""
    data = file_path.read_bytes()
    header = build_inner_header(file_path, data)
    plaintext = header + data
    return plaintext


def encrypt_plaintext(plaintext: bytes, password: str) -> bytes:
    """AES-256-CTR で平文ブロックを暗号化。"""
    key, iv = derive_key_iv(password)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext


# ============================================================
#  nibble 生成とグレースケール変換
# ============================================================

def nibbles_from_bytes(ciphertext: bytes) -> np.ndarray:
    """
    ciphertext (uint8) から hi/lo nibble を取り出し、
    hi, lo を REPETITIONS 回ずつ繰り返した nibble 配列を返す。
    例: REPETITIONS=2 の場合、1 byte→[hi,hi,lo,lo]（4 nibbles）。
    """
    ct = np.frombuffer(ciphertext, dtype=np.uint8)
    hi = (ct >> 4) & 0x0F
    lo = ct & 0x0F

    # shape: (len(ct), REPETITIONS) に並べて平坦化
    hi_rep = np.repeat(hi[:, None], REPETITIONS, axis=1)  # (N, REP)
    lo_rep = np.repeat(lo[:, None], REPETITIONS, axis=1)  # (N, REP)

    # 例: REP=2 の場合、[hi,hi,lo,lo]
    nibbles = np.concatenate([hi_rep, lo_rep], axis=1).reshape(-1)
    return nibbles.astype(np.uint8)


def nibbles_to_gray(nibbles: np.ndarray) -> np.ndarray:
    """
    nibble(0..15) の配列を 0..255 グレースケール配列に変換する。
    gray = round(n * 255 / 15)
    """
    n = nibbles.astype(np.float32)
    gray = np.round(n * (255.0 / 15.0)).astype(np.uint8)
    return gray


# ============================================================
#  フレーム生成（numpyベース）
# ============================================================

def precompute_data_indices() -> Tuple[np.ndarray, np.ndarray]:
    """
    全論理ピクセルのフラットインデックスを作り、
    四隅を除いた「データ格納用インデックス」と、
    四隅のインデックスを返す。
    """
    h, w = LOGICAL_HEIGHT, LOGICAL_WIDTH
    total = h * w
    all_indices = np.arange(total, dtype=np.int32)

    # corner のフラットインデックスを計算
    corner_indices = []
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        idx = cy * LOGICAL_WIDTH + cx
        corner_indices.append(idx)
    corner_indices = np.array(corner_indices, dtype=np.int32)

    # データ用インデックス: corner 以外
    mask = np.ones(total, dtype=bool)
    mask[corner_indices] = False
    data_indices = all_indices[mask]

    return data_indices, corner_indices


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
    need = min(DATA_PIXELS_PER_FRAME, len(gray_stream) - offset)
    # フレーム全体は 0 (黒) で初期化
    frame_flat = np.zeros(total_pixels, dtype=np.uint8)

    if need > 0:
        frame_flat[data_indices[:need]] = gray_stream[offset: offset + need]
    # 四隅は白(255)にしてマーカー
    frame_flat[corner_indices] = 255

    frame_gray = frame_flat.reshape(h, w)
    return frame_gray, offset + need


# ============================================================
#  最前面化（Windows 用）
# ============================================================

def make_window_topmost():
    """Windows の場合、WinAPI でウィンドウを TOPMOST にする。"""
    if platform.system() == "Windows":
        try:
            hwnd = pygame.display.get_wm_info().get("window")
            if hwnd:
                user32 = ctypes.windll.user32
                HWND_TOPMOST = -1
                SWP_NOSIZE = 0x0001
                SWP_NOMOVE = 0x0002
                user32.SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    0, 0, 0, 0,
                    SWP_NOMOVE | SWP_NOSIZE,
                )
        except Exception as e:
            print("[!] TOPMOST failed:", e)


# ============================================================
#  フルスクリーン再生
# ============================================================

def playback(ciphertext: bytes, fps: int):
    if BYTES_PER_FRAME <= 0:
        raise ValueError("BYTES_PER_FRAME must be positive")

    total_bytes = len(ciphertext)
    frame_count = math.ceil(total_bytes / BYTES_PER_FRAME)
    print(f"[+] bytes={total_bytes}, bytes/frame={BYTES_PER_FRAME}, frames={frame_count}, fps={fps}")

    # ciphertext → nibble配列 → グレースケール配列
    nibbles = nibbles_from_bytes(ciphertext)
    gray_stream = nibbles_to_gray(nibbles)
    print(f"[+] nibbles={len(nibbles)}, gray_stream={len(gray_stream)}")

    # データ位置と corner 位置を事前計算
    data_indices, corner_indices = precompute_data_indices()

    pygame.init()
    screen = pygame.display.set_mode(
        (PHYSICAL_WIDTH, PHYSICAL_HEIGHT),
        pygame.FULLSCREEN | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Encrypted Visual Stream (960x540 logical, 2x2 blocks, REP=2)")
    pygame.mouse.set_visible(False)

    make_window_topmost()

    clock = pygame.time.Clock()

    offset = 0  # gray_stream の読み出し位置
    running = True

    for fi in range(frame_count):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        # 1フレーム分のグレースケール画像を生成
        frame_gray, offset = generate_frame_array(gray_stream, offset, data_indices, corner_indices)
        # (H,W) → (H,W,3) RGB
        frame_rgb = np.stack([frame_gray]*3, axis=-1)

        # numpy → pygame.Surface
        # surfarray.make_surface は (W,H,3) 形式を期待するので転置
        surface = surfarray.make_surface(frame_rgb.swapaxes(0, 1))

        # そのまま表示（論理 960x540 を 2x2 → 1920x1080 に拡大）
        scaled = pygame.transform.scale(surface, (PHYSICAL_WIDTH, PHYSICAL_HEIGHT))
        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        clock.tick(fps)

    pygame.time.wait(300)
    pygame.quit()


# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fullscreen encoder (960x540 logical, 2x2 blocks, 16-level grayscale, REPETITIONS=2, numpy-optimized)."
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

    playback(ciphertext, args.fps)
    print("[+] Playback finished. Exiting.")


if __name__ == "__main__":
    main()
