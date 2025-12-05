#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import math
import platform
import ctypes
from pathlib import Path
from typing import Tuple

import pygame
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ============================================================
#  設定値
# ============================================================

# 論理解像度（データを詰めるグリッド）
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 540

# 各論理ピクセルを 2x2 のブロックとして表示する
BLOCK_SIZE = 2

# 実際の描画解像度（ウィンドウ/フルスクリーン）
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
REPETITIONS = 3       # 各 nibble を3回繰り返す
LOGICAL_PIXELS_PER_BYTE = NIBBLES_PER_BYTE * REPETITIONS  # 6 ピクセルで1バイト

# 1 フレームに使えるデータ用論理ピクセル数
DATA_PIXELS_PER_FRAME = LOGICAL_WIDTH * LOGICAL_HEIGHT - len(CORNER_LOGICAL_COORDS)
BYTES_PER_FRAME = DATA_PIXELS_PER_FRAME // LOGICAL_PIXELS_PER_BYTE

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
#  nibble → グレースケール（16レベル）
# ============================================================

def nibble_to_gray(n: int) -> int:
    """
    0..15 の nibble を 0..255 のグレースケールにマップする。
    16段階の等間隔レベル。
    """
    n = max(0, min(15, n))
    return round(n * 255 / 15)


class NibbleGenerator:
    """
    ciphertext から nibble を順番に生成する。
    各バイトごとに:
      hi = b >> 4
      lo = b & 0x0F
    hi を REPETITIONS 回、lo を REPETITIONS 回返す。
    ciphertext を使い切ったら 0 nibble を返し続ける。
    """

    def __init__(self, ciphertext: bytes, repetitions: int):
        self.ct = ciphertext
        self.rep = repetitions
        self.byte_i = 0
        self.phase = 0  # 0..(2*rep-1)

    def next(self) -> int:
        if self.byte_i < len(self.ct):
            b = self.ct[self.byte_i]
            hi = (b >> 4) & 0x0F
            lo = b & 0x0F
            nib = hi if self.phase < self.rep else lo
            self.phase += 1
            if self.phase >= 2 * self.rep:
                self.phase = 0
                self.byte_i += 1
            return nib
        # データが尽きたら 0（黒）で埋める
        return 0


# ============================================================
#  フレーム生成（論理解像度で生成 → 2x2 拡大）
# ============================================================

def generate_logical_frame(nib_gen: NibbleGenerator) -> pygame.Surface:
    """
    1フレーム分の「論理解像度」Surface (960x540) を生成する。
    ここで 4bit nibble → 16段階グレースケールに変換し、
    四隅以外の論理ピクセルを塗る。
    """
    surf = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))
    surf.lock()

    for y in range(LOGICAL_HEIGHT):
        for x in range(LOGICAL_WIDTH):
            if (x, y) in CORNER_LOGICAL_COORDS:
                continue
            nib = nib_gen.next()
            gray = nibble_to_gray(nib)
            surf.set_at((x, y), (gray, gray, gray))

    # 四隅を白で上書き（位置合わせマーカー）
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        surf.set_at((cx, cy), (255, 255, 255))

    surf.unlock()
    return surf


# ============================================================
#  最前面化（Windows 用）
# ============================================================

def make_window_topmost():
    """
    Windows の場合、WinAPI でウィンドウを TOPMOST にする。
    他 OS ではフルスクリーン + フォーカスに依存。
    """
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

    print(f"[+] frames={frame_count}, fps={fps}, bytes/frame={BYTES_PER_FRAME}")

    pygame.init()
    # フルスクリーン 1920x1080（物理解像度）
    screen = pygame.display.set_mode(
        (PHYSICAL_WIDTH, PHYSICAL_HEIGHT),
        pygame.FULLSCREEN | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Encrypted Visual Stream (960x540 logical, 2x2 blocks)")
    pygame.mouse.set_visible(False)

    make_window_topmost()

    clock = pygame.time.Clock()
    nib = NibbleGenerator(ciphertext, REPETITIONS)

    running = True
    for fi in range(frame_count):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if not running:
            break

        # 論理解像度でフレーム生成
        logical = generate_logical_frame(nib)
        # 2x2 拡大 → 1920x1080 で描画
        scaled = pygame.transform.scale(logical, (PHYSICAL_WIDTH, PHYSICAL_HEIGHT))

        screen.blit(scaled, (0, 0))
        pygame.display.flip()

        clock.tick(fps)

    pygame.time.wait(500)
    pygame.quit()


# ============================================================
#  main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fullscreen encoder (960x540 logical, 2x2 blocks, 16-level grayscale, repeated nibbles)."
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
