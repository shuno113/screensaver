#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import math
from pathlib import Path
from typing import Tuple

import pygame
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# 論理解像度（エンコード単位）
LOGICAL_WIDTH = 480
LOGICAL_HEIGHT = 270

# 各論理ピクセルは 4x4 のブロックとして表示する想定
BLOCK_SIZE = 4  # 拡大倍率

# 物理解像度（ウィンドウ/フルスクリーン）
PHYSICAL_WIDTH = LOGICAL_WIDTH * BLOCK_SIZE   # 1920
PHYSICAL_HEIGHT = LOGICAL_HEIGHT * BLOCK_SIZE  # 1080

# 四隅の予約論理ピクセル（位置合わせマーカー用、常に白）
CORNER_LOGICAL_COORDS = [
    (0, 0),
    (LOGICAL_WIDTH - 1, 0),
    (0, LOGICAL_HEIGHT - 1),
    (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
]

# 16色 → 4bit nibble
NIBBLES_PER_BYTE = 2  # hi, lo
REPETITIONS = 3       # 各 nibble を3回繰り返す
LOGICAL_PIXELS_PER_BYTE = NIBBLES_PER_BYTE * REPETITIONS  # 6 論理ピクセルで1バイト

# 1 フレームに使えるデータ用論理ピクセル数
DATA_PIXELS_PER_FRAME = LOGICAL_WIDTH * LOGICAL_HEIGHT - len(CORNER_LOGICAL_COORDS)
BYTES_PER_FRAME = DATA_PIXELS_PER_FRAME // LOGICAL_PIXELS_PER_BYTE

PBKDF2_SALT = b"FV-ENC-1"
PBKDF2_ITERATIONS = 200_000
PBKDF2_OUTPUT_LEN = 48  # 32 bytes key + 16 bytes IV


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
    # 残りは 0 埋めのまま
    return bytes(header)


def build_plaintext_block(file_path: Path) -> bytes:
    """平文ブロック = [内側ヘッダ][元ファイル] を構築する。"""
    data = file_path.read_bytes()
    header = build_inner_header(file_path, data)
    plaintext = header + data
    return plaintext


def encrypt_plaintext(plaintext: bytes, password: str) -> bytes:
    """AES-256-CTR で平文ブロックを暗号化し ciphertext を返す。"""
    key, iv = derive_key_iv(password)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext


def nibble_to_gray(nibble: int) -> int:
    """
    0..15 の nibble を 0..255 のグレースケールにマップする。
    16段階の等間隔レベル。
    """
    nibble = max(0, min(15, nibble))
    return round(nibble * 255 / 15)


class NibbleGenerator:
    """
    ciphertext から nibble を順番に生成するジェネレータ。
    各バイトごとに:
      hi = b >> 4
      lo = b & 0x0F
    hi を REPETITIONS 回、lo を REPETITIONS 回返す。
    ciphertext を使い切ったら 0 nibble を返し続ける。
    """

    def __init__(self, ciphertext: bytes, repetitions: int):
        self.ciphertext = ciphertext
        self.repetitions = repetitions
        self.byte_index = 0
        self.phase = 0  # 0..(2*repetitions - 1)

    def next_nibble(self) -> int:
        if self.byte_index < len(self.ciphertext):
            b = self.ciphertext[self.byte_index]
            hi = (b >> 4) & 0x0F
            lo = b & 0x0F
            if self.phase < self.repetitions:
                nib = hi
            else:
                nib = lo
            self.phase += 1
            if self.phase >= 2 * self.repetitions:
                self.phase = 0
                self.byte_index += 1
            return nib
        else:
            # ciphertext を使い切ったあとは 0 nibble（=黒）で埋める
            return 0


def generate_logical_frame_surface(ciphertext: bytes,
                                   nib_gen: NibbleGenerator) -> pygame.Surface:
    """
    1フレーム分の「論理解像度」Surface (480x270) を生成する。
    ここで 4bit nibble → 16段階グレースケールに変換し、
    四隅以外の論理ピクセルを塗る。
    """
    surface = pygame.Surface((LOGICAL_WIDTH, LOGICAL_HEIGHT))
    surface.lock()

    for y in range(LOGICAL_HEIGHT):
        for x in range(LOGICAL_WIDTH):
            if (x, y) in CORNER_LOGICAL_COORDS:
                # corner は後でまとめて白にするので一旦スキップしてもよいが、
                # ここで直接白を書いてもよい
                continue
            nib = nib_gen.next_nibble()
            gray = nibble_to_gray(nib)
            surface.set_at((x, y), (gray, gray, gray))

    # 四隅を白で上書き（位置合わせマーカー）
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        surface.set_at((cx, cy), (255, 255, 255))

    surface.unlock()
    return surface


def run_fullscreen_playback(ciphertext: bytes, fps: int) -> None:
    """
    ciphertext をフレーム列にマッピングし、
    フルスクリーンで順次表示して終了する。
    """
    if BYTES_PER_FRAME <= 0:
        raise ValueError("BYTES_PER_FRAME must be positive")

    total_bytes = len(ciphertext)
    frame_count = math.ceil(total_bytes / BYTES_PER_FRAME)

    print(f"[+] Fullscreen playback: {frame_count} frames at {fps} fps")

    # pygame 初期化
    pygame.init()
    # フルスクリーン & ダブルバッファ
    screen = pygame.display.set_mode(
        (PHYSICAL_WIDTH, PHYSICAL_HEIGHT),
        pygame.FULLSCREEN | pygame.DOUBLEBUF
    )
    pygame.display.set_caption("Encrypted Stream Encoder")
    pygame.mouse.set_visible(False)  # マウスカーソル非表示

    clock = pygame.time.Clock()
    nib_gen = NibbleGenerator(ciphertext, REPETITIONS)

    running = True
    for frame_index in range(frame_count):
        # イベント処理（ESC で中断可能）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not running:
            break

        # 1フレーム分の論理 Surface を生成
        logical_surf = generate_logical_frame_surface(ciphertext, nib_gen)
        # 物理解像度に拡大（最近傍補間）
        scaled_surf = pygame.transform.scale(
            logical_surf, (PHYSICAL_WIDTH, PHYSICAL_HEIGHT)
        )

        # 描画
        screen.blit(scaled_surf, (0, 0))
        pygame.display.flip()

        # fps 制御
        clock.tick(fps)

    # 少し待ってから終了
    pygame.time.wait(300)
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a file into an encrypted fullscreen visual stream "
            "(1920x1080, 4x4 blocks, 16-level grayscale, repeated nibbles)."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-p", "--password", help="Password (if omitted, prompt securely)")
    parser.add_argument(
        "--fps", type=int, default=10,
        help="Frames per second for playback (default: 10)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    password = args.password
    if password is None:
        password = getpass.getpass("Password: ")

    print(f"[+] Building plaintext block from {input_path} ...")
    plaintext = build_plaintext_block(input_path)
    print(f"    Plaintext size: {len(plaintext)} bytes")

    print("[+] Encrypting plaintext with AES-256-CTR ...")
    ciphertext = encrypt_plaintext(plaintext, password)
    print(f"    Ciphertext size: {len(ciphertext)} bytes")

    # フルスクリーン連続表示
    run_fullscreen_playback(ciphertext, args.fps)

    print("[+] Playback finished. Exiting.")


if __name__ == "__main__":
    main()
