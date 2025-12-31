#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from PIL import Image

# 論理解像度（4x4 ブロック単位）
LOGICAL_WIDTH = 480
LOGICAL_HEIGHT = 270
BLOCK_SIZE = 4  # 4x4 物理ピクセル = 1 論理ピクセル

# 物理解像度（動画の実サイズ）
PHYSICAL_WIDTH = LOGICAL_WIDTH * BLOCK_SIZE   # 1920
PHYSICAL_HEIGHT = LOGICAL_HEIGHT * BLOCK_SIZE  # 1080

# 四隅の予約論理ピクセル（4x4 ブロック単位で白固定）
CORNER_LOGICAL_COORDS = [
    (0, 0),
    (LOGICAL_WIDTH - 1, 0),
    (0, LOGICAL_HEIGHT - 1),
    (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
]

# 16色 → 4bit nibble
NIBBLES_PER_BYTE = 2  # hi, lo
REPETITIONS = 3       # 各 nibble を3回繰り返す
LOGICAL_PIXELS_PER_BYTE = NIBBLES_PER_BYTE * REPETITIONS  # 6

DATA_PIXELS_PER_FRAME = LOGICAL_WIDTH * LOGICAL_HEIGHT - len(CORNER_LOGICAL_COORDS)
BYTES_PER_FRAME = DATA_PIXELS_PER_FRAME // LOGICAL_PIXELS_PER_BYTE  # 1フレームで運べるバイト数

PBKDF2_SALT = b"FV-ENC-1"
PBKDF2_ITERATIONS = 200_000
PBKDF2_OUTPUT_LEN = 48  # 32 bytes key + 16 bytes IV


def derive_key_iv(password: str) -> Tuple[bytes, bytes]:
    """Derive AES-256 key and IV from password using PBKDF2-HMAC-SHA256."""
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
    """Build 256-byte inner header with metadata."""
    header = bytearray(256)
    # Magic "FVD1"
    header[0:4] = b"FVD1"
    # Header size = 256 (uint32 BE)
    header[4:8] = (256).to_bytes(4, "big")
    # File size (uint64 BE)
    file_size = len(file_bytes)
    header[8:16] = file_size.to_bytes(8, "big")
    # SHA-256 hash of file
    sha = hashlib.sha256(file_bytes).digest()
    header[16:48] = sha

    # File name (no directory)
    name = file_path.name.encode("utf-8")
    if len(name) > 200:
        name = name[:200]
    header[48] = len(name)  # name_len (uint8)
    header[49:49+len(name)] = name
    # remaining bytes are already zero
    return bytes(header)


def build_plaintext_block(file_path: Path) -> bytes:
    """Construct plaintext_block = [header][file_bytes][optional padding]."""
    data = file_path.read_bytes()
    header = build_inner_header(file_path, data)
    plaintext = header + data
    # 以前は 3 の倍数にしていたが、新方式では必須ではない。
    # 必要なら少しだけゼロパディングしても良いし、無しでも良い。
    return plaintext


def encrypt_plaintext(plaintext: bytes, password: str) -> bytes:
    """Encrypt plaintext with AES-256-CTR using key/iv derived from password."""
    key, iv = derive_key_iv(password)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext


def logical_to_physical_block(x_log: int, y_log: int) -> Tuple[int, int]:
    """Convert logical pixel coordinates to top-left physical pixel coordinates."""
    x_phys = x_log * BLOCK_SIZE
    y_phys = y_log * BLOCK_SIZE
    return x_phys, y_phys


def fill_block(image: Image.Image, x_log: int, y_log: int, gray: int) -> None:
    """Fill a 4x4 block corresponding to a logical pixel with given grayscale value."""
    x0, y0 = logical_to_physical_block(x_log, y_log)
    color = (gray, gray, gray)
    for dy in range(BLOCK_SIZE):
        for dx in range(BLOCK_SIZE):
            image.putpixel((x0 + dx, y0 + dy), color)


def is_corner_logical(x_log: int, y_log: int) -> bool:
    return (x_log, y_log) in CORNER_LOGICAL_COORDS


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


def generate_frames(ciphertext: bytes, tmp_dir: Path, fps: int) -> int:
    """Generate PNG frames from ciphertext using nibble repetition scheme. Returns frame count."""
    if BYTES_PER_FRAME <= 0:
        raise ValueError("BYTES_PER_FRAME must be positive")

    total_bytes = len(ciphertext)
    frame_count = math.ceil(total_bytes / BYTES_PER_FRAME)

    nib_gen = NibbleGenerator(ciphertext, REPETITIONS)

    for frame_index in range(frame_count):
        img = Image.new("RGB", (PHYSICAL_WIDTH, PHYSICAL_HEIGHT), (0, 0, 0))

        for y_log in range(LOGICAL_HEIGHT):
            for x_log in range(LOGICAL_WIDTH):
                if is_corner_logical(x_log, y_log):
                    # reserved for white marker
                    continue
                nibble = nib_gen.next_nibble()
                gray = nibble_to_gray(nibble)
                fill_block(img, x_log, y_log, gray)

        # ensure corners are pure white blocks
        for (cx, cy) in CORNER_LOGICAL_COORDS:
            fill_block(img, cx, cy, 255)

        frame_name = tmp_dir / f"frame_{frame_index:06d}.png"
        img.save(frame_name, format="PNG")

    return frame_count


def run_ffmpeg(tmp_dir: Path, output_path: Path, fps: int) -> None:
    """Call ffmpeg to convert PNG frames to MP4 video."""
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "frame_%06d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "5",        # ここはロスレスではないが、16色+繰り返しである程度耐性を持たせる前提
        "-pix_fmt", "yuv444p",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}\n"
            f"stdout: {result.stdout.decode(errors='ignore')}\n"
            f"stderr: {result.stderr.decode(errors='ignore')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode a file into an encrypted H.264 video (1920x1080, 4x4 blocks, 16-level grayscale, repeated nibbles)."
    )
    parser.add_argument("-i", "--input", required=True, help="Input file path")
    parser.add_argument("-o", "--output", required=True, help="Output MP4 file path")
    parser.add_argument("-p", "--password", help="Password (if omitted, prompt securely)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

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

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        print(f"[+] Generating frames in {tmpdir} ...")
        frame_count = generate_frames(ciphertext, tmpdir, args.fps)
        print(f"    Generated {frame_count} frames")

        print("[+] Running ffmpeg to create video ...")
        run_ffmpeg(tmpdir, output_path, args.fps)

    print(f"[+] Done. Output video: {output_path}")


if __name__ == "__main__":
    main()
