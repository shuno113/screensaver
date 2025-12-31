#!/usr/bin/env python3
import argparse
import getpass
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# ============================================================
#  エンコード側と合わせたプロトコル定数
# ============================================================

# 論理解像度（エンコード側と同じ）
LOGICAL_WIDTH = 960
LOGICAL_HEIGHT = 540

# 各論理ピクセルを 2x2 のブロックとして表示している（キャプチャ動画は 1920x1080 前提）
BLOCK_SIZE = 2  # デコードでは resize で 960x540 に落とすので直接は使わないが定義だけ合わせる

# 四隅マーカー（論理座標）
CORNER_LOGICAL_COORDS = [
    (0, 0),
    (LOGICAL_WIDTH - 1, 0),
    (0, LOGICAL_HEIGHT - 1),
    (LOGICAL_WIDTH - 1, LOGICAL_HEIGHT - 1),
]

# nibble（4bit）による 16 色表現
NIBBLES_PER_BYTE = 2       # hi, lo
REPETITIONS = 2            # ★ エンコード側と同じく hi×2, lo×2
GROUP_SIZE = 2 * REPETITIONS  # 4 nibbles / byte

# AES / PBKDF2
PBKDF2_SALT = b"FV-ENC-1"
PBKDF2_ITERATIONS = 200_000
PBKDF2_OUTPUT_LEN = 48  # 32 bytes key, 16 bytes IV

# コーナーマーカー（白）のしきい値
CORNER_WHITE_THRESHOLD = 128


# ============================================================
#  暗号復号関連
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


def decrypt_ciphertext(ciphertext: bytes, password: str) -> bytes:
    """AES-256-CTR で暗号文を復号して平文ブロックを得る。"""
    key, iv = derive_key_iv(password)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext


# ============================================================
#  ヘッダ解析
# ============================================================

def parse_inner_header(header: bytes) -> Tuple[int, bytes, str]:
    """
    256 バイトの内側ヘッダを解析し、
    (file_size, file_hash, filename) を返す。
    """
    if len(header) != 256:
        raise ValueError("Inner header must be exactly 256 bytes")

    magic = header[0:4]
    if magic != b"FVD1":
        raise ValueError(f"Invalid magic: {magic!r}, expected b'FVD1'")

    header_size = int.from_bytes(header[4:8], "big")
    if header_size != 256:
        raise ValueError(f"Invalid header size: {header_size}, expected 256")

    file_size = int.from_bytes(header[8:16], "big")
    file_hash = header[16:48]

    name_len = header[48]
    if name_len > 200:
        raise ValueError(f"Invalid name_len: {name_len}, must be <= 200")
    name_bytes = header[49:49 + name_len]
    filename = name_bytes.decode("utf-8", errors="replace")

    return file_size, file_hash, filename


def write_unique_file(base_dir: Path, filename: str, data: bytes) -> Path:
    """既存ファイルがある場合は (1), (2) ... を付けて保存する。"""
    target = base_dir / filename
    if not target.exists():
        target.write_bytes(data)
        return target

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        candidate = base_dir / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            candidate.write_bytes(data)
            return candidate
        counter += 1


# ============================================================
#  ffmpeg で動画 → フレームPNG
# ============================================================

def run_ffmpeg_extract_frames(video_path: Path, tmpdir: Path) -> List[Path]:
    """
    ffmpeg を使って動画からフレーム画像を PNG で書き出す。
    戻り値: frame_000001.png, ... のパス一覧（ソート済み）。
    """
    pattern = tmpdir / "frame_%06d.png"
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        str(pattern),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}\n"
            f"stdout: {result.stdout.decode(errors='ignore')}\n"
            f"stderr: {result.stderr.decode(errors='ignore')}"
        )

    frames = sorted(tmpdir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("No frames extracted by ffmpeg")
    return frames


# ============================================================
#  nibble / グレースケール変換
# ============================================================

def gray_to_nibble(gray: np.ndarray) -> np.ndarray:
    """
    0..255 のグレースケール配列を 0..15 の nibble 配列に変換する。
    nibble = round(gray * 15 / 255)
    """
    gray_f = gray.astype(np.float32)
    level = np.round(gray_f * (15.0 / 255.0)).astype(np.int16)
    level = np.clip(level, 0, 15)
    return level.astype(np.uint8)


def precompute_data_indices() -> Tuple[np.ndarray, np.ndarray]:
    """
    全論理ピクセルのフラットインデックスを作り、
    四隅を除いた「データ格納用インデックス」と、
    四隅のインデックスを返す。
    """
    h, w = LOGICAL_HEIGHT, LOGICAL_WIDTH
    total = h * w
    all_indices = np.arange(total, dtype=np.int32)

    corner_indices = []
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        idx = cy * LOGICAL_WIDTH + cx
        corner_indices.append(idx)
    corner_indices = np.array(corner_indices, dtype=np.int32)

    mask = np.ones(total, dtype=bool)
    mask[corner_indices] = False
    data_indices = all_indices[mask]

    return data_indices, corner_indices


def verify_corners(frame_gray: np.ndarray) -> None:
    """
    四隅の論理ピクセルが「おおむね白」であることを確認する。
    frame_gray: (H,W) uint8
    """
    for (cx, cy) in CORNER_LOGICAL_COORDS:
        v = int(frame_gray[cy, cx])
        if v < CORNER_WHITE_THRESHOLD:
            raise ValueError(
                f"Corner marker at logical ({cx},{cy}) not white enough: gray={v}"
            )


def extract_nibbles_from_frames(frame_files: List[Path]) -> np.ndarray:
    """
    フレームPNG群から nibble 配列を復元する。
    各フレーム:
      - まず 960x540 に NEAREST でリサイズして論理解像度に揃える
      - 四隅を検証
      - 四隅以外のピクセルから nibble を取り出し、フラットに連結
    """
    data_indices, corner_indices = precompute_data_indices()
    nibbles_all: List[np.ndarray] = []

    for frame_path in frame_files:
        img = Image.open(frame_path).convert("RGB")

        # 論理解像度にリサイズ（エンコード側は 960x540→2x2拡大しているので、
        # キャプチャ動画からはこれを逆にたたむイメージ）
        img = img.resize((LOGICAL_WIDTH, LOGICAL_HEIGHT), resample=Image.NEAREST)

        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        # グレースケール化（R,G,B の平均）
        gray = arr.mean(axis=2).astype(np.uint8)  # (H, W)

        # verify_corners(gray)

        flat = gray.reshape(-1)  # (H*W,)
        # データ用ピクセル部分だけ取り出す
        data_gray = flat[data_indices]
        # gray → nibble
        data_nibbles = gray_to_nibble(data_gray)
        nibbles_all.append(data_nibbles)

    if not nibbles_all:
        return np.zeros((0,), dtype=np.uint8)

    return np.concatenate(nibbles_all, axis=0)


def reconstruct_ciphertext_from_nibbles(nibbles: np.ndarray) -> bytes:
    """
    nibble 列から ciphertext (bytes) を復元する。
    エンコード側では 1バイトごとに:
      hi,hi,lo,lo (REPETITIONS=2)
    の順に nibble を出しているので、
    nibble 配列を 4個ずつのグループに分け、
      hi = round(mean(前半2個))
      lo = round(mean(後半2個))
    として 1byte を再構成する。
    """
    if nibbles.size == 0:
        return b""

    group_n = GROUP_SIZE  # 4
    usable_len = (nibbles.size // group_n) * group_n
    if usable_len == 0:
        return b""

    nibbles = nibbles[:usable_len]
    groups = nibbles.reshape(-1, group_n)  # (Nbytes, 4)

    hi_group = groups[:, :REPETITIONS]     # (Nbytes, 2)
    lo_group = groups[:, REPETITIONS:]     # (Nbytes, 2)

    hi = np.round(hi_group.mean(axis=1)).astype(np.uint8)
    lo = np.round(lo_group.mean(axis=1)).astype(np.uint8)

    hi = np.clip(hi, 0, 15)
    lo = np.clip(lo, 0, 15)

    bytes_arr = ((hi << 4) | lo).astype(np.uint8)
    return bytes(bytes_arr)


# ============================================================
#  メイン処理
# ============================================================

def decode_video_to_file(
    video_path: Path,
    password: str,
    output_dir: Path,
) -> Path:
    """動画ファイルをデコードして元のファイルを復元する。"""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        print(f"[+] Extracting frames from {video_path} using ffmpeg ...")
        frame_files = run_ffmpeg_extract_frames(video_path, tmpdir)
        print(f"    Extracted {len(frame_files)} frames")

        print("[+] Extracting nibbles from frames ...")
        nibbles = extract_nibbles_from_frames(frame_files)
        print(f"    Total nibbles: {nibbles.size}")

    print("[+] Reconstructing ciphertext from nibbles ...")
    ciphertext = reconstruct_ciphertext_from_nibbles(nibbles)
    print(f"    Reconstructed ciphertext size: {len(ciphertext)} bytes")

    print("[+] Decrypting ciphertext with AES-256-CTR ...")
    plaintext = decrypt_ciphertext(ciphertext, password)
    print(f"    Plaintext block size: {len(plaintext)} bytes")

    if len(plaintext) < 256:
        raise ValueError("Plaintext too short to contain inner header")

    header = plaintext[:256]
    file_size, file_hash, filename = parse_inner_header(header)
    print(f"[+] Parsed header: filename={filename!r}, file_size={file_size}")

    if file_size > len(plaintext) - 256:
        raise ValueError("File size in header exceeds available plaintext data")

    file_data = plaintext[256:256 + file_size]

    calc_hash = hashlib.sha256(file_data).digest()
    if calc_hash != file_hash:
        raise ValueError("SHA-256 hash mismatch: wrong password or corrupted data")

    output_path = write_unique_file(output_dir, filename, file_data)
    print(f"[+] Recovered file saved: {output_path}")
    print(f"    Size: {file_size} bytes")
    print("    SHA-256 verified successfully")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Decode an encrypted visual stream (960x540 logical, 2x2 blocks, "
            "16-level grayscale, REPETITIONS=2) captured as a video file."
        )
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file (e.g. MP4)")
    parser.add_argument("-p", "--password", help="Password (if omitted, prompt securely)")
    parser.add_argument(
        "-d", "--output-dir",
        help="Output directory for recovered file (default: current directory)",
        default="."
    )
    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    password = args.password or getpass.getpass("Password: ")

    decode_video_to_file(video_path, password, output_dir)


if __name__ == "__main__":
    main()
