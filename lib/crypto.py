#!/usr/bin/env python3
"""暗号処理モジュール"""
from typing import Tuple

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from lib.config import PBKDF2_SALT, PBKDF2_ITERATIONS, PBKDF2_OUTPUT_LEN


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


def encrypt_plaintext(plaintext: bytes, password: str) -> bytes:
    """AES-256-CTR で平文ブロックを暗号化。"""
    key, iv = derive_key_iv(password)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext


def decrypt_ciphertext(ciphertext: bytes, password: str) -> bytes:
    """AES-256-CTR で暗号文を復号して平文ブロックを得る。"""
    key, iv = derive_key_iv(password)
    cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    return plaintext
