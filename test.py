# check_encode.py - エンコード側の診断
from pathlib import Path
from common import (
    build_plaintext_block, precompute_data_indices,
    nibbles_from_bytes, nibbles_to_gray, generate_frame_array,
)

# 任意の小さなテストファイルを指定
test_file = Path("any_small_file.txt")  # ← 実際に存在するファイルに変更してください
data = build_plaintext_block(test_file)

print(f"Magic bytes: {data[:4]}")
print(f"First 4 bytes hex: {data[:4].hex()}")

gray_stream = nibbles_to_gray(nibbles_from_bytes(data))
data_indices, corner_indices = precompute_data_indices()

frame, _ = generate_frame_array(gray_stream, 0, data_indices, corner_indices)
flat = frame.reshape(-1)

print(f"\nFirst 10 pixels of generated frame:")
for i in range(10):
    print(f"  Index {i}: gray={flat[i]}")

print(f"\nCorner indices: {corner_indices}")
print(f"Pixels at corners:")
for idx in corner_indices:
    print(f"  Index {idx}: gray={flat[idx]}")