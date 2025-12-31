# Transporter 実装詳細 (Implementation Details)

本ドキュメントでは、現在のPythonスクリプトの実装詳細、ライブラリ依存関係、および既知の問題について記述する。

## 1. コンポーネント実装

### 1.1. 共通ライブラリと依存関係
*   **Python**: 3.x
*   **Cryptography**: `cryptography` ライブラリを使用 (AES-256-CTR, PBKDF2HMAC)。
*   **Image Processing**: `Pillow` (PIL), `numpy`。
*   **Video Processing**: `ffmpeg` (コマンドラインツール) を `subprocess` 経由で呼び出し。

### 1.2. `screensaver/screensaver.py` (Encoder / Player)
*   **概要**: 暗号化データをリアルタイムで画面に描画する。
*   **GUI**: `tkinter` を使用してフルスクリーンウィンドウを作成。
*   **描画ループ**: `numpy` でフレームデータを生成し、`PIL.Image` -> `ImageTk.PhotoImage` を経由して `tk.Label` に設定。
*   **実装仕様**: `spec.md` の仕様 (960x540, Block=2, Rep=2) に準拠。

### 1.3. `decode.py` (Decoder)
*   **概要**: 動画ファイルを入力とし、元のファイルを復元する。
*   **処理フロー**:
    1. `ffmpeg` で全フレームをPNG画像として一時ディレクトリに抽出。
    2. 画像を `LOGICAL_WIDTH` x `LOGICAL_HEIGHT` に *Nearest Neighbor* でリサイズ。
    3. コーナーマーカーの輝度をチェック (閾値 128)。
    4. 各ピクセルからニブル値を抽出。
    5. `REPETITIONS` 分の値を平均化してニブルを確定。
    6. ニブルを結合して暗号文を復元し、AES復号を行う。
    7. ヘッダ解析とハッシュ検証を経てファイルを保存。
*   **実装仕様**: `spec.md` の仕様 (960x540, Block=2, Rep=2) に準拠。

### 1.4. `encode.py` (Video Encoder)
*   **概要**: ファイルから直接動画ファイル (MP4) を生成する。
*   **処理フロー**:
    1. ファイルを読み込み、ヘッダを付与して暗号化。
    2. 暗号文からニブル列を生成。
    3. `PIL` を使用してフレーム画像を生成 (描画)。
    4. `ffmpeg` を使用して画像連番を動画ファイルに変換。
*   **実装仕様 (現状)**: **`spec.md` と不一致 (後述)。**

---

## 2. 実装上の不整合と課題

### 🚨 重要: コンポーネント間の互換性欠如

現在、`encode.py` の実装のみが、他のコンポーネント (`screensaver.py`, `decode.py`) や `spec.md` の仕様と異なっている。
このため、**`encode.py` で生成した動画ファイルは、`decode.py` で正しくデコードできない**。

#### パラメータ比較

| パラメータ | `spec.md` / `screensaver.py` / `decode.py` | `encode.py` (現状の実装) | 状態 |
| :--- | :--- | :--- | :--- |
| **論理解像度** | **960 x 540** | **480 x 270** | ❌ 不一致 |
| **ブロックサイズ** | **2** (2x2) | **4** (4x4) | ❌ 不一致 |
| **ニブル繰り返し (Repetitions)** | **2** | **3** | ❌ 不一致 |
| **1バイトあたりのピクセル** | **4** | **6** | ❌ 不一致 |

### 解決計画
互換性を確保するため、`encode.py` の実装を修正し、`screensaver.py` / `decode.py` の仕様 (`spec.md`) に合わせる必要がある。

**修正すべき定数:**
*   `LOGICAL_WIDTH` = 960
*   `LOGICAL_HEIGHT` = 540
*   `BLOCK_SIZE` = 2
*   `REPETITIONS` = 2
