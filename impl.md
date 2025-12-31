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
*   **実装仕様**: `spec.md` の仕様 (960x540, Block=2, Rep=2) に準拠 (修正済み)。

---

## 2. 変更履歴

### 2024-12-31: `encode.py` の仕様不整合修正
以前の実装では、`encode.py` が他のコンポーネントと異なる解像度とプロトコルを使用しており、生成された動画がデコードできない問題が存在した。
現在は修正され、すべてのコンポーネントが統一された仕様 (`spec.md`) に準拠している。

*   **修正内容**:
    *   `LOGICAL_WIDTH`: 480 -> **960**
    *   `LOGICAL_HEIGHT`: 270 -> **540**
    *   `BLOCK_SIZE`: 4 -> **2** (2x2)
    *   `REPETITIONS`: 3 -> **2**
