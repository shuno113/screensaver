# Bakery 実装詳細 (Implementation Details)

本ドキュメントでは、現在のPythonスクリプトの実装詳細、ライブラリ依存関係、および既知の問題について記述する。

## 1. コンポーネント実装

### 1.1. 共通ライブラリと依存関係
*   **Python**: 3.x

*   **Image Processing**: `Pillow` (PIL), `numpy`。
*   **Video Processing**: `ffmpeg` (コマンドラインツール) を `subprocess` 経由で呼び出し。

### 1.2. ファイル構成

#### `common.py` (共通モジュール)
*   **概要**: 設定値、ヘッダ処理、nibble変換、フレーム処理など、共通機能を提供する。
*   **主な定数**: `LOGICAL_WIDTH`, `LOGICAL_HEIGHT`, `BLOCK_SIZE`, `BYTES_PER_FRAME` など。
*   **主な関数**: `build_plaintext_block()`, `parse_inner_header()`, `nibbles_from_bytes()`, `precompute_data_indices()` など。

#### `screensaver.py` (スクリーンセーバーCLI)
*   **概要**: データをリアルタイムで画面に描画する。旧 `bake` サブコマンド相当。
*   **GUI**: `tkinter` を使用してフルスクリーンウィンドウを作成。
*   **描画ループ**: `numpy` でフレームデータを生成し、`PIL.Image` -> `ImageTk.PhotoImage` を経由して `tk.Label` に設定。
*   **キーフレーム**: 開始時に緑ベタ塗り、終了時に赤ベタ塗りを各1秒表示。
*   **使用方法**: `python screensaver.py -i file.txt [--fps 15]`

#### `decode.py` (デコードCLI)
*   **概要**: 動画ファイルを入力とし、元のファイルを復元する。旧 `taste` サブコマンド相当。
*   **処理フロー**:
    1. `ffmpeg` で全フレームをPNG画像として一時ディレクトリに抽出。
    2. **キーフレーム検出**: 緑/赤のベタ塗りフレームを自動検出。
    3. キーフレーム間のフレームのみを処理対象とする。
    4. 画像を `LOGICAL_WIDTH` x `LOGICAL_HEIGHT` に *Nearest Neighbor* でリサイズ。
    5. 各ピクセルからニブル値を抽出。
    6. `REPETITIONS` 分の値を平均化してニブルを確定。
    7. ニブルを結合してデータを復元する。
    8. ヘッダ解析とハッシュ検証を経てファイルを保存。
*   **使用方法**: `python decode.py -i output.mp4 [-d ./output]`

すべての機能は `spec.md` の仕様 (960x540, Block=2, Rep=2) に準拠。

---

## 2. 変更履歴

### 2026-01-30: キーフレーム検出機能
データフレームの前後に開始/終了キーフレーム（緑/赤ベタ塗り、各1秒）を追加。decode側でRGBチャンネル平均値による自動検出を実装し、キーフレーム外の動画内容を無視するようになった。

### 2026-01-16: 独立CLIツールへの分割
`bakery.py` を3つのファイルに分割。`common.py`（共通モジュール）、`screensaver.py`（スクリーンセーバーCLI）、`decode.py`（デコードCLI）として独立したツールに再構成。

### 2026-01-05: 動画エンコード機能の廃止
`encode` サブコマンドを廃止。ファイルからMP4動画への変換機能を削除し、`bake`（リアルタイム再生）と`taste`（動画からの復元）のみを提供。コマンド名を料理メタファーに変更。

### 2026-01-05: 暗号化機能の廃止
AES-256-CTR による暗号化機能を廃止。`cryptography` ライブラリへの依存を削除し、パスワード入力が不要になった。

### 2026-01-01: `bakery.py` への統合
従来の分離スクリプト (`screensaver.py`, `decode.py`, `encode.py`, `lib/common.py`) を単一の `bakery.py` に統合。CLIサブコマンド (`bake`, `taste`) で各機能にアクセスする構成に変更。

### 2024-12-31: `encode.py` の仕様不整合修正
以前の実装では、`encode.py` が他のコンポーネントと異なる解像度とプロトコルを使用しており、生成された動画がデコードできない問題が存在した。

*   **修正内容**:
    *   `LOGICAL_WIDTH`: 480 -> **960**
    *   `LOGICAL_HEIGHT`: 270 -> **540**
    *   `BLOCK_SIZE`: 4 -> **2** (2x2)
    *   `REPETITIONS`: 3 -> **2**
