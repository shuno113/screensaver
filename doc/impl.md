# Bakery 実装詳細 (Implementation Details)

本ドキュメントでは、現在のPythonスクリプトの実装詳細、ライブラリ依存関係、および既知の問題について記述する。

## 1. コンポーネント実装

### 1.1. 共通ライブラリと依存関係
*   **Python**: 3.x
*   **Image Processing**: `numpy`
*   **Video Processing**: `ffmpeg` (コマンドラインツール) を `subprocess` 経由で呼び出し
*   **GUI**: `pygame` (フルスクリーン描画)
*   **最適化**: `numba` (オプション、JIT高速化)

### 1.2. ファイル構成

#### `common.py` (共通モジュール)
*   **概要**: 設定値、ヘッダ処理、nibble変換、フレーム処理など、共通機能を提供する。
*   **主な定数**: `PHYSICAL_WIDTH`, `PHYSICAL_HEIGHT`, `BLOCK_SIZE`, `REPETITIONS`, `HEADER_SIZE`, `FRAME_HEADER_SIZE` など。
*   **主な関数**: 
    *   `configure_block_size()`: ブロックサイズを動的に設定
    *   `build_inner_header()`, `parse_inner_header()`: FVD2ヘッダ構築・解析
    *   `build_frame_header()`, `parse_frame_header()`, `verify_frame_header()`: フレームヘッダ処理
    *   `nibbles_from_bytes()`, `gray_to_nibble()`: nibble変換
    *   `precompute_data_indices()`, `generate_frame_array()`: フレーム処理

#### `screensaver.py` (スクリーンセーバーCLI)
*   **概要**: データをリアルタイムで画面に描画する。
*   **GUI**: `pygame` を使用してフルスクリーンウィンドウを作成。
*   **描画方式**: ストリーミング方式。バックグラウンドスレッドでフレームを事前生成し、キューに追加。
*   **キーフレーム**: 開始時に緑ベタ塗り、終了時に赤ベタ塗りを表示（デフォルト各3秒）。
*   **使用方法**:
    ```
    python screensaver.py -i file.txt
    python screensaver.py -i file.txt --fps 60 --repetitions 1 --block-size 2
    ```
*   **CLIオプション**:
    *   `-i, --input`: 入力ファイル（必須）
    *   `--fps`: フレームレート（デフォルト: 10）
    *   `--repetitions`: nibbleの繰り返し回数（デフォルト: 1）
    *   `--block-size`: ブロックサイズ（デフォルト: 1）
    *   `--countdown`: 開始前カウントダウン秒数（デフォルト: 5）
    *   `--key-duration`: キーフレーム表示秒数（デフォルト: 3）

#### `decode.py` (デコードCLI)
*   **概要**: 動画ファイルを入力とし、元のファイルを復元する。
*   **処理フロー**:
    1. `ffmpeg` から rawvideo を pipe 経由でストリーミング取得（PNG一時保存なし）。
    2. **キーフレーム検出**: 緑/赤のベタ塗りフレームを自動検出。
    3. **遷移フレームスキップ**: キーフレーム直後の不安定フレームを自動スキップ。
    4. **ブロック平均リサイズ**: 物理解像度から論理解像度への変換。
    5. **重複フレーム検出**: 前フレームと同一内容ならスキップ（60fps録画対応）。
    6. **輝度補正**: 4コーナーの最大値を基準にスケーリング。
    7. 各ピクセルからニブル値を抽出。
    8. **フレームヘッダ検証**: 連番の連続性とCRC32を確認。
    9. ヘッダ解析とSHA-256ハッシュ検証を経てファイルを保存。
*   **最適化**:
    *   **シングルパス・ストリーミング処理**: メモリ使用量削減、高速化
    *   **Numba JIT**: ホットパス（RGB→グレー変換、ブロック平均、nibble抽出）を高速化
    *   **GPUデコード**: macOS videotoolbox 対応
*   **使用方法**:
    ```
    python decode.py -i output.mp4
    python decode.py -i output.mp4 --gpu
    ```
*   **CLIオプション**:
    *   `-i, --input`: 入力動画ファイル（必須）
    *   `-d, --output-dir`: 出力ディレクトリ（デフォルト: カレント）
    *   `--gpu`: GPUデコード有効化 (macOS videotoolbox)
    *   `--no-dedup`: 重複フレーム検出を無効化

---

## 2. 変更履歴

### 2026-01-30: シングルパス・ストリーミング処理への移行
decode.py を PNG 一時保存方式からシングルパス・ストリーミング処理に移行。ffmpeg から rawvideo を pipe 経由で受け取り、メモリ使用量を大幅に削減。Numba JIT によるホットパス高速化、GPUデコード (videotoolbox) 対応を追加。

### 2026-01-30: フレームヘッダ仕様の導入
各データフレームに 8 バイトのヘッダ（連番 4B + CRC32 4B）を埋め込み。フレーム落ちおよびデータ破損を厳密に検出可能に。

### 2026-01-30: FVD2ヘッダ形式への移行
ファイルヘッダを FVD2 形式に更新。エンコーディングパラメータ（frame_count, nibble_count, repetitions, block_size, fps）を 256 バイトのヘッダ領域に格納し、デコード時のパラメータ自動認識を実現。

### 2026-01-30: デコード処理の堅牢化
外部キャプチャ環境（OBS + HDMIキャプチャカード）でのデコード信頼性を大幅に向上。
*   **遷移フレームスキップ**: キーフレーム直後の不安定フレームを自動検出してスキップ。
*   **ブロック平均リサイズ**: Nearest Neighborからブロック平均法に変更し、ティアリング影響を軽減。
*   **重複フレーム検出**: 前フレームと同一内容のフレームをスキップ（60fps録画対応）。
*   **max-corner輝度補正**: 4コーナーの最大値を白レベル基準として全体をスケーリング。

### 2026-01-30: キーフレーム検出機能
データフレームの前後に開始/終了キーフレーム（緑/赤ベタ塗り）を追加。decode側でRGBチャンネル平均値による自動検出を実装し、キーフレーム外の動画内容を無視するようになった。

### 2026-01-30: pygame への移行
screensaver.py のGUIを tkinter から pygame に移行。ストリーミング方式でバックグラウンドスレッドがフレームを事前生成し、高速なフルスクリーン描画を実現。

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

