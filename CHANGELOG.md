# Changelog

このファイルは `base_pyfile` の主な変更を記録します。
バージョン番号の決め方は [VERSIONING.md](VERSIONING.md) を参照してください。

## [Unreleased]

### Added

- 開発中AIのOllamaモデル構成を、CPU向け（`ai_models.json`、既定）とRTX 5060 Ti向け（`ai_models_5060ti.json`）の2プロファイルに分割。`BASE_PYFILE_AI_MODELS=cpu|5060ti` または `ai_clipboard --profile cpu|5060ti` で切り替えられる。
- 両プロファイルともクラウドモデルを使わず、ローカルOllamaのみで動作するようにした。CPU版は既存の軽量モデルで動作し、GPU版は8GBでは7B前後、16GBでは14B前後を優先する。

### Changed

- `ai_clipboard` に `--profile` オプションを追加。`base-pyfile ai ...` と `.bat` ランチャーからも指定できる。
- `README.md` / `AGENTS.md` にモデル構成の切り替え方とOllamaモデルの導入例を追記。
- パッケージ・テスト・スクリプト・リポジトリ直下の独立Pythonファイルを Black / isort（black プロファイル）で整形。あわせて未使用 import を整理した。
- `get_all_subfolders()` の `depth` を「ルート直下を1階層」と数えるように修正。`depth=1` が子の子まで返していた off-by-one を解消した（`depth=0` は従来どおり直下のみ）。

### Fixed

- `is_model_installed()` が `qwen2.5:1.5b` を `qwen2.5:0.5b` のインストール済み判定に巻き込んでいた問題を修正。タグを明示したモデルは完全一致で判定し、タグ省略時だけファミリー一致を許可する。
- `unique_path("a{}.txt")` と `a.txt` が共存すると無限ループしていた問題を修正。テンプレートの候補以外（`a.txt` や `a_.txt`）を存在確認に含めず、通常ファイル名には衝突時だけ接尾辞を付けるようにした。`existing_image` に NumPy 配列を渡したときの真偽値評価エラーも修正。
- `sanitize_windows_filename()` が相対パスの親ディレクトリを重複させていた問題を修正。ファイル名の途中のドット、拡張子内の不正文字、末尾のドット・空白も正規化する。
- `read_config()` で `[DEFAULT]` の継承値が後続セクションの値を上書きしていた問題を修正。セクション固有の値だけを適用する。
- `automation_tools.learning_materials()` の invalid escape sequence (`\m`) を修正。`probability < 1` と画像読み込み失敗を明確な `ValueError` にした。
- `pdf_tiff_converter` で変換途中に例外が起きた場合も、作成済みの PDF/TIFF 画像リソースを解放するようにした。未対応拡張子の検証を出力フォルダ作成より先に行う。

## [4.11.0] - 2026-10-22

### Added

- 統合CLI `base-pyfile`（`python -m base_pyfile`）を追加。`doctor` / `run` / `open` / `pdf-to-png` / `tiff-to-pdf` / `image-to-pdf` / `ai` / `version` サブコマンドを提供する。
- `pdf_to_png()` / `convert_to_png()` に `dpi` オプションを追加。CLI の `pdf-to-png --dpi` から指定できる。
- 開発中AIのクリップボード処理に、`--preview`（差分確認）、`--undo` / `--history`（履歴・アンドゥ）、`--hotkey`（グローバルホットキー）を追加。履歴は既定で最新100件を保存する。
- `watch_hotkey()` / `undo_last()` / `load_history()` / `format_diff()` を `base_pyfile.dev` の公開名に追加。
- `tests/test_cli.py` / `tests/test_ai_clipboard.py` を追加。`scripts/check_dist.py` で console script と新しい同梱ファイルを検証する。
- `VERSIONING.md` を追加。SemVer の判断基準、リリース手順、AI/エージェント向けの指示例を記載。
- `CHANGELOG.md` を追加。
- `tests/test_versioning.py` を追加。現在の `__version__` が changelog に記載されているかを CI で確認する。
- `scripts/check_dist.py` を追加。sdist / wheel の同梱内容とバージョンの一致を検証する。
- `tests/test_versioning.py` に changelog の見出し形式・日付・並び順・重複の検証を追加。

### Changed

- `base_pyfile.dev.ai_clipboard` の CLI を argparse ベースに変更。従来の `python -m base_pyfile.dev.ai_clipboard [mode] [lang]` はそのまま使える。
- `ai_clipboard.bat` が2つ目以降の引数をPython側へ渡すよう変更。`--preview` / `--once` / `--hotkey` などを利用できる。
- `AGENTS.md` / `README.md` にバージョン方針と統合CLI・クリップボード安全機能への導線を追加。
- `AGENTS.md` に Git 運用（コミット後の push 同期）を追記。
- GitHub Actions に `package` ジョブを追加し、ビルド成果物を検証する。
- sdist に `VERSIONING.md` / `CHANGELOG.md` / `AGENTS.md` / `scripts/check_dist.py` を含める。

## [4.10.0] - 2026-10-21

### Added

- `doctor()` と `python -m base_pyfile.diagnostics` を追加。実行環境・任意依存・利用可能な機能を JSON で確認できる。
- `run_command()` と `CommandResult` を追加。外部コマンドの出力を UTF-8 / CP932 自動判別で取得できる。
- `atomic_write()` を追加。一時ファイル + `os.replace` で安全にファイルを書き換え、Windows の一時的な `PermissionError` をリトライする。

### Fixed

- `run_command()` の CP932 判定を英語 Windows でも安定するよう修正。
- `doctor()` が OpenCV の contrib 版（`opencv-contrib-python`）も検出するよう修正。

## [4.9.1] - 2026-10-21

### Changed

- `specified_color()` / `specified_color_fast_ver()` / `templates_matching()` の `found=True` は bool のみを返すよう統一。座標取得には `found=False`（既定）を使う。

## [4.9.0] - 2026-10-20

### Fixed

- `make_directory()` の既定動作、`write_file()` の失敗時 `OSError`、`get_urls()` の絶対 URL 化、PDF → PNG の連番、ini の `[DEFAULT]`、QRreader の http(s) 限定など、安定版ユーティリティの不具合を修正。
- `automation_tools` のマウス位置既定値、`log_setting` のハンドラ置き換え、開発中 AI の翻訳・クリップボード処理を修正。

## [4.8.0] - 2026-10-20

### Changed

- 安定版 `base_pyfile/` と開発中 AI `base_pyfile/dev/` を分離。通常の `pip install .` の wheel には dev を含めない。
- `QRreader.py` をパッケージ外の独立スクリプトへ移動。
- バージョン管理を `__all__` の件数からの自動計算から、`base_pyfile/_version.py` の固定値へ変更。
