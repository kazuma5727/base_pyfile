# Changelog

このファイルは `base_pyfile` の主な変更を記録します。
バージョン番号の決め方は [VERSIONING.md](VERSIONING.md) を参照してください。

## [Unreleased]

### Added

- `VERSIONING.md` を追加。SemVer の判断基準、リリース手順、AI/エージェント向けの指示例を記載。
- `CHANGELOG.md` を追加。
- `tests/test_versioning.py` を追加。現在の `__version__` が changelog に記載されているかを CI で確認する。

### Changed

- `AGENTS.md` / `README.md` にバージョン方針への導線を追加。
- `AGENTS.md` に Git 運用（コミット後の push 同期）を追記。
- sdist に `VERSIONING.md` / `CHANGELOG.md` / `AGENTS.md` を含める。

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
