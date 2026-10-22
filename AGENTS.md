# AGENTS.md

`base_pyfile` は Windows 向けの Python ユーティリティ集。README.md 冒頭の「安定版と開発中機能」と「AI・エージェント向けクイックリファレンス」が現行の入口。

## 構成（安定版 / 開発中）

- `base_pyfile/` … 安定版。通常の `pip install .` にはこちらだけが入る。統合CLIは `base_pyfile/cli.py` と `base_pyfile/__main__.py`（`base-pyfile` / `python -m base_pyfile`）。
- `base_pyfile/dev/` … 開発中の AI（`ai_router` / `ai_clipboard` / `ai_corrector` / `ai_translator` / `modle_edit` / `ai_models.json` / `ai_models_5060ti.json`）。wheel には含めない（`pyproject.toml` の `packages.find.exclude = ["base_pyfile.dev*"]`）。
- リポジトリ直下や `pip install -e .` では `base_pyfile.dev` が存在するため、`base_pyfile/__init__.py` が dev の公開名を自動マージする。`from base_pyfile import dispatch` の後方互換はこの仕組みで維持している。
- 安定版の `__all__` / `_MODULE_OF` に dev の名前を直接書かない。dev の公開名を増やす場合は `base_pyfile/dev/__init__.py` の `__all__` と `_MODULE_OF` を更新する。
- dev 内の相互参照は `from base_pyfile.dev.xxx import yyy`。安定版（`log_setting` など）への参照は `from base_pyfile.xxx import yyy` のまま。
- `base_pyfile/QRreader.py` はリポジトリ直下の `QRreader.py`（独立スクリプト）へ移動済み。パッケージに戻さない。

## 検証

- `python -m pytest tests -q`（安定版: `tests/test_base_pyfile.py`、統合CLI: `tests/test_cli.py`、開発中AI: `tests/test_dev_ai.py` / `tests/test_ai_clipboard.py`、バージョン方針: `tests/test_versioning.py`）
- テストは軽量・オフライン前提。Ollama、cv2、PyMuPDF などを要求するテストは追加しない。`ai_router.generate` や PDF 変換関数は monkeypatch する。
- GitHub Actions（`.github/workflows/tests.yml`）が Windows / Python 3.10・3.12 で editable インストールして pytest を実行する。lint / typecheck は無い。
- `package` ジョブが sdist / wheel をビルドし、`twine check` と `scripts/check_dist.py` でメタデータ・同梱内容を検証する。

## 公開 API と遅延 import

- 安定版の公開関数を追加するときは `base_pyfile/__init__.py` の `_MODULE_OF` と `__all__` に登録する。`__version__` は `base_pyfile/_version.py` の固定値。公開名の増減だけでは変えず、バージョンの上げ方は `VERSIONING.md` の SemVer 基準に従う。
- 統合CLIのサブコマンドやオプションも公開 API として扱う。`base_pyfile/cli.py` の変更は `README.md` と `CHANGELOG.md` を更新し、`tests/test_cli.py` を追加・修正する。コンソールスクリプトは `pyproject.toml` の `[project.scripts]` が正。
- パッケージの import 時に重い依存を読まない。cv2 / pyautogui / fitz / bs4 などは各モジュール内か関数内で遅延 import する。`base_pyfile/cli.py` は `base_pyfile.dev` を `ai` サブコマンド実行時だけ import する。
- 設定ファイルを追加したら `pyproject.toml` の `[tool.setuptools.package-data]` にも登録する。テストが同梱を確認している。
- `run_agent` / `run_repl` は未実装のため公開 API から外している。実装するまで戻さない。

## バージョンとリリース

- `__version__` は `base_pyfile/_version.py` の固定値。`__all__` の件数とは連動しない。
- バージョンはリリース単位。ドキュメント・テスト・CI だけの変更は `_version.py` を変えず、`CHANGELOG.md` の `[Unreleased]` に追記する。
- 公開 API を変更したら `VERSIONING.md` の基準で MAJOR / MINOR / PATCH を判定する。
  - 破壊的変更: MAJOR
  - 後方互換のある追加: MINOR
  - 修正のみ: PATCH
- リリース時は `_version.py` と `CHANGELOG.md` を同じコミットで更新する。`tests/test_versioning.py` が現在の `__version__` の changelog 記載を確認する。
- `base_pyfile.dev` は実験的で SemVer 対象外。dev だけの変更では安定版のバージョンを上げない。
- 迷ったら「既存の公開 API 利用コードが変更なしで動くか」で判断する。動かないなら MAJOR。
- AI/エージェントは作業前に `VERSIONING.md` を読む。外部AIに渡すプロンプト例も同ファイルにある。

## Git 運用

- コミットしたら、原則として `git push` まで実行して `origin` と同期する。
- push 前に `git fetch` と `git status -sb` でリモートとの差分を確認する。
- push できない場合（認証・権限・レビュー待ちなど）は、その理由を報告する。

## 設定・実行

- `base_pyfile/dev/ai_models.json` がCPU向け（既定）、`base_pyfile/dev/ai_models_5060ti.json` がRTX 5060 Ti向けの Ollama ルーティング・校正・翻訳・agent 設定。モデル追加は各JSONの `models` だけでよい。既定はローカルのみで、クラウドフォールバックは使わない。
- モデル構成は `BASE_PYFILE_AI_MODELS=cpu|5060ti`（`gpu` も可）または `--profile cpu|5060ti` で切り替える。`ai_router._model_profile_path()` が解決し、存在しない場合はCPU既定へ戻る。
- `ai_router` は `http://localhost:11434` 固定。生成の timeout は 120 秒（`GENERATE_TIMEOUT`）。
- 統合CLI は `base-pyfile <command>`（`python -m base_pyfile <command>` も同じ）。開発中AI は `base-pyfile ai [answer|correct|translate] [lang]` または `python -m base_pyfile.dev.ai_clipboard ...`。`ai_clipboard.bat` の `-honyaku` / `-gojidatuji` / `-kaitou` はこの CLI の薄いラッパーで、2つ目以降の引数（`--preview` / `--undo` / `--hotkey` など）をそのまま渡す。
- パッケージ定義は `pyproject.toml` が正。`setup.py` は旧ツール用の shim。コア依存は natsort / requests / pyperclip のみで、重い依存は extras（`automation` / `web` / `pdf` / `all` / `dev`）で選ぶ。
- `requirements.txt` は開発用に `-e .[all,dev]` を参照する。依存を追加するときは `pyproject.toml` 側を更新する。
- `QRreader.py` は Windows 専用の独立スクリプト。`Pillow` と `pyzbar` が必要で、パッケージの依存には含めない。
