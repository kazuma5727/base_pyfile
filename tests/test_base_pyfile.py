"""base_pyfile の安定版モジュール向け回帰テスト。

開発中 AI（base_pyfile/dev/）のテストは tests/test_dev_ai.py にある。
重い依存（cv2 / pyautogui / fitz など）はここでは読み込まない。
Ollama など外部サービスも使わない。
"""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import base_pyfile
from base_pyfile import function_timer
from base_pyfile.file_manager import read_text_file, write_file
from base_pyfile.ini_filemanager import read_config, save_config
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import make_directory, reset_existing_files, unique_path


# ------------------------------------------------------------
# package / public API
# ------------------------------------------------------------
def test_public_api_is_registered_and_unique():
    """__all__ の名前は重複せず、遅延importの対応表にも登録されていること。"""
    assert len(base_pyfile.__all__) == len(set(base_pyfile.__all__))
    assert set(base_pyfile.__all__) <= set(base_pyfile._MODULE_OF)


def test_ini_functions_are_available_from_top_level():
    """ini_filemanager の主要関数を base_pyfile 直下から使えること。"""
    assert base_pyfile.read_config is read_config
    assert base_pyfile.save_config is save_config


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", base_pyfile.__version__)


def test_package_data_files_exist():
    """安定版の設定INIが同梱され、開発中JSONは直下に混ざっていないこと。"""
    package_dir = Path(base_pyfile.__file__).parent
    assert (package_dir / "settings.ini").is_file()
    assert not (package_dir / "ai_models.json").is_file()


def test_importing_package_does_not_load_heavy_dependencies():
    """import base_pyfile だけで重い依存やAI用依存を読み込まないこと。"""
    code = (
        "import base_pyfile, sys; "
        "print(','.join(sorted(m for m in "
        "('cv2', 'fitz', 'pyautogui', 'pynput', 'bs4', 'requests', 'pyperclip') "
        "if m in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""


# ------------------------------------------------------------
# log_setting
# ------------------------------------------------------------
def test_get_log_handler_with_missing_file_path(tmp_path):
    """file_path が実在しなくても UnboundLocalError にならないこと。"""
    handler = get_log_handler(
        log_level=10, file_path=tmp_path / "not_created.py", log_folder="logs"
    )
    try:
        assert (tmp_path / "logs").is_dir()
    finally:
        handler.close()


def test_get_log_handler_unknown_level(tmp_path):
    """標準外のログレベルでも KeyError にならないこと。"""
    handler = get_log_handler(
        log_level=25, file_path=tmp_path / "prog.py", log_folder="logs"
    )
    try:
        assert (tmp_path / "logs" / "LEVEL25_prog.log").exists()
    finally:
        handler.close()


def test_make_logger_does_not_duplicate_handlers():
    """同名ロガーを再設定してもハンドラが積み重ならないこと。"""
    name = "test_dup_logger_regression"
    first = make_logger(name)
    second = make_logger(name)
    assert first is second
    assert len(second.handlers) == 1


# ------------------------------------------------------------
# function_timer
# ------------------------------------------------------------
def test_logger_timer_rejects_zero():
    with pytest.raises(ValueError):
        function_timer.logger_timer(n=0)


def test_timer_returns_result():
    @function_timer.timer
    def double(x):
        return x * 2

    assert double(3) == 6


# ------------------------------------------------------------
# path_manager
# ------------------------------------------------------------
def test_unique_path_increments_on_collision(tmp_path):
    reset_existing_files()
    target = tmp_path / "a{}.txt"

    first = unique_path(str(target))
    assert Path(first).name == "a1.txt"

    # 1つ目を使い切った状態にして、次の番号が振られること
    Path(first).write_text("x", encoding="utf-8")
    second = unique_path(str(target))
    assert Path(second).name == "a2.txt"


def test_reset_existing_files(tmp_path):
    reset_existing_files()
    target = tmp_path / "b{}.txt"
    first = unique_path(str(target))

    reset_existing_files()
    after_reset = unique_path(str(target))
    assert Path(first).name == Path(after_reset).name == "b1.txt"


def test_make_directory_creates_and_returns(tmp_path):
    target = tmp_path / "new_dir"
    result = make_directory(target)
    assert target.is_dir()
    assert Path(result) == target


def test_make_directory_handles_dotted_name(tmp_path):
    """ドット入りのディレクトリ名をファイルと誤判定しないこと。"""
    target = tmp_path / "folder.with.dot"
    make_directory(target)
    assert target.is_dir()


def test_make_directory_recreates_after_deletion(tmp_path):
    """作成後に削除しても、再度呼べば作り直せること。"""
    target = tmp_path / "recreate"
    make_directory(target)
    target.rmdir()
    make_directory(target)
    assert target.is_dir()


def test_make_directory_file_mode_uses_parent(tmp_path):
    target = tmp_path / "folder.with.dot" / "memo.txt"
    make_directory(target, is_file=True)
    assert target.parent.is_dir()
    assert not target.exists()


# ------------------------------------------------------------
# file_manager
# ------------------------------------------------------------
def test_write_read_roundtrip_utf8(tmp_path):
    file_path = tmp_path / "hello.txt"
    write_file(file_path, "こんにちは", backup=False)
    assert read_text_file(file_path) == "こんにちは"


def test_write_read_roundtrip_sjis(tmp_path):
    file_path = tmp_path / "sjis.txt"
    write_file(file_path, "テスト", file_encoding="shift_jis", backup=False)
    assert read_text_file(file_path) == "テスト"


def test_write_file_keeps_explicit_extension(tmp_path):
    """明示的な拡張子を勝手に .txt へ書き換えないこと。"""
    file_path = tmp_path / "data.json"
    returned = write_file(file_path, "{}", backup=False)
    assert returned.suffix == ".json"
    assert file_path.exists()


def test_read_text_file_missing_returns_empty(tmp_path):
    assert read_text_file(tmp_path / "missing.txt") == ""


def test_write_file_creates_dotted_parent_directory(tmp_path):
    file_path = tmp_path / "folder.with.dot" / "memo.txt"
    write_file(file_path, "hello", backup=False)
    assert file_path.read_text(encoding="utf-8") == "hello"


def test_write_file_raises_on_write_error(tmp_path, monkeypatch):
    """書き込み失敗を成功したかのように握りつぶさないこと。"""
    target = tmp_path / "fail.txt"
    original_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == target:
            raise OSError("write denied")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match="write denied"):
        write_file(target, "x", backup=False)


def test_read_text_file_removes_utf8_bom(tmp_path):
    file_path = tmp_path / "bom.txt"
    file_path.write_bytes(b"\xef\xbb\xbfhello")
    assert read_text_file(file_path) == "hello"


# ------------------------------------------------------------
# ini_filemanager
# ------------------------------------------------------------
def test_read_config_reads_default_section(tmp_path):
    ini_file = tmp_path / "settings.ini"
    ini_file.write_text(
        "[DEFAULT]\n"
        "width = 1920\n"
        "ratio = 1.5\n"
        "auto_mode = True\n"
        "negative = -2\n",
        encoding="utf-8",
    )

    obj = SimpleNamespace()
    read_config(obj, ini_file)

    assert obj.width == 1920
    assert isinstance(obj.width, int)
    assert obj.ratio == 1.5
    assert obj.auto_mode is True
    assert obj.negative == -2
    assert isinstance(obj.negative, int)


def test_save_config_roundtrip(tmp_path):
    ini_file = tmp_path / "settings.ini"
    obj = SimpleNamespace(width=1920, height=1080, auto_mode=True, ratio=1.5)
    save_config(obj, ini_file)

    loaded = SimpleNamespace()
    read_config(loaded, ini_file)

    assert loaded.width == 1920
    assert loaded.height == 1080
    assert loaded.auto_mode is True
    assert loaded.ratio == 1.5
