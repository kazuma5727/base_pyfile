"""base_pyfile の軽量モジュール向け回帰テスト。

重い依存（cv2 / pyautogui / fitz など）が必要なモジュールは
ここでは読み込まない。Ollama など外部サービスも使わない。
"""

from pathlib import Path

import pytest

from base_pyfile import ai_clipboard, ai_corrector, ai_translator, function_timer
from base_pyfile.ai_router import _match_model_name, _parse_label, build_label_map
from base_pyfile.file_manager import read_text_file, write_file
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import make_directory, reset_existing_files, unique_path


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


# ------------------------------------------------------------
# ai_router（純粋関数のみ。Ollamaには接続しない）
# ------------------------------------------------------------
def test_build_label_map_and_parse_label():
    candidates = [{"name": "a:1b"}, {"name": "b:2b"}]
    label_map = build_label_map(candidates)

    assert label_map["A"] is None  # A は常に「わからない」
    assert label_map["B"] == "a:1b"
    assert label_map["C"] == "b:2b"

    assert _parse_label("B", label_map) == "B"
    assert _parse_label("答えは C です", label_map) == "C"
    assert _parse_label("", label_map) is None
    assert _parse_label("XYZ", label_map) is None


def test_match_model_name():
    candidates = [{"name": "qwen2.5-coder:3b"}, {"name": "deepseek-r1:1.5b"}]
    assert _match_model_name("qwen2.5-coder:3b", candidates) == "qwen2.5-coder:3b"
    # タグ省略でも一致させる
    assert _match_model_name("使うのは deepseek-r1 です", candidates) == "deepseek-r1:1.5b"
    assert _match_model_name("unknown", candidates) is None


# ------------------------------------------------------------
# ai_corrector（ネットワークを使わず、generateはモンキーパッチ）
# ------------------------------------------------------------
PROFILES_FOR_CORRECTION = {
    "router": {"default_target": "big"},
    "correction": {
        "min_chars": 1,
        "tiers": [
            {"max_chars": 10, "model": "small"},
            {"max_chars": None, "model": "big"},
        ],
        "fallback": ["mid"],
    },
}


def test_select_model_by_length():
    installed = ["small", "mid", "big"]
    assert (
        ai_corrector.select_model_by_length("short", PROFILES_FOR_CORRECTION, installed)
        == "small"
    )
    assert (
        ai_corrector.select_model_by_length(
            "x" * 50, PROFILES_FOR_CORRECTION, installed
        )
        == "big"
    )


def test_select_model_by_length_falls_back_when_missing():
    # small が未インストールなので fallback の mid が選ばれる
    assert (
        ai_corrector.select_model_by_length(
            "short", PROFILES_FOR_CORRECTION, ["mid"]
        )
        == "mid"
    )


def test_clean_model_output():
    assert ai_corrector._clean_model_output("```\nhello\n```") == "hello"
    assert ai_corrector._clean_model_output('"hello"') == "hello"
    assert ai_corrector._clean_model_output("hello") == "hello"


def test_correct_text_returns_corrected(monkeypatch):
    monkeypatch.setattr(ai_corrector, "generate", lambda *a, **k: ("なおした", 12.0))
    result = ai_corrector.correct_text(
        "なおして", profiles=PROFILES_FOR_CORRECTION, model="dummy"
    )
    assert result["corrected"] == "なおした"
    assert result["changed"] is True
    assert result["model"] == "dummy"


def test_correct_text_keeps_original_on_failure(monkeypatch):
    monkeypatch.setattr(ai_corrector, "generate", lambda *a, **k: (None, 5.0))
    result = ai_corrector.correct_text(
        "そのまま", profiles=PROFILES_FOR_CORRECTION, model="dummy"
    )
    assert result["corrected"] == "そのまま"
    assert result["changed"] is False


def test_correct_text_empty_text():
    result = ai_corrector.correct_text("   ", profiles=PROFILES_FOR_CORRECTION)
    assert result["corrected"] == "   "
    assert result["changed"] is False


# ------------------------------------------------------------
# ai_translator（ネットワークを使わず、generateはモンキーパッチ）
# ------------------------------------------------------------
PROFILES_FOR_TRANSLATION = {
    "router": {"default_target": "mid"},
    "translation": {
        "min_chars": 1,
        "target_language": "ja",
        "tiers": [
            {"max_chars": 10, "model": "small"},
            {"max_chars": None, "model": "big"},
        ],
        "fallback": ["mid"],
    },
}


def test_language_label():
    assert ai_translator.language_label("ja") == "日本語"
    assert ai_translator.language_label("EN") == "英語"
    assert ai_translator.language_label("Klingon") == "Klingon"


def test_build_translation_prompt_mentions_target():
    prompt = ai_translator.build_translation_prompt("hello", "ja")
    assert "日本語" in prompt
    assert "hello" in prompt


def test_translate_select_model_by_length():
    installed = ["small", "mid", "big"]
    assert (
        ai_translator.select_model_by_length(
            "short", PROFILES_FOR_TRANSLATION, installed, section="translation"
        )
        == "small"
    )
    assert (
        ai_translator.select_model_by_length(
            "x" * 50, PROFILES_FOR_TRANSLATION, installed, section="translation"
        )
        == "big"
    )


def test_translate_text_returns_translated(monkeypatch):
    monkeypatch.setattr(ai_translator, "generate", lambda *a, **k: ("こんにちは", 9.0))
    result = ai_translator.translate_text(
        "hello", profiles=PROFILES_FOR_TRANSLATION, model="dummy"
    )
    assert result["translated"] == "こんにちは"
    assert result["changed"] is True
    assert result["model"] == "dummy"
    assert result["target_language"] == "ja"


def test_translate_text_keeps_original_on_failure(monkeypatch):
    monkeypatch.setattr(ai_translator, "generate", lambda *a, **k: (None, 3.0))
    result = ai_translator.translate_text(
        "hello", profiles=PROFILES_FOR_TRANSLATION, model="dummy"
    )
    assert result["translated"] == "hello"
    assert result["changed"] is False


def test_ai_clipboard_translate_replacement():
    assert ai_clipboard._replacement_of({"translated": "x"}, "translate") == "x"
    assert ai_clipboard._replacement_of({"corrected": "y"}, "correct") == "y"
    assert ai_clipboard._replacement_of({"answer": "z"}, "answer") == "z"
