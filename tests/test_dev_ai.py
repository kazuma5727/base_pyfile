"""base_pyfile/dev/ の開発中 AI 機能向け回帰テスト。

ネットワークを使わず、`generate` などはモンキーパッチする。
Ollama など外部サービスも使わない。
"""

from pathlib import Path

import pytest

import base_pyfile
from base_pyfile import dev
from base_pyfile.dev import ai_clipboard, ai_corrector, ai_router, ai_translator
from base_pyfile.dev.ai_router import _match_model_name, _parse_label, build_label_map


# ------------------------------------------------------------
# dev package / public API
# ------------------------------------------------------------
def test_dev_public_api_is_registered_and_unique():
    assert len(dev.__all__) == len(set(dev.__all__))
    assert set(dev.__all__) <= set(dev._MODULE_OF)


def test_dev_package_data_exists():
    package_dir = Path(base_pyfile.__file__).parent
    assert (package_dir / "dev" / "ai_models.json").is_file()


def test_top_level_keeps_backward_compatibility():
    """リポジトリ/editable では base_pyfile 直下からも AI 機能を使えること。"""
    assert base_pyfile.dispatch is ai_router.dispatch
    assert base_pyfile.route_to_model is ai_router.route_to_model
    assert base_pyfile.correct_text is ai_corrector.correct_text
    assert base_pyfile.translate_text is ai_translator.translate_text


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


def test_translate_text_uses_profile_target_language(monkeypatch):
    """target_language 省略時は ai_models.json 側の設定を使うこと。"""
    profiles = {
        **PROFILES_FOR_TRANSLATION,
        "translation": {
            **PROFILES_FOR_TRANSLATION["translation"],
            "target_language": "en",
        },
    }
    prompts = []

    def fake_generate(prompt, **kwargs):
        prompts.append(prompt)
        return "hello", 1.0

    monkeypatch.setattr(ai_translator, "generate", fake_generate)
    result = ai_translator.translate_text(
        "こんにちは", profiles=profiles, model="dummy"
    )

    assert result["target_language"] == "en"
    assert "英語" in prompts[0]


def test_ai_clipboard_translate_replacement():
    assert ai_clipboard._replacement_of({"translated": "x"}, "translate") == "x"
    assert ai_clipboard._replacement_of({"corrected": "y"}, "correct") == "y"
    assert ai_clipboard._replacement_of({"answer": "z"}, "answer") == "z"
