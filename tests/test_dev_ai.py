"""base_pyfile/dev/ の開発中 AI 機能向け回帰テスト。

ネットワークを使わず、`generate` などはモンキーパッチする。
Ollama など外部サービスも使わない。
"""

import json
import os
from pathlib import Path

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
    assert (package_dir / "dev" / "ai_models_5060ti.json").is_file()


def test_model_profiles_are_valid_and_local_only():
    package_dir = Path(base_pyfile.__file__).parent / "dev"
    cpu = json.loads((package_dir / "ai_models.json").read_text(encoding="utf-8"))
    gpu = json.loads(
        (package_dir / "ai_models_5060ti.json").read_text(encoding="utf-8")
    )

    assert cpu["profile"]["id"] == "cpu"
    assert gpu["profile"]["id"] == "5060ti"

    for profile in (cpu, gpu):
        model_names = {model["name"] for model in profile["models"]}
        assert profile["profile"]["cloud"] is False
        assert not [name for name in model_names if "cloud" in name.lower()]
        assert profile["router"]["preferred"] in model_names
        assert profile["router"]["safe_target"] in model_names
        assert profile["router"]["default_target"] in model_names
        assert profile["agent"]["model"] in model_names
        assert all(model.get("best_for") for model in profile["models"])

        for section in ("correction", "translation"):
            for tier in profile[section]["tiers"]:
                assert tier["model"] in model_names
            assert all(name in model_names for name in profile[section]["fallback"])


def test_is_model_installed_requires_matching_tag():
    installed = ["qwen2.5:0.5b", "qwen2.5-coder:3b"]

    assert ai_router.is_model_installed("qwen2.5:0.5b", installed)
    assert not ai_router.is_model_installed("qwen2.5:1.5b", installed)
    # タグ省略時だけファミリー一致を許す（既存の挙動）。
    assert ai_router.is_model_installed("qwen2.5", installed)


def test_model_profiles_fall_back_to_installed_models():
    package_dir = Path(base_pyfile.__file__).parent / "dev"
    cpu = json.loads((package_dir / "ai_models.json").read_text(encoding="utf-8"))
    gpu = json.loads(
        (package_dir / "ai_models_5060ti.json").read_text(encoding="utf-8")
    )

    # CPU版は現在インストール済みの4モデルだけで動く。
    cpu_installed = [
        "tev1:0.8b",
        "qwen2.5:0.5b",
        "qwen2.5-coder:3b",
        "deepseek-r1:1.5b",
    ]
    assert (
        ai_corrector.select_model_by_length(
            "あ" * 500, profiles=cpu, installed=cpu_installed
        )
        == "qwen2.5-coder:3b"
    )

    # 8GB版は14Bが無くても7Bへフォールバックする。
    gpu_8gb = ["qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5-coder:7b", "deepseek-r1:8b"]
    assert (
        ai_corrector.select_model_by_length(
            "あ" * 1000, profiles=gpu, installed=gpu_8gb
        )
        == "qwen2.5:7b"
    )

    # 16GB版で14Bが入っていれば14Bを使う。
    gpu_16gb = gpu_8gb + ["qwen2.5:14b"]
    assert (
        ai_corrector.select_model_by_length(
            "あ" * 1000, profiles=gpu, installed=gpu_16gb
        )
        == "qwen2.5:14b"
    )


def test_model_profile_selection(monkeypatch):
    monkeypatch.delenv(ai_router.MODEL_PROFILE_ENV, raising=False)
    assert ai_router._model_profile_path().name == "ai_models.json"
    assert ai_router.load_model_profiles()["profile"]["id"] == "cpu"

    monkeypatch.setenv(ai_router.MODEL_PROFILE_ENV, "5060ti")
    assert ai_router._model_profile_path().name == "ai_models_5060ti.json"
    assert ai_router.load_model_profiles()["profile"]["id"] == "5060ti"

    monkeypatch.setenv(ai_router.MODEL_PROFILE_ENV, "gpu")
    assert ai_router._model_profile_path().name == "ai_models_5060ti.json"


def test_model_profile_custom_path_and_missing_fallback(monkeypatch, tmp_path):
    custom = tmp_path / "custom.json"
    custom.write_text('{"profile": {"id": "custom"}}', encoding="utf-8")
    monkeypatch.setenv(ai_router.MODEL_PROFILE_ENV, str(custom))
    assert ai_router._model_profile_path() == custom

    monkeypatch.setenv(ai_router.MODEL_PROFILE_ENV, str(tmp_path / "missing.json"))
    assert ai_router._model_profile_path().name == "ai_models.json"


def test_ai_clipboard_profile_option_sets_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(ai_router.MODEL_PROFILE_ENV, raising=False)
    exit_code = ai_clipboard.main(
        [
            "--history",
            "--profile",
            "5060ti",
            "--history-path",
            str(tmp_path / "history.json"),
        ]
    )

    assert exit_code == 0
    assert os.environ[ai_router.MODEL_PROFILE_ENV] == "5060ti"


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
    assert (
        _match_model_name("使うのは deepseek-r1 です", candidates) == "deepseek-r1:1.5b"
    )
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
        ai_corrector.select_model_by_length("short", PROFILES_FOR_CORRECTION, ["mid"])
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
