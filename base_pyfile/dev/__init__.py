"""開発中の AI 機能（Ollama 振り分け・クリップボード校正/翻訳など）。

このサブパッケージは開発中であり、通常の `pip install .` で作られる
安定版 wheel には含まれない。リポジトリ直下で作業するか、
`pip install -e .` の editable インストールでのみ利用できる。

使い方:
    from base_pyfile.dev import route_to_model, dispatch
    # リポジトリ/editable 環境では後方互換のため
    from base_pyfile import route_to_model, dispatch
    も同じものを返す。
"""

import importlib
from typing import Any, Dict

__all__ = [
    "dispatch",
    "generate",
    "get_available_profiles",
    "is_model_installed",
    "list_installed_models",
    "load_model_profiles",
    "resolve_model",
    "route_to_model",
    "paste_answer",
    "process_clipboard_once",
    "watch_clipboard",
    "watch_hotkey",
    "format_diff",
    "load_history",
    "undo_last",
    "process_text",
    "correct_text",
    "build_correction_prompt",
    "select_model_by_length",
    "translate_text",
    "build_translation_prompt",
]

# 公開名 -> 定義モジュールの対応表。
_MODULE_OF: Dict[str, str] = {
    # ai_router
    "dispatch": "base_pyfile.dev.ai_router",
    "generate": "base_pyfile.dev.ai_router",
    "get_available_profiles": "base_pyfile.dev.ai_router",
    "is_model_installed": "base_pyfile.dev.ai_router",
    "list_installed_models": "base_pyfile.dev.ai_router",
    "load_model_profiles": "base_pyfile.dev.ai_router",
    "resolve_model": "base_pyfile.dev.ai_router",
    "route_to_model": "base_pyfile.dev.ai_router",
    # ai_clipboard
    "paste_answer": "base_pyfile.dev.ai_clipboard",
    "process_clipboard_once": "base_pyfile.dev.ai_clipboard",
    "watch_clipboard": "base_pyfile.dev.ai_clipboard",
    "watch_hotkey": "base_pyfile.dev.ai_clipboard",
    "format_diff": "base_pyfile.dev.ai_clipboard",
    "load_history": "base_pyfile.dev.ai_clipboard",
    "undo_last": "base_pyfile.dev.ai_clipboard",
    "process_text": "base_pyfile.dev.ai_clipboard",
    # ai_corrector（文字数に応じた誤字脱字の校正）
    "correct_text": "base_pyfile.dev.ai_corrector",
    "build_correction_prompt": "base_pyfile.dev.ai_corrector",
    "select_model_by_length": "base_pyfile.dev.ai_corrector",
    # ai_translator（文字数に応じた翻訳。既定は日本語へ）
    "translate_text": "base_pyfile.dev.ai_translator",
    "build_translation_prompt": "base_pyfile.dev.ai_translator",
}


def __getattr__(name: str) -> Any:
    """未定義の属性へアクセスされたときに、対応モジュールを遅延読み込みする。"""
    module_path = _MODULE_OF.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
