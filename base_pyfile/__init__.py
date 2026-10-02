"""base_pyfile: Pythonを扱いやすくするユーティリティ群。

このパッケージは遅延import（lazy import）方式を採用しています。
`import base_pyfile` した時点では軽い読み込みのみを行い、各機能は
実際にその名前へアクセスしたときに初めて対応モジュールを読み込みます。

これにより、たとえば画像系の cv2 や ブラウザ操作の pyautogui が
インストールされていなくても、log_setting や path_manager だけを
使いたい利用者は問題なくそれらを利用できます。

使い方:
    from base_pyfile import make_logger, unique_path, dispatch
    # ↑ アクセスした名前のモジュールだけが読み込まれる
"""

import importlib
from typing import Any, Dict

# 公開する名前（この一覧は後方互換のために明示的に保持する）
__all__ = [
    "run_agent",
    "run_repl",
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
    "process_text",
    "correct_text",
    "build_correction_prompt",
    "select_model_by_length",
    "translate_text",
    "build_translation_prompt",
    "fast_click",
    "full_templatematching",
    "learning_materials",
    "move_and_click",
    "search_color",
    "specified_color",
    "specified_color_fast_ver",
    "templates_matching",
    "read_text_file",
    "write_file",
    "logger_timer",
    "timer",
    "get_log_handler",
    "make_logger",
    "find_empty_folders",
    "get_all_files",
    "get_all_subfolders",
    "get_files",
    "get_folders_and_files",
    "get_latest_folder",
    "make_directory",
    "reset_existing_files",
    "sanitize_windows_filename",
    "unique_path",
    "open_page",
    "get_urls",
    "tab_delete",
]

# 公開名 -> 定義モジュールの対応表。
# __all__ に未掲載でも従来から import できていたもの（PDF系）も含める。
_MODULE_OF: Dict[str, str] = {
    # ai_router
    "dispatch": "base_pyfile.ai_router",
    "generate": "base_pyfile.ai_router",
    "get_available_profiles": "base_pyfile.ai_router",
    "is_model_installed": "base_pyfile.ai_router",
    "list_installed_models": "base_pyfile.ai_router",
    "load_model_profiles": "base_pyfile.ai_router",
    "resolve_model": "base_pyfile.ai_router",
    "route_to_model": "base_pyfile.ai_router",
    # agent（新規作成中。未実装でもアクセスするまでエラーにしない）
    "run_agent": "base_pyfile.agent",
    "run_repl": "base_pyfile.agent",
    # ai_clipboard
    "paste_answer": "base_pyfile.ai_clipboard",
    "process_clipboard_once": "base_pyfile.ai_clipboard",
    "watch_clipboard": "base_pyfile.ai_clipboard",
    "process_text": "base_pyfile.ai_clipboard",
    # ai_corrector（文字数に応じた誤字脱字の校正）
    "correct_text": "base_pyfile.ai_corrector",
    "build_correction_prompt": "base_pyfile.ai_corrector",
    "select_model_by_length": "base_pyfile.ai_corrector",
    # ai_translator（文字数に応じた翻訳。既定は日本語へ）
    "translate_text": "base_pyfile.ai_translator",
    "build_translation_prompt": "base_pyfile.ai_translator",
    # automation_tools（cv2 / pyautogui / pynput が必要）
    "fast_click": "base_pyfile.automation_tools",
    "full_templatematching": "base_pyfile.automation_tools",
    "learning_materials": "base_pyfile.automation_tools",
    "move_and_click": "base_pyfile.automation_tools",
    "search_color": "base_pyfile.automation_tools",
    "specified_color": "base_pyfile.automation_tools",
    "specified_color_fast_ver": "base_pyfile.automation_tools",
    "templates_matching": "base_pyfile.automation_tools",
    # file_manager
    "read_text_file": "base_pyfile.file_manager",
    "write_file": "base_pyfile.file_manager",
    # function_timer
    "logger_timer": "base_pyfile.function_timer",
    "timer": "base_pyfile.function_timer",
    # log_setting
    "get_log_handler": "base_pyfile.log_setting",
    "make_logger": "base_pyfile.log_setting",
    # path_manager
    "find_empty_folders": "base_pyfile.path_manager",
    "get_all_files": "base_pyfile.path_manager",
    "get_all_subfolders": "base_pyfile.path_manager",
    "get_files": "base_pyfile.path_manager",
    "get_folders_and_files": "base_pyfile.path_manager",
    "get_latest_folder": "base_pyfile.path_manager",
    "make_directory": "base_pyfile.path_manager",
    "reset_existing_files": "base_pyfile.path_manager",
    "sanitize_windows_filename": "base_pyfile.path_manager",
    "unique_path": "base_pyfile.path_manager",
    # web_open（bs4 / tqdm などが必要）
    "open_page": "base_pyfile.web_open",
    "get_urls": "base_pyfile.web_open",
    "tab_delete": "base_pyfile.web_open",
    # pdf_tiff_converter（PyMuPDF / Pillow が必要）
    "convert_to_png": "base_pyfile.pdf_tiff_converter",
    "image_to_pdf": "base_pyfile.pdf_tiff_converter",
    "pdf_to_png": "base_pyfile.pdf_tiff_converter",
    "pdf_to_tiff": "base_pyfile.pdf_tiff_converter",
    "tiff_to_pdf": "base_pyfile.pdf_tiff_converter",
    "tiff_to_png": "base_pyfile.pdf_tiff_converter",
}


def __getattr__(name: str) -> Any:
    """未定義の属性へアクセスされたときに、対応モジュールを遅延読み込みする。

    PEP 562 のモジュールレベル __getattr__。`from base_pyfile import make_logger`
    のような形でも呼ばれる。
    """
    module_path = _MODULE_OF.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    # 2回目以降は通常の属性としてキャッシュする
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


# パッケージ内のモジュール数を数える
num_modules = len(__all__)

# バージョン番号を更新
tens_place = num_modules // 10
ones_place = num_modules % 10
__version__ = f"{tens_place}.{ones_place}.4"
