"""base_pyfile: Pythonを扱いやすくするユーティリティ群。

このパッケージは遅延import（lazy import）方式を採用しています。
`import base_pyfile` した時点では軽い読み込みのみを行い、各機能は
実際にその名前へアクセスしたときに初めて対応モジュールを読み込みます。

これにより、たとえば画像系の cv2 や ブラウザ操作の pyautogui が
インストールされていなくても、log_setting や path_manager だけを
使いたい利用者は問題なくそれらを利用できます。

安定版の公開名はこのファイルで管理し、開発中の AI 機能は
`base_pyfile/dev/` に分離しています。通常の `pip install .` で作られる
wheel には dev が含まれず、リポジトリ直下や `pip install -e .` の
editable インストールのときだけ、後方互換のため公開名に追加されます。

使い方:
    from base_pyfile import make_logger, unique_path

    # 開発環境（リポジトリ/editable）では AI 機能も従来どおり使える
    from base_pyfile import dispatch
"""

import importlib
from typing import Any, Dict

from base_pyfile._version import __version__

# 安定版として公開する名前（この一覧は後方互換のために明示的に保持する）
__all__ = [
    "fast_click",
    "full_templatematching",
    "learning_materials",
    "move_and_click",
    "search_color",
    "specified_color",
    "specified_color_fast_ver",
    "templates_matching",
    "atomic_write",
    "read_text_file",
    "write_file",
    "run_command",
    "CommandResult",
    "doctor",
    "read_config",
    "save_config",
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
# AI 関連（base_pyfile/dev/）は下で自動的にマージする。
_MODULE_OF: Dict[str, str] = {
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
    "atomic_write": "base_pyfile.file_manager",
    "read_text_file": "base_pyfile.file_manager",
    "write_file": "base_pyfile.file_manager",
    # process_manager
    "CommandResult": "base_pyfile.process_manager",
    "run_command": "base_pyfile.process_manager",
    # diagnostics
    "doctor": "base_pyfile.diagnostics",
    # ini_filemanager
    "read_config": "base_pyfile.ini_filemanager",
    "save_config": "base_pyfile.ini_filemanager",
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

# 開発中機能（base_pyfile/dev/）の公開名を取り込む。
# 安定版 wheel には dev が含まれないため、ImportError になった場合は
# 安定版の名前だけを公開する。
try:
    from base_pyfile.dev import __all__ as _dev_all
    from base_pyfile.dev import _MODULE_OF as _dev_module_of
except ImportError:
    pass
else:
    __all__ = list(__all__) + [name for name in _dev_all if name not in __all__]
    _MODULE_OF = {**_MODULE_OF, **_dev_module_of}


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


# バージョンは base_pyfile/_version.py で管理する。
# 以前は __all__ の件数から自動計算していたが、公開名を増減するだけで
# バージョンが変わってしまうため、明示的な値に変更した。
