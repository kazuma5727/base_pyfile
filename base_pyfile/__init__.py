from base_pyfile.file_manager import read_text_file, write_file
from base_pyfile.function_timer import logger_timer, timer
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import (
    get_all_items,
    get_all_subfolders,
    get_files,
    make_directory,
    unique_path,
)

__all__ = [
    "read_text_file",
    "write_file",
    "logger_timer",
    "timer",
    "get_log_handler",
    "make_logger",
    "get_all_items",
    "get_all_subfolders",
    "get_files",
    "make_directory",
    "unique_path",
]
# パッケージ内のモジュール数を数える
num_modules = len(__all__)

# バージョン番号を更新
tens_place = num_modules // 10
ones_place = num_modules % 10
__version__ = f"{tens_place}.{ones_place}.1"
