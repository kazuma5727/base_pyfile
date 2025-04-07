from base_pyfile.automation_tools import (
    fast_click,
    full_templatematching,
    learning_materials,
    move_and_click,
    search_color,
    specified_color,
    specified_color_fast_ver,
    templates_matching,
)
from base_pyfile.file_manager import read_text_file, write_file
from base_pyfile.function_timer import logger_timer, timer
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import (
    find_empty_folders,
    get_all_files,
    get_all_subfolders,
    get_files,
    get_folders_and_files,
    get_latest_folder,
    make_directory,
    sanitize_windows_filename,
    unique_path,
)
from base_pyfile.pdf_tiff_converter import (
    convert_to_png,
    image_to_pdf,
    pdf_to_png,
    pdf_to_tiff,
    tiff_to_pdf,
    tiff_to_png,
)
from base_pyfile.web_open import get_urls, open_page, tab_delete

# from base_pyfile.scraping import download_image

__all__ = [
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
    "sanitize_windows_filename",
    "unique_path",
    "open_page",
    "get_urls",
    "tab_delete",
]
# パッケージ内のモジュール数を数える
num_modules = len(__all__)

# バージョン番号を更新
tens_place = num_modules // 10
ones_place = num_modules % 10
__version__ = f"{tens_place}.{ones_place}.2"
