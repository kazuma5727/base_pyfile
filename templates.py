import os
import sys
from logging import getLogger, NullHandler

module_path = r"C:\tool\base_pyfile"
sys.path.append(module_path)
from log_setting import make_logger, get_log_handler
from path_manager import get_files, unique_path
from file_manager import read_text_file, write_file


logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def temp():
    """_summary_"""
    logger.debug("debug")
    logger.info("info")
    logger.warning("warn")
    logger.error("error")


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    temp()
