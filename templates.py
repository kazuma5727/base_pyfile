import os
import site
import sys
from logging import NullHandler, getLogger

module_path = r"C:\tool\base_pyfile"
site.addsitedir(module_path)
from file_manager import read_text_file, write_file
from function_timer import logger_timer, timer
from log_setting import get_log_handler, make_logger
from path_manager import get_files, unique_path

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


@logger_timer()
def temp():
    """_summary_"""
    logger.debug("debug")
    logger.info("info")
    logger.warning("warn")
    logger.error("error")


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    temp()
