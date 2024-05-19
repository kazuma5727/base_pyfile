import os
import sys
from logging import NullHandler, getLogger
from pathlib import Path

from base_pyfile import (
    get_all_files,
    get_all_subfolders,
    get_files,
    get_folders_and_files,
    get_log_handler,
    logger_timer,
    make_directory,
    make_logger,
    read_text_file,
    unique_path,
    write_file,
)

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
