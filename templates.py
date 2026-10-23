from logging import NullHandler, getLogger

from base_pyfile import (
    get_log_handler,
    logger_timer,
    make_logger,
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
