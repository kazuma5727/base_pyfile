import time
import webbrowser
from logging import NullHandler, getLogger

from base_pyfile.log_setting import get_log_handler, make_logger

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def open_page(url, delay=2):
    webbrowser.open(url)
    web_count += 1
    time.sleep(delay)
    return web_count


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    open_page()
