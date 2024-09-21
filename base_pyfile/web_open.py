import time
import webbrowser
from logging import NullHandler, getLogger

import pyautogui

from base_pyfile.log_setting import get_log_handler, make_logger

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def open_page(url: str, delay: int = 2, sumaho: bool = False) -> int:
    """
    指定されたURLを開き、一定の遅延時間の後に開いたページの数を返します。

    Args:
        url (str): 開くURL。
        delay (int, optional): ページを開いた後の遅延時間（秒）。デフォルトは2秒。

    Returns:
        int: 開いたページの数（常に1）。
    """
    web_count = 0  # 開いたウェブページの数をカウントする変数
    webbrowser.open(url)  # 指定されたURLをブラウザで開く
    web_count += 1  # 開いたページの数をインクリメント
    time.sleep(delay)  # 指定された遅延時間だけ待機
    if sumaho:
        pyautogui.press("F12")
        time.sleep(2)
        pyautogui.press("F5")
        time.sleep(delay)  # 指定された遅延時間だけ待機
        pyautogui.press("F5")
        time.sleep(delay)  # 指定された遅延時間だけ待機

    return web_count  # 開いたページの数を返す


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    open_page()
