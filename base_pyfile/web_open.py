import time
import webbrowser
from logging import NullHandler, getLogger

import pyautogui
import pyperclip
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from base_pyfile.automation_tools import move_and_click, search_color
from base_pyfile.log_setting import get_log_handler, make_logger

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def open_page(url: str, delay: int = 2, sumaho: bool = False, paste=False) -> int:
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
        if not search_color(59, 59, 63, 122, 122):
            move_and_click(2050, 130)
            time.sleep(2)
        pyautogui.press("F5")
        time.sleep(delay)  # 指定された遅延時間だけ待機
        if paste == True:
            pyperclip.copy(url)
            move_and_click(1270, 60)
            pyautogui.hotkey("ctrl", "a")
            time.sleep(0.5)
            pyautogui.hotkey("ctrl", "v")
            pyautogui.press("enter")

        pyautogui.press("F5")
        time.sleep(delay)  # 指定された遅延時間だけ待機

    return web_count  # 開いたページの数を返す


def get_urls(url: str, web_open=False) -> list:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    current_urls = [link.get("href") for link in soup.find_all("a") if link.get("href")]
    if web_open:
        if not isinstance(web_open, int):
            web_open = 0
        for url in current_urls:
            open_page(url, delay=0)
    return current_urls


def tab_delete(count=1):
    time.sleep(1)
    for i in tqdm(range(count)):
        time.sleep(0.3)
        pyautogui.hotkey("ctrl", "w")
    return 0


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # open_page()
