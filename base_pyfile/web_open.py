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


def open_page(urls: list, delay: int = 2, sumaho: bool = False, paste=False) -> int:
    """
    指定されたURLを開き、一定の遅延時間の後に開いたページの数を返します。

    Args:
        urls (str): 開くURL。
        delay (int, optional): ページを開いた後の遅延時間（秒）。デフォルトは2秒。

    Returns:
        int: 開いたページの数（常に1）。
    """
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        logger.debug(f"open page: {url}")
        try:
            webbrowser.open(url)  # 指定されたURLをブラウザで開く
            time.sleep(delay)  # 指定された遅延時間だけ待機
            if sumaho:
                pyautogui.press("F12")
                time.sleep(2)
                if not search_color(59, 59, 63, 122, 122) or not search_color(236, 236, 236, 122, 122):
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
        except Exception as e:
            logger.error(f"Error opening page: {e}")

    return len(urls)  # 開いたページの数を返す


def get_urls(url: str, web=False) -> list:
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    current_urls = [link.get("href") for link in soup.find_all("a") if link.get("href")]
    if web:
        if not isinstance(web, int):
            web = 0
        for current_url in current_urls:
            open_page(current_url, delay=web)
    return current_urls


def tab_delete(count=1,delay=0.3):
    time.sleep(1)
    for i in tqdm(range(count)):
        time.sleep(delay)
        pyautogui.hotkey("ctrl", "w")
    return 0


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # open_page()
