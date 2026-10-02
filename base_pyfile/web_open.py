"""ウェブブラウザの操作に関するユーティリティ関数を提供します。

URLを開く、タブを閉じる、ページ内のリンクを抽出するなど、
ブラウザ操作を自動化するための基本的な機能を含みます。
"""

# --- 標準ライブラリのインポート ---
import time
import webbrowser
from logging import NullHandler, getLogger
from typing import List, Union

# --- 外部ライブラリのインポート ---
import pyautogui
import pyperclip
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- 独自モジュールのインポート ---
from base_pyfile.automation_tools import move_and_click, search_color
from base_pyfile.log_setting import get_log_handler, make_logger

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

# ブラウザUI操作で使う座標・色の定数。
# 解像度やブラウザのUIが変わった場合はここを調整する。
POPUP_CLOSE_PROBE_XY = (1090, 680)      # 閉じたいポップアップの確認位置
POPUP_CLOSE_RGB = (51, 51, 51)          # ポップアップが表示されているときの色
DEVTOOLS_COLOR_PROBE_XY = (122, 122)    # DevToolsの表示状態を確認する位置
DEVTOOLS_DARK_RGB = (59, 59, 63)
DEVTOOLS_LIGHT_RGB = (236, 236, 236)
DEVICE_TOOLBAR_TOGGLE_XY = (2050, 130)  # スマホ表示切替ボタン
ADDRESS_BAR_XY = (1270, 60)             # アドレスバー


def open_page(
    urls: Union[str, List[str]],
    delay: int = 2,
    smartphone_mode: bool = False,
    paste_and_go: bool = False,
) -> int:
    """指定されたURLをウェブブラウザで開きます。

    スマートフォン表示への切り替えや、URLを直接アドレスバーに貼り付けて
    ページを開くといった高度な操作も可能です。

    Args:
        urls (Union[str, List[str]]): 開きたい単一のURL、またはURLのリスト。
        delay (int, optional): 各ページを開いた後の待機時間(秒)。デフォルトは2。
        smartphone_mode (bool, optional): Trueの場合、開発者ツールを使ってスマホ表示に切り替えます。デフォルトはFalse。
        paste_and_go (bool, optional): Trueの場合、URLをコピー＆ペーストしてページを開きます。デフォルトはFalse。

    Returns:
        int: 正常に開こうと試みたURLの数。
    """
    if isinstance(urls, str):
        urls = [urls]

    # 処理の前に、特定のポップアップやダイアログが表示されていたら閉じる試み
    # TODO: この座標(1090, 680)と色(51, 51, 51)が何に対応するのかコメントで説明が必要
    if search_color(*POPUP_CLOSE_RGB, xy=POPUP_CLOSE_PROBE_XY):
        move_and_click(POPUP_CLOSE_PROBE_XY)

    for url in urls:
        logger.debug(f"ページを開きます: {url}")
        try:
            webbrowser.open(url)
            time.sleep(delay)

            if smartphone_mode:
                # F12キーで開発者ツールを開く
                pyautogui.press("F12")
                time.sleep(2)
                # TODO: この座標と色が何を示すのか、より具体的な説明が望ましい
                # 例: "スマホ表示モードの切り替えボタンが有効かチェック"
                is_dev_tools_ready = search_color(
                    *DEVTOOLS_DARK_RGB, xy=DEVTOOLS_COLOR_PROBE_XY
                ) or search_color(*DEVTOOLS_LIGHT_RGB, xy=DEVTOOLS_COLOR_PROBE_XY)
                if not is_dev_tools_ready:
                    move_and_click(DEVICE_TOOLBAR_TOGGLE_XY) # スマホ表示切替ボタン
                    time.sleep(2)
                pyautogui.press("F5") # ページをリロードして表示を確定
                time.sleep(delay)

            if paste_and_go:
                pyperclip.copy(url)
                move_and_click(ADDRESS_BAR_XY) # アドレスバーをクリック
                pyautogui.hotkey("ctrl", "a")
                time.sleep(0.5)
                pyautogui.hotkey("ctrl", "v")
                time.sleep(0.5)
                pyautogui.press("enter") # Enterキーでページ遷移
                time.sleep(delay)

        except Exception as e:
            logger.error(f"ページのオープンに失敗しました: {url}, エラー: {e}")

    return len(urls)


def get_urls(
    url: str, open_in_browser: bool = False, delay: int = 1, timeout: int = 10
) -> List[str]:
    """指定URLのHTMLから全てのリンク(href)を抽出します。

    Args:
        url (str): リンクを抽出したいページのURL。
        open_in_browser (bool, optional): Trueの場合、抽出したURLを順次ブラウザで開きます。デフォルトはFalse。
        delay (int, optional): `open_in_browser`がTrueの場合の、各ページを開く間隔(秒)。デフォルトは1。
        timeout (int, optional): HTTP取得のタイムアウト秒。デフォルトは10。

    Returns:
        List[str]: 抽出されたURLのリスト。
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status() # HTTPエラーがあれば例外を発生
        soup = BeautifulSoup(response.content, "html.parser")
        # `href`属性を持つ`a`タグからリンクを抽出
        extracted_urls = [a["href"] for a in soup.find_all("a", href=True)]

        if open_in_browser:
            # httpから始まる有効なURLのみを開く
            valid_urls = [u for u in extracted_urls if u.startswith("http")]
            logger.info(f"{len(valid_urls)}個のリンクをブラウザで開きます。")
            open_page(valid_urls, delay=delay)

        return extracted_urls

    except requests.RequestException as e:
        logger.error(f"URLへのアクセスに失敗しました: {url}, エラー: {e}")
        return []


def tab_delete(count: int = 1, delay: float = 0.3):
    """現在アクティブなブラウザのタブを閉じます。

    Args:
        count (int, optional): 閉じるタブの数。デフォルトは1。
        delay (float, optional): 各タブを閉じる操作の間隔(秒)。デフォルトは0.3。
    """
    time.sleep(1) # 操作の安定性を確保するための待機
    for _ in tqdm(range(count), desc="タブを閉じています"):
        pyautogui.hotkey("ctrl", "w")
        time.sleep(delay)


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # --- 使用例 ---
    # 1. 単一のページを開く
    # open_page("https://www.google.com")

    # 2. 複数のページをスマホモードで開く
    # open_page(["https://www.youtube.com", "https://www.itmedia.co.jp"], smartphone_mode=True)

    # 3. ページからURLを抽出し、それらをブラウザで開く
    # urls = get_urls("https://www.itmedia.co.jp/news/", open_in_browser=True, delay=2)
    # print(f"抽出したURLの数: {len(urls)}")

    # 4. タブを3つ閉じる
    # print("5秒後にタブを3つ閉じます...")
    # time.sleep(5)
    # tab_delete(count=3)

    pass