import os
import time
from logging import NullHandler, getLogger

import requests

from base_pyfile import get_log_handler, make_logger, unique_path

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def download_image(url: str, filename: str = None) -> bool:
    """指定したURLから画像をダウンロードして、指定したファイル名で保存する。

    Args:
        url (str): ダウンロードする画像のURL
        filename (str, optional): 保存するファイル名。指定しない場合はタイムスタンプで自動生成される。

    Returns:
        bool: ダウンロードに成功した場合はTrue、失敗した場合はFalse
    """
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError,
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ) as e:
        print(f"リクエストエラー：{e}")
        return False

    # 拡張子をチェックして、指定されたものでなければデフォルトの拡張子にする
    _, ext = os.path.splitext(url)
    if not ext.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".svg", ".webp"]:
        ext = ".png"

    # ファイル名が指定されていない場合はタイムスタンプでファイル名を生成する
    if not filename:
        timestamp = str(int(time.time()))
        filename = os.path.join("images", f"{timestamp}")

    # 拡張子が含まれていない場合はファイル名に拡張子を追加する
    if not "." in filename:
        filename += ext

    with open(unique_path(filename), "wb") as f:
        f.write(response.content)

    # 1秒待機する
    time.sleep(1)
    return True


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))
    filename = ""
    # download_image("https://tutorials.chainer.org/ja/img/top/fv-thumbnail.png", filename)
