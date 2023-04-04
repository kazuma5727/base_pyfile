import os
import time
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Union

import requests

from base_pyfile import get_log_handler, make_logger, unique_path

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def download_image(url: str, filename: Union[str, Path] = None) -> bool:
    """
    指定されたURLから画像をダウンロードし、指定されたファイル名で保存する。

    Args:
        url (str): 画像のURL
        filename (Union[str, Path]): 保存するファイル名。指定されていない場合は、タイムスタンプでファイル名を生成する。

    Returns:
        bool: ダウンロードが成功した場合はTrue、失敗した場合はFalse
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
    # pathlibを使って書き換え
    if not filename:
        # 現在時刻を取得
        current_time = time.localtime()

        # ファイル名を生成
        filename = (
            Path("images")
            / f"{str(current_time.tm_year)[2:]}_{current_time.tm_mon:02}_{current_time.tm_mday:02}_{current_time.tm_hour:02}_{current_time.tm_min:02}_{current_time.tm_sec:02}"
        )

    # 拡張子が含まれていない場合はファイル名に拡張子を追加する
    else:
        filename = Path(filename)

    if not filename.suffix:
        filename = filename.with_suffix(ext)

    with open(unique_path(filename), "wb") as f:
        f.write(response.content)

    # 1秒待機する
    time.sleep(2)
    return True


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))
    filename = ""
    # download_image("https://tutorials.chainer.org/ja/img/top/fv-thumbnail.png", filename)
