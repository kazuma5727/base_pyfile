"""ini ファイルと Python オブジェクトの属性を相互に変換するユーティリティ。

``settings.ini`` のような ``[DEFAULT]`` 形式と、任意のセクションを持つ
ini ファイルの両方を読み込めます。値は bool / int / float / str に変換して
オブジェクトへ設定します。
"""

import configparser
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Any, Union

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def _convert_value(value: str) -> Any:
    """ini の文字列値を、可能なら bool / int / float へ変換する。"""
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"

    # int を先に試すことで、"-1" が -1.0 になるのを防ぐ。
    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return value


def read_config(
    obj: Any,
    ini_file: Union[str, Path, None] = None,
) -> None:
    """iniファイルを読み込み、オブジェクトの属性に設定します。

    ``[DEFAULT]`` セクションの値も読み込みます。同名のキーが複数の
    セクションにある場合は、後から読んだセクションの値で上書きされます。

    Args:
        obj (Any): 設定を適用するオブジェクト。
        ini_file (Union[str, Path], optional): iniファイルのパス。省略時は
            パッケージ同梱の ``settings.ini`` を使う。
    """
    ini_path = (
        Path(ini_file)
        if ini_file is not None
        else Path(__file__).with_name("settings.ini")
    )
    config = configparser.ConfigParser()
    if not config.read(ini_path, encoding="utf-8"):
        logger.warning(f"iniファイルを読み込めませんでした: {ini_path}")
        return

    # [DEFAULT] は config.sections() に含まれないため、明示的に読む。
    defaults = config.defaults()
    for key, value in defaults.items():
        setattr(obj, key, _convert_value(value))

    for section in config.sections():
        for key, value in config.items(section):
            # config.items(section) は [DEFAULT] の値も継承して返す。
            # そのまま設定すると、後続セクションを読むたびに DEFAULT の値で
            # 前のセクションの設定を上書きしてしまうため、継承値はスキップする。
            if key in defaults and value == defaults[key]:
                continue
            setattr(obj, key, _convert_value(value))


def save_config(obj: Any, ini_file: Union[str, Path]) -> None:
    """現在の設定をiniファイルに保存します。

    オブジェクトの bool / int / float / str 属性を ``[DEFAULT]`` に保存します。
    セクション構造は保持しません（読み込み時は DEFAULT として扱われます）。

    Args:
        obj (Any): 設定を保存するオブジェクト。
        ini_file (Union[str, Path]): iniファイルのパス。
    """
    ini_path = Path(ini_file)
    ini_path.parent.mkdir(parents=True, exist_ok=True)

    config = configparser.ConfigParser()
    config["DEFAULT"] = {}
    for attr in dir(obj):
        if attr.startswith("__"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            # プロパティなど、取得時に例外を出す属性は無視する。
            continue
        if isinstance(value, (int, float, bool, str)):
            config["DEFAULT"][attr] = str(value)

    with ini_path.open("w", encoding="utf-8") as configfile:
        config.write(configfile)


class ExampleConfigurableClass:
    """read_config / save_config の使用例。"""

    def __init__(self, ini_file: Union[str, Path]):
        self.ini_file = Path(ini_file)
        self.width = 1280
        self.height = 480
        self.auto_mode = False
        read_config(self, self.ini_file)

    def save_settings(self) -> None:
        save_config(self, self.ini_file)

    def display_settings(self) -> None:
        print(
            f"Width: {self.width}, Height: {self.height}, Auto_mode: {self.auto_mode}"
        )


# 使用例
if __name__ == "__main__":
    ini_path = Path(__file__).with_name("settings.ini")
    example = ExampleConfigurableClass(ini_path)

    # 設定を表示
    example.display_settings()

    # 設定を変更
    example.width = 1920
    example.height = 1080
    example.auto_mode = True

    # プログラム終了時に設定を保存
    example.save_settings()
