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

import configparser
from pathlib import Path
from typing import Any, Dict

def read_config(obj: Any, ini_file: Path = Path(__file__).with_name('settings.ini')) -> None:
    """
    iniファイルを読み込み、オブジェクトの属性に設定します。

    Args:
        obj (Any): 設定を適用するオブジェクト。
        ini_file (Path): iniファイルのパス。
    """
    config = configparser.ConfigParser()
    config.read(ini_file)

    for section in config.sections():
        for key, value in config.items(section):
            # オブジェクトに属性として設定
            setattr(obj, key, value)
            # Booleanの場合の変換
            if value.lower() in ['true', 'false']:
                setattr(obj, key, config.getboolean(section, key))
            # 数字の場合の変換
            elif value.isdigit():
                setattr(obj, key, int(value))
            else:
                try:
                    setattr(obj, key, float(value))
                except ValueError:
                    setattr(obj, key, value)

def save_config(obj: Any, ini_file: Path) -> None:
    """
    現在の設定をiniファイルに保存します。

    Args:
        obj (Any): 設定を保存するオブジェクト。
        ini_file (Path): iniファイルのパス。
    """
    config = configparser.ConfigParser()

    for attr in dir(obj):
        if not attr.startswith("__") and not callable(getattr(obj, attr)):
            value = getattr(obj, attr)
            if isinstance(value, (int, float, bool, str)):
                if 'DEFAULT' not in config:
                    config['DEFAULT'] = {}
                config['DEFAULT'][attr] = str(value)

    with open(ini_file, 'w') as configfile:
        config.write(configfile)

class ExampleConfigurableClass:
    def __init__(self, ini_file: Path):
        self.ini_file = ini_file
        self.width = 1280
        self.height = 480
        self.auto_mode = False
        read_config(self, self.ini_file)

    def save_settings(self):
        save_config(self, self.ini_file)

    def display_settings(self):
        print(f"Width: {self.width}, Height: {self.height}, Auto_mode: {self.auto_mode}")

# 使用例
if __name__ == "__main__":
    ini_path = Path(__file__).with_name('settings.ini')
    example = ExampleConfigurableClass(ini_path)
    
    # 設定を表示
    example.display_settings()
    
    # 設定を変更
    example.width = 1920
    example.height = 1080
    example.auto_mode = True

    # プログラム終了時に設定を保存
    example.save_settings()

