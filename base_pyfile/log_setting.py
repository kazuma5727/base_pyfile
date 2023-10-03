import os
import sys
from logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    FileHandler,
    Formatter,
    Handler,
    Logger,
    NullHandler,
    StreamHandler,
    basicConfig,
    getLogger,
)
from pathlib import Path

# トップレベルのロガーを作成し、その子として、このモジュールのロガーを作成
logger = getLogger("log").getChild(__name__)

# NullHandlerを追加し、ログを出力しないように設定
logger.addHandler(NullHandler())
# basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",level=DEBUG)


def get_log_handler(
    log_level: int = WARNING, file_path: Path = Path(sys.argv[0]), log_folder: str = ""
) -> Handler:
    """ログハンドラを作成

    Args:
        log_level (int, optional): ログレベル。デフォルトは WARNING
        file_path (str, optional): ログファイルを保存するファイルのパス。デフォルトは使用したプログラム
        log_folder (str, optional): ログフォルダ作成時のパス。推奨は".log"

    Returns:
        Handler: 作成されたハンドラ
    """

    LOG_LEVEL_NAMES = {50: "CRITICAL", 40: "ERROR", 30: "WARNING", 20: "INFO", 10: "DEBUG"}

    # ログフォルダが指定された場合、ログフォルダを作成し、ログファイルを作成
    if log_folder:
        file_path = Path(file_path)
        if file_path.is_file():
            file_folder_path = file_path.parent
        log_folder_path = file_folder_path / log_folder
        log_folder_path.mkdir(parents=True, exist_ok=True)
        log_file_name = f"{LOG_LEVEL_NAMES[log_level]}_{file_path.stem}.log"
        log_file_path = log_folder_path / log_file_name
        logger.info(f"ログファイルを作成: {log_file_path}")
        handler = FileHandler(filename=log_file_path)

    # ログフォルダが指定されない場合、標準出力に出力
    else:
        handler = StreamHandler()

    handler.setLevel(log_level)
    formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    return handler


def make_logger(
    logger_name: str = "log",
    level: int = DEBUG,
    log_folder: str = "",
    handler: Handler = None,
) -> Logger:
    """ロガーを取得する

    Args:
        logger_name (str, optional): ロガー名。デフォルトは"log"
        level (int, optional): ログレベル。デフォルトはDEBUG
        log_folder (str, optional): ログファイルのパス。デフォルトは標準出力
        handler (Handler, optional): ハンドラ。デフォルトで自動生成

    Returns:
        Logger: 作成されたロガー
    """

    # ロガーオブジェクトを作成し、名前を設定
    logger = getLogger(logger_name)

    # handlerが与えられた場合はそれを使用し、与えられなかった場合はget_log_handler()関数で作成したハンドラを使用
    if handler:
        logger.addHandler(handler)
        level = handler.level

    else:
        logger.addHandler(get_log_handler(level, log_folder))

    # レベルを設定
    logger.setLevel(level)

    # 親ロガーにログを伝播させないように設定
    logger.propagate = False

    return logger


if __name__ == "__main__":
    logger = make_logger()

    logger.debug("デバッグ")
    logger.info("インフォ")
    logger.warning("ワーニング")
    logger.error("エラー")
    logger.critical("クリティカル")
