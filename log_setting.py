import os
from logging import (
    getLogger,
    StreamHandler,
    Formatter,
    FileHandler,
    NullHandler,
    Handler,
    Logger,
    basicConfig,
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    CRITICAL,
)

# トップレベルのロガーを作成し、その子として、このモジュールのロガーを作成
logger = getLogger("log").getChild(__name__)

# NullHandlerを追加し、ログを出力しないように設定
logger.addHandler(NullHandler())
# basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",level=DEBUG)


def get_log_handler(
    log_level: int = WARNING, log_file_path: str = __file__, log_folder: str = ".log"
) -> Handler:
    """ログハンドラを作成する

    Args:
        log_level (int, optional): ログレベル。デフォルトは WARNING
        log_file_path (str, optional): ログファイルのパス。デフォルトはこのモジュールのパス
        log_folder (str, optional): ログフォルダのパス。デフォルトは".log"

    Returns:
        Handler: 作成されたハンドラ
    """

    log_level_names = {50: "CRITICAL", 40: "ERROR", 30: "WARNING", 20: "INFO", 10: "DEBUG"}

    # ログフォルダが指定された場合、ログフォルダを作成し、ログファイルを作成
    if log_folder and log_file_path != __file__:
        os.makedirs(log_folder, exist_ok=True)
        log_file_name = f"{log_level_names[log_level]}_{os.path.splitext(os.path.basename(log_file_path))[0]}.log"
        log_file_path = os.path.join(log_folder, log_file_name)
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
    log_file_path: str = __file__,
    handler: Handler = None,
) -> Logger:
    """ロガーを取得する

    Args:
        logger_name (str, optional): ロガー名。デフォルトは"log"
        level (int, optional): ログレベル。デフォルトはDEBUG
        log_file_path (str, optional): ログファイルのパス。デフォルトはこのモジュールのパス
        handler (Handler, optional): ハンドラ。デフォルトはNone

    Returns:
        Logger: 作成されたロガー
    """

    # ロガーオブジェクトを作成し、名前とレベルを設定
    logger = getLogger(logger_name)
    logger.setLevel(level)

    # handlerが与えられた場合はそれを使用し、与えられなかった場合はcreate_handler()関数で作成したハンドラを使用
    if handler:
        logger.addHandler(handler)

    else:
        logger.addHandler(get_log_handler(level, log_file_path))

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
