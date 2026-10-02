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
    getLevelName,
    getLogger,
)
from pathlib import Path

LOG_LEVEL_NAMES = {50: "CRITICAL", 40: "ERROR", 30: "WARNING", 20: "INFO", 10: "DEBUG"}
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# トップレベルのロガーを作成し、その子として、このモジュールのロガーを作成
logger = getLogger("log").getChild(__name__)

# NullHandlerを追加し、ログを出力しないように設定
logger.addHandler(NullHandler())


def _level_name(level: int) -> str:
    """ログレベルを、ログファイル名に使える安全な文字列へ変換する。

    標準レベル（DEBUG/INFO/WARNING/ERROR/CRITICAL）以外や未知の値でも
    KeyError にならないようにするためのヘルパー。
    """
    name = getLevelName(level)
    # 未知のレベルでは "Level 5" のような空白入りの文字列が返るため除外
    if isinstance(name, str) and " " not in name:
        return name
    return f"LEVEL{level}"


def get_log_handler(
    log_level: int = WARNING,
    file_path: Path = Path(sys.argv[0]),
    log_folder: str = "",
    log_format: str = DEFAULT_LOG_FORMAT,
) -> Handler:
    """ログハンドラを作成

    Args:
        log_level (int, optional): ログレベル。デフォルトは WARNING
        file_path (Path, optional): ログファイルを保存するファイルのパス。デフォルトは使用したプログラム
        log_folder (str, optional): ログフォルダ作成時のパス。推奨は".log"
        log_format (str, optional): ログのフォーマット。デフォルトは DEFAULT_LOG_FORMAT

    Returns:
        Handler: 作成されたハンドラ
    """
    # Pathに正規化しておく（文字列・Pathどちらでも受け付ける）
    file_path = Path(file_path)

    # ログフォルダが指定された場合、ログフォルダを作成し、ログファイルを作成
    if log_folder:
        # file_path がファイルでなくても親ディレクトリを基準にする
        # （以前は file_path.is_file() のときだけ代入しており UnboundLocalError になっていた）
        log_folder_path = file_path.parent / log_folder
        log_folder_path.mkdir(parents=True, exist_ok=True)
        log_file_name = f"{_level_name(log_level)}_{file_path.stem}.log"
        log_file_path = log_folder_path / log_file_name
        logger.info(f"ログファイルを作成: {log_file_path}")
        handler = FileHandler(filename=log_file_path, encoding="utf-8")

    # ログフォルダが指定されない場合、標準出力に出力
    else:
        handler = StreamHandler()

    handler.setLevel(log_level)
    formatter = Formatter(log_format)
    handler.setFormatter(formatter)
    return handler


def make_logger(
    logger_name: str = "log",
    level: int = DEBUG,
    log_folder: str = "",
    handler: Handler = None,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> Logger:
    """ロガーを取得する

    Args:
        logger_name (str, optional): ロガー名。デフォルトは"log"
        level (int, optional): ログレベル。デフォルトはDEBUG
        log_folder (str, optional): ログファイルのパス。デフォルトは標準出力
        handler (Handler, optional): ハンドラ。デフォルトで自動生成
        log_format (str, optional): ログのフォーマット。デフォルトは DEFAULT_LOG_FORMAT

    Returns:
        Logger: 作成されたロガー
    """
    # ロガーオブジェクトを作成し、名前を設定
    # （getLoggerは同名なら同じオブジェクトを返す点に注意）
    logger = getLogger(logger_name)

    # 既存ハンドラを除去する。
    # 同名ロガーで make_logger を複数回呼ぶと、以前はハンドラが積み重なり
    # 同じログが重複出力されていた。
    for existing_handler in list(logger.handlers):
        logger.removeHandler(existing_handler)

    # handlerが与えられた場合はそれを使用し、与えられなかった場合はget_log_handler()関数で作成したハンドラを使用
    if handler:
        target_handler = handler
    else:
        target_handler = get_log_handler(level, log_folder=log_folder, log_format=log_format)
    logger.addHandler(target_handler)

    # レベルを設定
    # 引数の level を優先する（以前は handler 指定時に handler.level で上書きしていた）
    logger.setLevel(level)

    # 親ロガーにログを伝播させないように設定
    logger.propagate = False

    return logger


if __name__ == "__main__":
    # デフォルトのログレベルとフォーマットでロガーを作成
    logger = make_logger(handler=get_log_handler(10))
    logger.debug("デバッグログテスト")
    logger.info("インフォログテスト")
