# base_pyfile
便利ツール
# log_setting.py
log_settingは、Pythonのloggingモジュールを使用して、ログを設定するためのユーティリティモジュールです。

## 使い方
以下のようにmake_logger()関数を呼び出すことで、loggerオブジェクトを取得できます。
```python
from log_setting import make_logger

logger = make_logger()
logger.debug("デバッグ")
logger.info("インフォ")
logger.warning("ワーニング")
logger.error("エラー")
logger.critical("クリティカル")
```
make_logger()関数には、以下のように引数を渡すことができます。
```python
def make_logger(
    logger_name: str = "log",
    level: int = logging.DEBUG,
    log_folder: str = "",
    handler: logging.Handler = None,
) -> logging.Logger:
    """ロガーを取得する

    Args:
        logger_name (str, optional): ロガー名。デフォルトは"log"
        level (int, optional): ログレベル。デフォルトはDEBUG
        log_folder (str, optional): ログファイルのパス。デフォルトは標準出力
        handler (logging.Handler, optional): ハンドラ。デフォルトはNone

    Returns:
        logging.Logger: 作成されたロガー
    """
```
log_folderを指定することで、ログファイルを保存するフォルダを指定できます。

また、handlerには、logging.Handlerオブジェクトを指定することもできます。これにより、ユーザー独自のハンドラを使用することができます。
## make_loggerの引数
```python
logger_name (str, optional): ロガー名。デフォルトは"log"
level (int, optional): ログレベル。デフォルトはDEBUG
log_folder (str, optional): ログファイルのパス。デフォルトは標準出力
handler (Handler, optional): ハンドラ。デフォルトで自動生成
```
## get_log_handlerの引数
```python
log_level (int, optional): ログレベル。デフォルトは WARNING
file_path (str, optional): ログファイルを保存するファイルのパス。デフォルトは使用したプログラム
log_folder (str, optional): ログフォルダ作成時のパス。推奨は".log"
```
## 出力
make_logger()関数で作成されたロガーは、以下のログレベルで出力が可能です。

* DEBUG
* INFO
* WARNING
* ERROR
* CRITICAL
ログの出力先は、以下の方法で設定できます。

* ログフォルダを指定する場合
    * ログファイルが、プログラムと同じフォルダに作成されます。
    * ファイル名は、ログレベルとプログラム名を含みます。
* ログフォルダを指定しない場合
    * 標準出力に出力されます。
