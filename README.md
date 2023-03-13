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

# function_timer.py
function_timer.pyはPythonの関数の実行時間を測定してログ出力するための機能を提供します。
## 使用方法
1. base_pyfileモジュールをインストールします。
2. log_settingモジュールをインポートし、ログを設定します。
3. timerデコレーターまたはlogger_timerデコレーターを関数に適用します。
4. プログラムを実行します。
## timerデコレーター
timerデコレーターを関数に適用すると、関数の実行時間を標準出力に表示します。
```python
from function_timer import timer

@timer
def my_function():
    # 関数の処理
```
## logger_timerデコレーター
logger_timerデコレーターを関数に適用すると、関数の実行時間をログ出力します。
```python
import logging
from function_timer import logger_timer

logger = logging.getLogger(__name__)

@logger_timer
def my_function():
    # 関数の処理
```
デコレーターの引数には、ログレベルや実行回数を指定することができます。
```python
@logger_timer(level=logging.INFO, n=5)
def my_function():
    # 関数の処理
```
## サンプルコード
以下のサンプルコードでは、フィボナッチ数列を計算する関数fibonacciに、timerデコレーターとlogger_timerデコレーターを適用しています。
```python
from function_timer import timer, logger_timer

@timer
@logger_timer(n=10)
@timer
def fibonacci(n):
    def _fib(n):
        if n < 2:
            return n
        return _fib(n - 1) + _fib(n - 2)

    return _fib(n)

print(fibonacci(30))
```
このサンプルコードを実行すると、以下のようなログが出力されます。
```cmd
fibonacci: 4.855932099977508 seconds
fibonacci: 3.9821334999869578 seconds
fibonacci: 3.6236166000016965 seconds
fibonacci: 3.763159199967049 seconds
fibonacci: 3.9010207999963313 seconds
fibonacci: 3.704733800026588 seconds
fibonacci: 4.606445599987637 seconds
fibonacci: 3.9488255999749526 seconds
fibonacci: 4.017697299947031 seconds
fibonacci: 4.278482699999586 seconds
2023-03-13 15:39:32,399 - log - DEBUG - fibonacci (10回実行の平均): 4.068646910000825 seconds
fibonacci: 40.68915900000138 seconds
832040
```
