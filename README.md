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
# path_manager.py
path_managerは、Pythonのpath周りを簡易に設定するためのユーティリティモジュールです。
## unique_path関数
ファイルパスを受け取り、ファイル名の末尾に接尾辞を追加し、ファイルパスがユニークであることを保証する関数です。接尾辞には、連番を使用することができます。既存のファイルが存在する場合、接尾辞を追加して、存在しないファイル名が得られるまで続けます。ファイル名に{}を含めると、その場所に接尾辞が追加されます。同じファイル名が渡された場合、それに対応する接尾辞が保持され、同じ接尾辞が再利用されます。
## 引数
file_path (str) - ファイルパス
counter (int, optional) - ファイルパスの接尾辞に付く連番（デフォルト値は1）
suffix (str, optional) - ファイルパスの接尾辞の文字列（デフォルト値は"_"）
existing_text (str, optional) - 既存のテキストファイルが存在する場合、ファイルが同じであるかを確認するための文字列
existing_image (numpy.ndarray, optional) - 既存の画像ファイルが存在する場合、ファイルが同じであるかを確認するためのndarray
## 戻り値
str - 一意になったファイルパス
## make_directory関数
指定されたパスのディレクトリを作成する関数です。cacheデコレータが付与されており、同じパスが複数回渡された場合、再帰的なディレクトリ作成を回避するためにキャッシュされます。

## 引数
path (str) - 作成するディレクトリのパス
## 戻り値
str - 渡されたパスをそのまま返します。
## get_files関数
フォルダ内にあるすべてのファイルの絶対パスを返す関数です。choice_keyを指定すると、ファイル名にキーワードが含まれているものだけを返します。

## 引数
path (str) - ファイルまたはフォルダの絶対パス
choice_key (str, optional) - ファイル名に含まれる必要な文字列。この文字列がファイル名に含まれている場合に、そのファイルを選択肢の候補として表示するようになります。例えば、choice_key="test"とすると、ファイル名に"test"が含まれているものだけが選択肢として表示されます。デフォルト値はNoneで、ファイル名によるフィルターは行われません。
## get_all_subfolders関数
### 概要
指定されたディレクトリ以下の全てのフォルダを再帰的に検索し、フォルダパスのリストを返す関数です。

### 引数
directory (str)：検索対象のディレクトリパス
depth (Optional[int])：検索する階層数。Noneの場合、全階層を検索する。デフォルトはNone。
### 返り値
List[str]：ディレクトリパスのリスト（自然順にソートされている）
## 内部関数
get_subfolders(directory: str, depth: Optional[int]) -> List[str]

指定されたディレクトリ以下のフォルダを再帰的に検索し、フォルダパスのリストを返す内部関数です。

### 引数
directory (str)：検索対象のディレクトリパス
depth (Optional[int])：検索する階層数。Noneの場合、全階層を検索する。デフォルトはNone。
### 返り値
List[str]：ディレクトリパスのリスト。
返り値は、検索対象のディレクトリ以下に存在する全てのディレクトリのパスのリストです。自然順にソートされています。フォルダが空の場合、空のリストが返されます。

例えば、以下のように実行すると、ディレクトリ test 以下に存在する全てのサブフォルダのパスを取得できます。
```python
dirs = get_all_subfolders('test')
print(dirs)
```
test ディレクトリの構造が以下の場合、上記コードの実行結果は次のようになります。
```cmd
test/
    ├── dir1/
    │   ├── file1.txt
    │   ├── file2.txt
    │   └── sub1/
    │       ├── file1.txt
    │       └── file2.txt
    └── dir2/
        ├── file1.txt
        └── sub2/
            └── file1.txt
```
実行結果:
```cmd
[    'test/dir1',    'test/dir1/sub1',    'test/dir2',    'test/dir2/sub2']
```