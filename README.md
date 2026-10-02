# base_pyfile
便利ツール

---

## AI・エージェント向けクイックリファレンス

このパッケージは **遅延import（lazy import）** 方式です。`import base_pyfile` した時点では軽い読み込みだけを行い、`from base_pyfile import make_logger` のように**名前へアクセスした時点で、対応モジュールだけ**が読み込まれます。そのため、cv2 / pyautogui / PyMuPDF などの重い依存が入っていない環境でも、`log_setting` や `path_manager` だけを安全に使えます。

### AIが守るべき import ルール
- 必ず `from base_pyfile import <名前>` の形で import する。`from file_manager import ...` のようなモジュール直下の import はパッケージとしては解決されない（過去の名残で、内部で一部使われていたが修正済み）。
- パッケージ内で相互参照する場合も `from base_pyfile.xxx import yyy` とフルパスで書く。
- 重い依存が必要な機能（画像・ブラウザ・PDF）は、呼び出し側の関数の中で import してよい（遅延importの推奨パターン）。

### 機能と依存の対応表
| やりたいこと | importする名前 | 元モジュール | 追加で必要な主な依存 |
| --- | --- | --- | --- |
| ログ設定 | `make_logger` / `get_log_handler` | `log_setting` | なし（標準ライブラリのみ） |
| 実行時間の計測 | `timer` / `logger_timer` | `function_timer` | なし |
| パス操作・連番 | `unique_path` / `make_directory` / `get_all_files` など | `path_manager` | `natsort`（無くても動作、並び順が劣化） |
| テキスト読み書き | `read_text_file` / `write_file` | `file_manager` | なし |
| LLMの振り分け・生成 | `route_to_model` / `dispatch` / `generate` | `ai_router` | `requests` + Ollamaサーバー |
| クリップボードAI | `watch_clipboard` / `process_clipboard_once` | `ai_clipboard` | `pyperclip`（自動貼付は `pyautogui`） |
| 誤字脱字の校正 | `correct_text` / `select_model_by_length` | `ai_corrector` | `requests` + Ollamaサーバー |
| 翻訳（既定は日本語へ） | `translate_text` / `build_translation_prompt` | `ai_translator` | `requests` + Ollamaサーバー |
| マウス・画像自動化 | `move_and_click` / `search_color` など | `automation_tools` | `opencv-python` `numpy` `pyautogui` `pynput` |
| ブラウザ操作 | `open_page` / `get_urls` / `tab_delete` | `web_open` | `requests` `beautifulsoup4` `tqdm` `pyautogui` |
| PDF/TIFF変換 | `pdf_to_png` / `tiff_to_pdf` など | `pdf_tiff_converter` | `PyMuPDF` `Pillow` |
| エージェント実行 | `run_agent` / `run_repl` | `agent` | **新規作成中（未実装）** |

> `run_agent` / `run_repl` は現在 `base_pyfile/agent.py` を新規作成中です。アクセスするまではエラーになりませんが、モジュールが無い間は import できません。

### 各機能の最小例
```python
from base_pyfile import make_logger, unique_path, read_text_file, write_file

logger = make_logger()                 # 標準出力へ。log_folder=".log" でファイル出力
logger.info("開始")

path = unique_path("out/result_{}.png") # 既存ファイルと衝突しない連番付きパスを返す
write_file("out/memo.txt", "hello")     # 親ディレクトリを自動作成
print(read_text_file("out/memo.txt"))  # UTF-8/SJIS等を自動判別
```

```python
from base_pyfile import route_to_model, dispatch

# 振り分け先だけを決める（Ollamaが必要）
model, ms = route_to_model("フィボナッチ数列を返すPython関数を書いて。")

# 振り分けて実行する
result = dispatch("3人の誕生日が同じになる確率を説明して。")
print(result["model"], result["answer"])
```

### 誤字脱字の校正（コピー → 貼り付けで直す）
`ai_corrector` を使うと、クリップボードの中身を校正してそのまま書き戻せます。使うモデルは **文字数**で決まり、`ai_models.json` の `correction.tiers`（`max_chars` → `model`）で設定します。

```python
from base_pyfile import correct_text, process_clipboard_once

# テキストを直接校正（失敗時は元のテキストが返り、changed=Falseになる）
result = correct_text("これはてすとです。")
print(result["model"], result["corrected"], result["changed"])

# クリップボードを1回だけ校正して書き戻す（そのまま Ctrl+V で直ったテキストが貼れる）
process_clipboard_once(mode="correct")
```

```bash
# クリップボードを監視し、コピーのたびに自動で校正（Ctrl+Cで停止）
python -m base_pyfile.ai_clipboard correct
```

- `watch_clipboard(mode="correct")` / `process_clipboard_once(mode="correct")` で校正モードになります（既定は `"answer"` で従来どおり回答生成）。
- モデルは `select_model_by_length()` が選び、未インストールなら `correction.fallback` → `router.default_target` へ自動で切り替えます。
- モデルが応答しない場合は **元のテキストをそのまま返します**（クリップボードを壊さない）。`changed` で修正があったかを判定できます。
- コード・URL・固有名詞は変更しないようプロンプトで指示し、出力を切り詰めないよう `num_predict` を文字数に応じて確保しています。

```json
"correction": {
  "min_chars": 1,
  "tiers": [
    { "max_chars": 100,  "model": "qwen2.5:0.5b" },
    { "max_chars": 600,  "model": "qwen3.5:9b" },
    { "max_chars": 3000, "model": "gpt-oss:20b" },
    { "max_chars": null, "model": "glm-5.3-flash:cloud" }
  ],
  "fallback": ["qwen2.5-coder:3b", "deepseek-r1:1.5b"]
}
```

### 翻訳（コピー → 貼り付けで翻訳）
`ai_translator` を使うと、クリップボードの中身を指定言語へ翻訳してそのまま書き戻せます。**既定は日本語へ翻訳**です。使うモデルは `ai_corrector` と同じく文字数で決まり、`ai_models.json` の `translation.tiers` で設定します。

```python
from base_pyfile import translate_text, process_clipboard_once

# テキストを直接翻訳（失敗時は元のテキストが返り、changed=Falseになる）
result = translate_text("The quick brown fox jumps over the lazy dog.")  # 既定で日本語へ
print(result["model"], result["translated"], result["changed"])

# 翻訳先を指定する（"en" などの言語コード、または "英語" のような表示名）
result = translate_text("こんにちは", target_language="en")

# クリップボードを1回だけ翻訳して書き戻す（そのまま Ctrl+V で翻訳結果が貼れる）
process_clipboard_once(mode="translate")
```

```bash
# クリップボードを監視し、コピーのたびに自動で日本語へ翻訳（Ctrl+Cで停止）
python -m base_pyfile.ai_clipboard translate

# 翻訳先の言語を指定する（例: 英語へ）
python -m base_pyfile.ai_clipboard translate en
```

- `watch_clipboard(mode="translate")` / `process_clipboard_once(mode="translate")` で翻訳モードになります（既定は `"answer"`）。
- 対応している言語コードは `ai_translator.LANGUAGE_NAMES`（ja / en / zh / ko / de / fr / es / pt）です。未知のコードは指定した文字列がそのままプロンプトに使われます。
- モデルは `select_model_by_length(..., section="translation")` が選び、未インストールなら `translation.fallback` → `router.default_target` へ自動で切り替えます。
- モデルが応答しない場合は **元のテキストをそのまま返します**（クリップボードを壊さない）。`changed` で翻訳が行われたかを判定できます。

```json
"translation": {
  "min_chars": 1,
  "target_language": "ja",
  "tiers": [
    { "max_chars": 200,  "model": "qwen2.5:0.5b" },
    { "max_chars": 1000, "model": "qwen3.5:9b" },
    { "max_chars": 4000, "model": "gpt-oss:20b" },
    { "max_chars": null, "model": "glm-5.3-flash:cloud" }
  ],
  "fallback": ["qwen2.5-coder:3b", "deepseek-r1:1.5b"]
}
```

### .bat ランチャー（ダブルクリック／ショートカット起動）
リポジトリ直下の `ai_clipboard.bat` を使うと、オプション付きでクリップボード監視を起動できます。起動後は常駐し、**クリップボードが更新されるたびに自動で処理**して結果を書き戻します。停止は Ctrl+C。

```bat
ai_clipboard.bat -honyaku       REM コピーした内容を日本語へ翻訳
ai_clipboard.bat -gojidatuji    REM コピーした内容の誤字脱字を校正
ai_clipboard.bat -kaitou        REM コピーした内容をAIへ振り分けて回答
ai_clipboard.bat -honyaku en    REM 翻訳先の言語を指定（既定は ja=日本語）
ai_clipboard.bat -help          REM 使い方を表示
```

- 内部では `python -m base_pyfile.ai_clipboard <mode>` を呼んでいるだけです（`-honyaku`→`translate`、`-gojidatuji`→`correct`、`-kaitou`→`answer`）。
- 先頭の `-` は省略可（`honyaku` でも起動できます）。
- Python は PATH 上の `python` を優先し、無ければ `py` ランチャーを使います。特定のPythonを使いたい場合は bat 内の `set "PY=python"` を書き換えてください。
- bat は自身のフォルダを作業ディレクトリにする（`cd /d "%~dp0"`）ため、`base_pyfile` パッケージが import できます。デスクトップ等からショートカットを張る場合は、この bat 本体を指してください。

### 動作の要点（AIがハマりやすい点）
- `make_logger()` を**同じロガー名で複数回呼んでも、ハンドラは置き換え**られ、ログが二重出力されません。`level` 引数が常に優先されます。
- `get_log_handler()` は `file_path` が実在しなくても動きます。標準外のログレベル（例: 25）でも `LEVEL25` としてファイル名に使え、KeyError になりません。
- `write_file()` は**拡張子が無いパスにだけ**既定の `.txt` を付けます。`data.json` のように既に拡張子があるパスは尊重し、勝手に `.txt` へ書き換えません。戻り値は書き込み後の `Path` です。
- `read_text_file()` は存在しないファイルに対して例外を投げず、空文字（または空リスト）を返します。
- `unique_path()` は「同じパスには同じ接尾辞を再利用」するキャッシュを最大 `path_manager.MAX_EXISTING_FILES`（既定1024）件まで保持し、古いものから捨てます。`reset_existing_files()` で明示的にクリアできます。
- `make_directory(path, is_file=None)` の `is_file` を省略すると、名前にドットを含むかで自動判定します。ドット入りのフォルダ名など誤判定しうる場合は `is_file=` を明示してください。
- `ai_router.generate()` のタイムアウト既定は `GENERATE_TIMEOUT`（120秒）です。一覧取得など軽い問い合わせは `LIST_MODELS_TIMEOUT`（5秒）を使います。生成はモデルのロードを含むと時間がかかるため、短いタイムアウトで `answer=None` になりがちでした（修正済み）。
- `automation_tools.move_and_click()` / `search_color()` は **引数を省略すると「呼び出した瞬間」のマウス位置**を使います（既定値を `None` にして関数内で取得する方式に修正済み）。
- `web_open` の座標・色は意味のある定数（`DEVICE_TOOLBAR_TOGGLE_XY` など）として `web_open.py` 冒頭にまとめています。解像度やブラウザUIが変わったらそこを調整します。
- `ai_clipboard.watch_clipboard()` は無限ループです。`stop_after=N` で回数を制限できます。

### テスト
軽量モジュールの回帰テストは `tests/test_base_pyfile.py` にあります。
```bash
python -m pytest tests -q
```
→ 現在 26 件が通過します。重い依存（Ollama・cv2・PyMuPDF など）は不要です。
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
# ai_router.py
ai_routerは、渡されたテキストを「どのローカルLLMに処理させるか」判断し、選んだモデルへ実際に投げるための振り分け（ルーティング）モジュールです。Ollamaサーバーとの通信もこのモジュールが担当します。

## 概要
1. `ai_models.json` から各モデルの得意・不得意を読み込む
2. Ollama（`/api/tags`）に問い合わせ、実際にインストール済みのモデルだけを候補に絞る
3. ルーターモデル（`router.preferred`、無ければ `router.fallback`）に入力内容を分類させ、ラベル（記号）を1つ選ばせる
4. 選ばれたモデルへ入力内容をそのまま渡して生成する（`dispatch`）

## 使い方
```python
from base_pyfile import (
    dispatch,
    route_to_model,
    load_model_profiles,
    generate,
    list_installed_models,
)

# インストール済みモデルの一覧
print(list_installed_models())

# 振り分け先だけを知りたい場合
model, elapsed_ms = route_to_model("フィボナッチ数列を返すPython関数を書いて。")
print(model, elapsed_ms)

# 振り分けて、そのモデルで実際に生成する場合
result = dispatch("3人の誕生日が同じになる確率を順を追って説明して。")
print(result["model"], result["answer"], result["total_ms"])
```

## 主な関数
`list_installed_models()` / `is_model_installed()` / `resolve_model()` / `generate()` / `load_model_profiles()` / `get_available_profiles()` / `build_label_map()` / `route_to_model()` / `dispatch()`

## 振り分け方式
- ラベル方式（既定, `use_labels=True`）：ルーターには記号だけを答えさせます。`A` は常に「わからない」で固定、`B` 以降がモデル候補です。モデル名を出力させないため小型モデルでも崩れにくく、候補を増やしても「わからない」の位置が動きません。
- 名前方式（`use_labels=False`）：ルーターにモデル名を直接答えさせます。

モデル名の正解は `tev1:0.8b` ですが、以前 `qwen2.5:0.5b` を使っていた環境もあるため、`router.preferred` / `router.fallback` に従って `resolve_model()` が利用可能なルーターモデルを選びます。振り分け先のモデルが未インストールだった場合は `router.default_target` へ切り替えます。

# ai_models.json
ai_models.jsonは、`ai_router` が振り分け判断に使うモデル特性の定義ファイルです。各モデルの得意・不得意・用途を記述しておくことで、振り分け先の判断材料になります。

## 構造
- `router`：振り分けの設定
    - `preferred`：振り分け判断に使うルーターモデル名
    - `fallback`：preferredが使えない場合の代替ルーターモデル
    - `safe_target`：ラベル `A`（わからない）のときに使う安パイモデル
    - `default_target`：判断できなかった場合などに使う既定モデル
- `models`：各モデルの特性のリスト
    - `name`：Ollama上のモデル名
    - `role` / `description`：役割と一言説明
    - `strengths` / `weaknesses`：長所・短所
    - `best_for` / `avoid_for`：向く用途・避けたい用途（`best_for` は振り分けプロンプトに使われる）

## 抜粋
```json
{
  "router": {
    "preferred": "tev1:0.8b",
    "fallback": ["qwen2.5:0.5b"],
    "safe_target": "deepseek-r1:1.5b",
    "default_target": "qwen2.5:0.5b"
  },
  "models": [
    {
      "name": "qwen2.5-coder:3b",
      "role": "code",
      "description": "コード生成に特化した最大モデル。",
      "strengths": ["コード生成", "コードの説明", "デバッグ", "リファクタリング"],
      "best_for": ["プログラムの作成", "コードレビュー", "エラー修正"]
    }
  ]
}
```

## モデルを追加する
`models` に要素を追加し、`name` をOllama上のモデル名（`ollama list` で確認）に合わせます。インストールされていないモデルは自動的に候補から除外されます。プログラム側の編集は不要で、`ai_models.json` だけで振り分け先の候補を変更できます。