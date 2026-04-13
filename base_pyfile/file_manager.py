"""ファイル操作に関するユーティリティ関数を提供します。

ファイルの読み書き、バックアップ作成など、基本的なファイル管理タスクを
簡単かつ安全に行うための関数群を含みます。
文字コードの自動判別や、書き込み時の自動バックアップなど、
煩雑になりがちな処理をカプセル化しています。
"""

# --- 標準ライブラリのインポート ---
import datetime
import os
import shutil
from logging import NullHandler, getLogger
from pathlib import Path
from typing import List, Optional, Tuple, Union

# --- 独自モジュールのインポート ---
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import get_files, make_directory, unique_path

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def read_text_file(
    file_path: Union[str, Path],
    delimiter: Optional[str] = None,
    return_encoding: bool = False,
) -> Union[str, List[str], Tuple[Union[str, List[str]], str]]:
    """テキストファイルを読み込み、その内容を返します。

    複数の一般的なエンコーディング（UTF-8, Shift_JISなど）を自動的に試し、
    読み込みに成功した内容を返します。区切り文字を指定して内容を分割することも可能です。

    Args:
        file_path (Union[str, Path]): 読み込むファイルのパス。
        delimiter (Optional[str], optional): テキストを分割する区切り文字。Noneの場合、全文を一つの文字列で返します。
        return_encoding (bool, optional): Trueの場合、(内容, エンコーディング名) のタプルを返します。

    Returns:
        Union[str, List[str], Tuple[Union[str, List[str]], str]]:
            - `delimiter=None`: ファイル全体の文字列。
            - `delimiter`指定時: 分割された文字列のリスト。
            - `return_encoding=True`: (上記の結果, エンコーディング名) のタプル。
            - 読み込み失敗時: 空の文字列またはリスト。
    """
    file_path = Path(file_path)
    # 試行するエンコーディングのリスト
    encodings = ["utf-8", "Shift_JIS", "euc_jp", "iso2022_jp", "cp932"]
    text = ""
    used_encoding = ""

    for enc in encodings:
        try:
            with file_path.open("r", encoding=enc) as f:
                text = f.read()
            used_encoding = enc
            logger.debug(f"'{file_path}' をエンコーディング '{enc}' で読み込みました。")
            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    else:
        logger.warning(f"'{file_path}' をどのエンコーディングでも開けませんでした。空の文字列を返します。")
        # 読み込み失敗時の返り値をdelimiterの有無で分岐
        if return_encoding:
            return ([], "") if delimiter else ("", "")
        return [] if delimiter else ""

    # 区切り文字で分割
    if delimiter:
        lines = text.split(delimiter)
        # 末尾が空要素であれば削除（splitの仕様による）
        if lines and lines[-1] == "":
            lines.pop()
        result = lines
    else:
        result = text

    if return_encoding:
        return result, used_encoding
    else:
        return result


def write_file(
    file_path: Union[str, Path],
    write_text: str = "",
    extension: str = ".txt",
    file_encoding: str = "utf-8",
    write_mode: str = "w",
    backup: bool = True,
) -> None:
    """指定されたパスにテキストを書き込みます。

    書き込み前に、既存ファイルとの内容を比較し、変更がある場合のみ
    自動でバックアップを作成する機能を持ちます。

    Args:
        file_path (Union[str, Path]): 書き込み先のファイルパス。
        write_text (str, optional): 書き込むテキストデータ。デフォルトは空文字列。
        extension (str, optional): 本来意図する拡張子。`file_path`の拡張子と異なる場合は修正されます。
        file_encoding (str, optional): 書き込み時の文字エンコーディング。デフォルトは"utf-8"。
        write_mode (str, optional): 書き込みモード。'w'(上書き)または'a'(追記)を指定。デフォルトは'w'。
        backup (bool, optional): Trueの場合、上書き前に内容が異なればバックアップを作成します。デフォルトはTrue。
    """
    file_path = Path(file_path)
    write_text = str(write_text)

    # 拡張子が意図したものと違う場合は修正
    if not extension.startswith("."):
        extension = "." + extension
    if file_path.suffix != extension:
        logger.info(f"拡張子を '{file_path.suffix}' から '{extension}' に変更します。")
        file_path = file_path.with_suffix(extension)

    # 親ディレクトリが存在しない場合は作成
    make_directory(file_path.parent)

    # バックアップ処理
    if backup and file_path.exists():
        # 追記モードでない、かつファイル内容が異なる場合のみバックアップ
        if write_mode == "w" and read_text_file(file_path) != write_text:
            logger.info(f"ファイル内容が異なるため、'{file_path}' のバックアップを作成します。")
            backup_file(file_path)
        else:
            logger.debug(f"'{file_path}' は内容が同一か追記モードのため、バックアップはスキップします。")

    # ファイルへの書き込み
    try:
        with file_path.open(mode=write_mode, encoding=file_encoding) as f:
            f.write(write_text)
        logger.debug(f"'{file_path}' にテキストを保存しました (モード: {write_mode})。")
    except IOError as e:
        logger.error(f"'{file_path}' への書き込みに失敗しました: {e}")
    return file_path

def backup_file(file_path: Union[str, Path}, use_datestamp: bool = True) -> None:
    """指定されたファイルのバックアップを作成します。

    バックアップは、元のファイルと同じディレクトリ内の `backup` サブディレクトリに、
    重複しないファイル名で保存されます。

    Args:
        file_path (Union[str, Path]): バックアップ対象のファイルパス。
        use_datestamp (bool, optional): Trueの場合、ファイル名に日付を付与します。デフォルトはTrue。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"バックアップ対象ファイル '{file_path}' が存在しません。")
        return

    # バックアップディレクトリのパスを生成
    backup_dir = file_path.parent / "backup"
    make_directory(backup_dir)

    # バックアップファイル名の生成
    if use_datestamp:
        jst = datetime.timezone(datetime.timedelta(hours=9), "JST")
        timestamp = datetime.datetime.now(jst).strftime("%y%m%d_%H%M%S")
        backup_base_name = f"{file_path.stem}_{timestamp}"
    else:
        backup_base_name = f"{file_path.stem}_backup"

    # 重複を避けたユニークなパスを生成
    backup_path_template = backup_dir / f"{backup_base_name}{{}}{file_path.suffix}"
    final_backup_path = unique_path(str(backup_path_template))

    # ファイルをコピーしてバックアップを作成
    try:
        shutil.copy2(file_path, final_backup_path)
        logger.info(f"'{file_path}' のバックアップを '{final_backup_path}' に作成しました。")
    except IOError as e:
        logger.error(f"バックアップの作成に失敗しました: {e}")


if __name__ == "__main__":
    # このスクリプトが直接実行された場合のテストコード
    logger = make_logger(handler=get_log_handler(10))  # ログレベルをDEBUGに設定

    # --- テスト用のファイルとディレクトリを準備 ---
    test_dir = Path("./test_file_manager")
    make_directory(test_dir)
    test_file = test_dir / "test.txt"
    test_file_sjis = test_dir / "test_sjis.txt"

    # --- write_file のテスト ---
    print("--- write_file test ---")
    write_file(test_file, "こんにちは、世界！", backup=False)
    # SJISで書き込み
    write_file(test_file_sjis, "テスト文字列", file_encoding="shift_jis", backup=False)

    # --- read_text_file のテスト ---
    print("\n--- read_text_file test ---")
    content_utf8 = read_text_file(test_file)
    print(f"UTF-8 ファイルの内容: {content_utf8}")
    content_sjis, enc = read_text_file(test_file_sjis, return_encoding=True)
    print(f"Shift_JIS ファイルの内容: {content_sjis} (エンコーディング: {enc})")

    # --- backup_file のテスト ---
    print("\n--- backup_file test ---")
    write_file(test_file, "更新された内容") # 内容を更新してバックアップをトリガー
    backup_file(test_file)
    backup_file(test_file, use_datestamp=False)

    print(f"\nテスト完了。'{test_dir}' とその中のファイルを確認してください。")
