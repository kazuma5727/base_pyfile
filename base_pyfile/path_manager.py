import os
import re
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

from base_pyfile.log_setting import get_log_handler, make_logger

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

# unique_path が「同じパスには同じ接尾辞を再利用する」ために保持するキャッシュ。
# 無制限に増えないよう、最大 MAX_EXISTING_FILES 件を超えた分は古いものから破棄する。
MAX_EXISTING_FILES = 1024
existing_files: Dict[str, int] = {}


def _touch_existing_path(file_path: str, counter: int) -> int:
    """接尾辞キャッシュを更新し、上限を超えた古いエントリを破棄する。

    Args:
        file_path (str): キャッシュのキーとなるファイルパス。
        counter (int): 新規登録時の初期連番。

    Returns:
        int: 更新後の連番。
    """
    if file_path in existing_files:
        # 直近で使われたので末尾（=新しい側）へ移動
        existing_files[file_path] = existing_files.pop(file_path)
    else:
        existing_files[file_path] = counter

    # dictは挿入順を保持するため、先頭が最も古いエントリ
    while len(existing_files) > MAX_EXISTING_FILES:
        oldest = next(iter(existing_files))
        del existing_files[oldest]

    return existing_files[file_path]


def reset_existing_files() -> None:
    """unique_path が保持する接尾辞キャッシュをクリアする。

    バッチ処理の区切りなど、命名の連番をリセットしたいときに呼ぶ。
    """
    existing_files.clear()


def unique_path(
    file_path: str,
    counter: int = 1,
    suffix: Optional[str] = "_",
    existing_text: Optional[str] = "",
    existing_image=None,
) -> str:
    """ファイルパスを連番にする関数
    Args:
        file_path (str): ファイルパス
        counter (int): ファイルパスの接尾辞に付く連番
        suffix (Optional[str]): ファイルパスの接尾辞の文字列
        existing_text (Optional[str]): 既存のテキストファイルが存在する場合、ファイルが同じであるかを確認するための文字列
        existing_image (Optional[np.ndarray]): 既存の画像ファイルが存在する場合、ファイルが同じであるかを確認するためのndarray

    Returns:
        str: 一意になったファイルパス
    """
    # Pathオブジェクトの場合、文字列に変換する
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    else:
        file_path = str(file_path)

    # すでに存在するファイルのリストにファイルパスを追加する
    # （同じパスを再び渡した場合は同じ連番を再利用する）
    _touch_existing_path(file_path, counter)

    # ファイル名と拡張子を分離する
    base, ext = os.path.splitext(file_path)
    suffix = suffix or ""

    # テンプレート（{} を含む）か、通常のファイル名かに応じて候補を作る。
    # 以前は候補以外のパス（例: a{}.txt に対する a.txt）も存在確認に含めていたため、
    # a.txt があるだけで無限ループになることがあった。
    has_placeholder = bool(re.search(r"{.*?}", file_path))

    def _candidate() -> str:
        if has_placeholder:
            return file_path.format(existing_files[file_path])
        return f"{base}{suffix}{existing_files[file_path]}{ext}"

    candidate = _candidate()

    # 通常のファイル名で、まだファイルがない場合は元のパスをそのまま返す。
    # このとき接尾辞付きの候補（a_1.txt）ではなく a.txt を返す必要がある。
    if not has_placeholder and not os.path.exists(file_path):
        logger.debug(f"path先に{file_path}ファイルはありませんでした")
        if ext:
            make_directory(os.path.dirname(file_path))
        else:
            make_directory(file_path)
        return file_path

    while os.path.exists(candidate):
        if existing_text:
            # 同一テキストファイル確認
            try:
                from base_pyfile.file_manager import read_text_file

                before_text = read_text_file(candidate)
                if before_text == existing_text:
                    logger.info("同じテキストがあります")
                    return candidate
            except ImportError:
                logger.error("module_pathが通っていない可能性があります")

        if existing_image is not None:
            # 同一画像ファイル確認
            try:
                import cv2
                import numpy as np

                if isinstance(existing_image, np.ndarray) and np.array_equal(
                    cv2.imread(candidate),
                    existing_image,
                ):
                    logger.info("同じ画像があります")
                    return candidate
            except ImportError:
                logger.warning("画像検索できません")

        existing_files[file_path] += 1
        candidate = _candidate()

    if ext:
        make_directory(os.path.dirname(candidate))
    else:
        make_directory(candidate)

    return candidate


def _make_directory(directory: Path) -> None:
    """ディレクトリを作成する。既に存在する場合は何もしない。

    以前は ``@cache`` で作成済みパスを記憶していたが、作成後に
    ディレクトリが削除されると再作成できなかったためキャッシュを廃止した。
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"{directory}のディレクトリを作成しました")


def make_directory(path: Union[str, Path], is_file: bool = False) -> Path:
    """指定されたパスのディレクトリを作成します。

    Args:
        path (str or Path): 作成するディレクトリのパス。
        is_file (bool, optional): True の場合、``path`` をファイルパスとみなし、
            その親ディレクトリを作成します。既定は False（常にディレクトリとして扱う）。
            従来の「名前にドットを含むかで自動判定」は、``folder.with.dot`` の
            ようなディレクトリ名を誤判定するため廃止しました。ファイルパスを
            渡す場合は明示的に ``is_file=True`` を指定してください。

    Returns:
        Path: 渡されたパスをそのまま返します。
    """
    path_obj = Path(path)
    directory = path_obj.parent.absolute() if is_file else path_obj.absolute()
    _make_directory(directory)

    return path_obj


def get_files(directory: Path, choice_key: str = "") -> List[Path]:
    """フォルダー内にあるすべてのファイルを絶対パスでリストとして返す。

    Args:
        directory (Path): ファイルまたはフォルダーの絶対パス
        choice_key (str): ファイル名に含まれる必要のあるキーワード

    Returns:
        List[Path]: ファイルパスのリスト。choice_keyが指定されている場合は、キーワードが含まれるファイルのみをリスト化する。
    """
    # 絶対パスを取得する
    directory = Path(directory).resolve()

    return (
        [
            files
            for files in natsorted(directory.iterdir())
            if files.is_file() and choice_key in files.name
        ]
        if directory.is_dir()
        else [directory] if directory.is_file() and choice_key in directory.name else []
    )


def get_all_subfolders(
    directory: Union[str, Path], depth: Optional[int] = None
) -> List[Path]:
    """
    指定されたディレクトリ以下の全てのフォルダを再帰的に検索し、
    フォルダパスのリストを返す。

    Args:
        directory (Union[str, Path]): 検索対象のディレクトリパス
        depth (Optional[int]): 検索する階層数。0 または 1 は直下のみ、
            2 以上はその階層まで、None は全階層を検索する。

    Returns:
        List[Path]: ディレクトリパスのリスト（自然順にソートされている）
    """

    def get_subfolders(directory: Path, depth: Optional[int]) -> List[Path]:
        """
        指定されたディレクトリ以下のフォルダを再帰的に検索し、
        フォルダパスのリストを返す。

        Args:
            directory (Path): 検索対象のディレクトリパス
            depth (Optional[int]): 検索する階層数。None の場合は全階層。

        Returns:
            List[Path]: ディレクトリパスのリスト
        """
        subfolders = []
        for entry in directory.iterdir():
            if not entry.is_dir():
                continue

            # depth は「ルート直下を1階層」と数える。0 は旧仕様との互換のため
            # 直下のみを返す。depth=1 で子の子まで返していた以前の off-by-one を修正。
            if depth is None:
                subfolders.extend(get_subfolders(entry, None))
            elif depth > 1:
                subfolders.extend(get_subfolders(entry, depth - 1))
            subfolders.append(entry)

        return subfolders

    directory = Path(directory).resolve()
    return natsorted(get_subfolders(directory, depth)) if directory.is_dir() else []


def get_all_files(
    directory: Union[str, Path], choice_key: str = "", depth: Optional[int] = None
) -> List[Path]:
    """
    指定されたディレクトリ以下の全てのファイルを再帰的に検索し、
    ファイルパスのリストを返す。

    Args:
        directory (Union[str, Path]): 検索対象のディレクトリパス
        choice_key (str): ファイル名に含まれる必要のあるキーワード
        depth (Optional[int]): 検索する階層数。Noneの場合、全階層を検索する。

    Returns:
        List[Path]: ファイルパスのリスト（自然順にソートされている）
    """
    file_paths = get_files(directory, choice_key=choice_key)
    for folder in get_all_subfolders(directory, depth):
        files = get_files(folder, choice_key=choice_key)
        # ファイルだけを抽出する
        file_paths.extend(files)

    if natsorted == sorted:
        logger.warning(
            "sort関数を使用しているため、予期せぬ並び順になっている場合があります"
        )
    return natsorted(file_paths)


def get_folders_and_files(directory: Union[str, Path]) -> List[Path]:
    """
    指定されたディレクトリに入っているフォルダとファイルをリストで返す。
    Args:
        directory (Union[str, Path]): フォルダパスを表す文字列またはPathオブジェクト

    Returns:
        list[Path]: フォルダおよびファイルのパスを表すPathオブジェクトのリスト
    """
    directory = Path(directory).resolve()
    return get_all_subfolders(directory, 0) + get_files(directory)


def find_empty_folders(
    folder_list: Union[str, Path, List[Union[str, Path]]],
) -> List[Path]:
    """
    渡されたフォルダのリストから、空のフォルダを探してリストで返す。

    Args:
        folder_list (Union[str, Path, List[Union[str, Path]]]): 調査するフォルダのリスト

    Returns:
        List[Path]: 空のフォルダのリスト
    """
    # フォルダがリストでない場合、リストに変換する
    if isinstance(folder_list, (str, Path)):
        folder_list = [folder_list]

    empty_folders = []
    for folder in folder_list:
        folder = Path(folder)
        if not get_folders_and_files(folder):
            empty_folders.append(folder)
    return empty_folders


def sanitize_windows_filename(non_regular_path: Union[str, Path]) -> Path:
    """
    Windowsのファイル名として不正な文字を正規化します。

    Args:
        non_regular_path (Union[str, Path]): 正規化する前のパス（文字列またはPathオブジェクト）

    Returns:
        Path: 不正な文字が正規化された後のパス（Pathオブジェクト）

    使用例:
        >>> sanitize_windows_filename("C:/Users/User/Documents<>file.txt")
        PosixPath('C:/Users/User/Documents＜＞file.txt')
    """

    # 不正な文字を正規な文字に置換するための変換テーブルを作成します
    translation_table = str.maketrans(
        {
            "<": "＜",
            ">": "＞",
            ":": "：",
            '"': "”",
            # "/": "／",
            # "\\": "＼",
            "|": "｜",
            "?": "？",
            "*": "＊",
        }
    )

    # 入力のパス文字列をPathオブジェクトに変換します
    non_regular_path = Path(non_regular_path)

    # ファイル名の不正な文字を置換して、正規のファイル名を取得します。
    # 拡張子側にも不正な文字が含まれることがあるため、stem と suffix を分けて処理します。
    base_name = non_regular_path.stem.translate(translation_table).strip().rstrip(" .")
    suffix = non_regular_path.suffix.translate(translation_table).strip()
    if not base_name:
        base_name = "_"

    # 特定の文字列はWindowsファイルシステムで予約されており、ファイル名として使用できないため、
    # 予約された文字列の場合は"_file"を付加して回避します
    if base_name.upper() in [
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ]:
        base_name += "_file"

    # 置換と予約文字列の処理を終えた正規化されたファイル名を元のパスと結合します。
    # with_name() は親ディレクトリを保持するため、parent を再度結合しません。
    return non_regular_path.with_name(base_name + suffix)


def get_latest_folder(directory: str) -> Path:
    """
    指定されたディレクトリ内で最も最近更新されたフォルダーの絶対パスを取得します。

    Parameters:
        directory (str): 最も最近のフォルダーを検索するディレクトリのパス。

    Returns:
        Path: 最も最近更新されたフォルダーの絶対パス。
    """
    # フォルダーのリストを取得
    directory_path = Path(directory)
    folders = [folder for folder in directory_path.iterdir() if folder.is_dir()]

    # フォルダーの更新時刻を取得して最も新しいものを見つける
    latest_folder = max(folders, key=lambda folder: folder.stat().st_mtime)

    # 最も新しいフォルダーの絶対パスを返す
    return latest_folder


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # sample: カレントディレクトリ以下のサブフォルダーを表示
    for folder in get_all_subfolders(Path.cwd()):
        print(folder)
