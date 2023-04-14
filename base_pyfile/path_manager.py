import os
import re
from functools import cache
from logging import NullHandler, getLogger
from pathlib import Path
from typing import List, Optional, Union

try:
    from natsort import natsorted
except ImportError:
    natsorted = sorted

from base_pyfile.log_setting import get_log_handler, make_logger

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


existing_files = {}


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
    global existing_files

    # Pathオブジェクトの場合、文字列に変換する
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    else:
        file_path = str(file_path)

    # すでに存在するファイルのリストにファイルパスを追加する
    if not file_path in existing_files:
        existing_files[file_path] = counter

    # ファイル名と拡張子を分離する
    base, ext = os.path.splitext(file_path)

    # ファイル名が存在しない場合、そのまま返す
    if not re.findall(r"{.*?}", file_path):
        new_path = base + suffix + "{}" + ext
        check_path = base + "{}" + ext
        return_path = file_path
    else:
        new_path = file_path
        check_path = file_path
        return_path = file_path.format(existing_files[file_path])

    if not (
        os.path.exists(new_path.format(""))
        or os.path.exists(new_path.format(existing_files[file_path]))
        or os.path.exists(check_path.format(""))
        or os.path.exists(check_path.format(existing_files[file_path]))
    ):
        logger.debug(f"path先に{return_path}ファイルはありませんでした")
        if ext:
            make_directory(os.path.dirname(return_path))
        else:
            make_directory(return_path)
        return return_path

    while os.path.exists(new_path.format(existing_files[file_path])):
        if existing_text:
            # 同一テキストファイル確認
            try:
                if new_path.format(existing_files[file_path]) and existing_text:
                    from file_manager import read_text_file

                    before_text = read_text_file(new_path.format(existing_files[file_path]))
                    if before_text == existing_text:
                        logger.info("同じテキストがあります")
                        return new_path.format(existing_files[file_path])
            except ImportError:
                logger.error("module_pathが通っていない可能性があります")

        if existing_image:
            # 同一画像ファイル確認
            try:
                import cv2
                import numpy as np

                if new_path.format(existing_files[file_path]) and isinstance(
                    existing_image, np.ndarray
                ):
                    if np.array_equal(
                        cv2.imread(new_path.format(existing_files[file_path])), existing_image
                    ):
                        logger.info("同じ画像があります")
                        return new_path.format(existing_files[file_path])
            except ImportError:
                logger.warning("画像検索できません")

        existing_files[file_path] += 1
    if ext:
        make_directory(os.path.dirname(new_path.format(existing_files[file_path])))
    else:
        make_directory(new_path.format(existing_files[file_path]))

    return new_path.format(existing_files[file_path])


@cache
def make_directory(path):
    """指定されたパスのディレクトリを作成します。

    Args:
        path (str or Path): 作成するディレクトリのパス

    Returns:
        str or Path: 渡されたパスをそのまま返します。
    """
    path_obj = Path(path)
    if "." in path_obj.name:
        directory = path_obj.parent.absolute()
    else:
        directory = path_obj.absolute()

    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"{directory}のディレクトリを作成しました")

    return path


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
        else [directory]
        if directory.is_file() and choice_key in directory.name
        else []
    )


def get_all_subfolders(directory: Union[str, Path], depth: Optional[int] = None) -> List[Path]:
    """
    指定されたディレクトリ以下の全てのフォルダを再帰的に検索し、
    フォルダパスのリストを返す。

    Args:
        directory (Union[str, Path]): 検索対象のディレクトリパス
        depth (Optional[int]): 検索する階層数。Noneの場合、全階層を検索する。

    Returns:
        List[Path]: ディレクトリパスのリスト（自然順にソートされている）
    """

    def get_subfolders(directory: Path, depth: Optional[int]) -> List[Path]:
        """
        指定されたディレクトリ以下のフォルダを再帰的に検索し、
        フォルダパスのリストを返す。

        Args:
            directory (Path): 検索対象のディレクトリパス
            depth (Optional[int]): 検索する階層数。Noneの場合、全階層を検索する。

        Returns:
            List[Path]: ディレクトリパスのリスト
        """
        subfolders = []
        for entry in directory.iterdir():
            if entry.is_dir():
                if depth is None or depth >= 1:
                    subfolders.extend(get_subfolders(entry, depth - 1 if depth else None))
                subfolders.append(entry)

        return subfolders

    directory = Path(directory).resolve()
    return [natsorted(get_subfolders(directory, depth))] if directory.is_dir() else []


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
        logger.warning("sort関数を使用しているため、予期せぬ並び順になっている場合があります")
    return natsorted(file_paths)


def get_folders_and_files(directory: Union[str, Path]) -> list[Path]:
    """
    指定されたディレクトリに入っているフォルダとファイルをリストで返す。
    Args:
        directory (Union[str, Path]): フォルダパスを表す文字列またはPathオブジェクト

    Returns:
        list[Path]: フォルダおよびファイルのパスを表すPathオブジェクトのリスト
    """

    directory = Path(directory).resolve()
    return get_all_subfolders(directory, 0) + get_files(directory)


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # sample
    # print(get_files(r""))
    # print(get_all_subfolders(r""))
    directory=Path(r"F:\Lexar")


    print(get_folders_and_files(directory))
