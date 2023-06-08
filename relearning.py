import os
import sys
from logging import NullHandler, getLogger
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from natsort import natsorted
from tqdm import tqdm

from base_pyfile import (
    get_all_files,
    get_all_subfolders,
    get_files,
    get_folders_and_files,
    get_log_handler,
    logger_timer,
    make_directory,
    make_logger,
    read_text_file,
    unique_path,
    write_file,
)

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def reannotation(file_path: str, class_range: List[str] = ["0"]) -> str:
    """
    指定されたファイルのデータを再注釈します。

    Args:
        file_path (str): ファイルのパス
        class_range (List[str], optional): 対象とするクラスの範囲. デフォルトは ["0"].

    Returns:
        str: 再注釈されたデータの結果

    """
    lines = read_text_file(file_path, "\n")
    retation = ""
    for line in lines:
        if line:
            data = line.split()
            if (
                data[0] in class_range
                and data[1]
                and data[2]
                and float(data[3]) > 0.1
                and float(data[4]) > 0.1
            ):
                retation = " ".join(data[:5])

    return retation


def recycle_annotation_file(
    src: Union[str, Path], dst: Union[str, Path] = "new", class_range: List[str] = ["0"]
) -> None:
    """
    注釈ファイルを再利用して新しいファイルに保存します。

    Args:
        src (Union[str, Path]): 元のファイルまたはディレクトリのパス
        dst (Union[str, Path], optional): 出力先のディレクトリのパス. デフォルトは "new".
        class_range (List[str], optional): 対象とするクラスの範囲. デフォルトは ["0"].

    Returns:
        None

    """
    if not isinstance(src, Path):
        src = Path(src)

    if dst == "new":
        dst = Path(src).parent / dst
        dst.mkdir(parents=True, exist_ok=True)
    else:
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)

    for file_path in get_files(src, ".txt"):
        annotation = reannotation(file_path, class_range)
        if annotation:
            write_file(dst / file_path.name, annotation)


def copy_lines_with_zero(input_file_list, output_folder):
    for file_path in input_file_list:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, file_name)

        with open(file_path, "r") as file_in:
            with open(output_file_path, "w") as file_out:
                for line in file_in:
                    if line.startswith("0"):
                        file_out.write(line)


def remove_empty_files(folder_path):
    for file_name in os.listdir(folder_path):
        delete = False
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and os.path.splitext(file_path)[1] == ".txt":
            with open(file_path, "r") as file:
                content = file.read().strip()
                if not content:
                    delete = True
        if delete:
            os.remove(file_path)


# 使用例
folder_path = r"C:\AI_projectg\yolov5\runs\detect\exp"  # フォルダのパス

# remove_empty_files(folder_path)


def yaml_create(input_path: Union[str, Path], extension: str = "", data_mode: str = "w") -> None:
    """
    概要:
        指定されたディレクトリ内のテキストファイルを元に、txtファイルとdata.yamlを作成します。

    引数:
        input_path (str): テキストファイルを検索するディレクトリのパス。
        extension (str, optional): 画像ファイルの拡張子。指定された場合、同じ名前の画像ファイルを探します。
        data_mode (str, optional): 書き込みモード。デフォルトは "w"。

    戻り値:
        None
    """
    learn_file = get_files(input_path)

    output_list = []
    dir_path = Path(input_path)

    train_path = dir_path / "train.txt"
    valid_path = dir_path / "valid.txt"
    test_path = dir_path / "test.txt"
    yaml_path = dir_path / ".." / "data.yaml"
    logger.info(f"{dir_path}にtxtファイルを作成します")

    for input_txt in tqdm(learn_file):
        if input_txt.suffix != ".txt":
            continue

        if extension:
            if not "." in extension:
                extension = "." + extension

            input_images = input_txt.parent / (input_txt.stem + extension)
            if input_images in learn_file:
                output_list.append(f"{input_images}\n{input_txt}\n")

        else:
            for image_extension in (".jpg", ".png", ".jpeg"):
                input_images = input_txt.stem + image_extension
                if input_images in learn_file:
                    output_list.append(f"{input_images}\n{input_txt}\n")
                    break

    # 中身シャッフル
    logger.debug("中身シャッフル")
    rng = np.random.default_rng()
    shuffle = rng.permutation(output_list)

    train_list = shuffle[: int(len(shuffle) * 0.8)]
    valid_list = shuffle[int(len(shuffle) * 0.8) : int(len(shuffle) * 0.9)]
    test_list = shuffle[int(len(shuffle) * 0.9) :]

    write_file(train_path, "".join(natsorted(train_list)))
    write_file(valid_path, "".join(natsorted(valid_list)))
    write_file(test_path, "".join(natsorted(test_list)))

    # classesからdata.yamlのデータ作成
    for classes in learn_file:
        if "classes" in classes.as_posix():
            data_yaml = read_text_file(classes, "\n")[:-1]

    # なぜかここでやらないとうまくいかなかった(コードの書き方に問題があるのかも)
    from ruamel import yaml

    yaml_content = yaml.load(
        """
    train: training
    val: validation
    test: test

    nc: 0
    names: class_name
    """,
        Loader=yaml.Loader,
    )

    yaml_content["train"] = str(Path(train_path).resolve())
    yaml_content["val"] = str(Path(valid_path).resolve())
    yaml_content["test"] = str(Path(test_path).resolve())
    yaml_content["nc"] = len(data_yaml)
    yaml_content["names"] = data_yaml

    # new_yaml = yaml.dump(yaml_content, Dumper=yaml.RoundTripDumper)
    with open(yaml_path, "w") as stream:
        yaml.dump(yaml_content, Dumper=yaml.RoundTripDumper, stream=stream)


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    # input_paths = get_files(r"F:\AI_project_y\train_A775")

    yaml_create(r"F:\AI_project_y\labels\new - コピー", "png")
