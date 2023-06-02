import os
import sys
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from natsort import natsorted

# from ruamel import yaml
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


filename = "data.txt"

lines = ["74 0.639063 0.613889 0.204688 0.2", "0 0.523828 0.229861 0.199219 0.456944"]
# lines = ["74 0.639063 0.613889 0.204688 0.2"]


def reannotation_file(file_path, list_range=[0]):
    lines = read_text_file(file_path, "\n")
    annotation = ""
    for line in lines:
        if line:
            data = line.split()
            if int(data[0]) in list_range:
                annotation += data + "\n"

    return annotation


def zahyou(file_path, list_range=[0]):
    lines = read_text_file(file_path, "\n")

    for line in lines:
        if line:
            data = line.split()
            # print(data)
            if int(data[0]) in list_range:
                result = (float(data[1]), float(data[2]))

    return result


import os


def copy_lines_with_zero(input_file_list, output_folder):
    for file_path in input_file_list:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, file_name)

        with open(file_path, "r") as file_in:
            with open(output_file_path, "w") as file_out:
                for line in file_in:
                    if line.startswith("0"):
                        file_out.write(line)


# 使用例
input_file_path = get_files(r"C:\AI_projectg\yolov5\runs\detect\exp22\labels")  # 入力ファイルのパス
output_file_path = r"C:\AI_projectg\yolov5\runs\detect\exp"  # 出力ファイルのパス

# copy_lines_with_zero(input_file_path, output_file_path)

import os


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

    for input_txt in tqdm(natsorted(learn_file)):
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
            data_yaml = read_text_file(classes, "\n")[]

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

    yaml_create(r"C:\AI_projectg\yolov5\runs\detect\exp", "png")
