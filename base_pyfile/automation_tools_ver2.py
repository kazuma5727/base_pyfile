import os
import sys
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyautogui

from base_pyfile.file_manager import write_file
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import unique_path

# from base_pyfile import get_log_handler, make_logger,unique_path,write_file


logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


def learning_materials(
    x: int,
    y: int,
    width: int = 200,
    height: int = 100,
    classes: int = 0,
    probability: int = 1000,
) -> tuple[int, int]:
    """
    学習用の素材の生成。

    Args:
        x (int): バウンディングボックスの中心のX座標。
        y (int): バウンディングボックスの中心のY座標。
        width (int, optional): バウンディングボックスの幅。デフォルトは200。
        height (int, optional): バウンディングボックスの高さ。デフォルトは100。
        classes (int, optional): クラスラベル。デフォルトは0。
        probability (int): 1/probabilityの割合で画像を保存。デフォルトは1000。

    Returns:
        tuple[int, int]: バウンディングボックスの中心座標 (x, y) のタプル。
    """
    # バウンディングボックスの中心座標を取得
    if np.random.randint(1, probability + 1) == 1:
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        w, h = image.shape[1], image.shape[0]
        text = "{classes} {:7f} {:7f} {:7f} {:7f}\n".format(
            classes,
            x / w,
            y / h,
            width / w,
            height / h,
        )
        learn = unique_path("learn\materials_{}.png")
        cv2.imwrite(str(learn), image)
        # 拡張子をpngから、txtに変更
        write_file(learn.with_suffix(".txt"), text)

    return x, y


def move_and_click(
    x_position: int,
    y_position: int,
    x_error: int = 0,
    y_error: int = 0,
    t: float = None,
    learning_probability=0,
) -> None:
    """
    マウスを指定された位置に移動してクリックします。

    Args:
        x_position (int): X軸の目標位置。
        y_position (int): Y軸の目標位置。
        x_error (int, optional): X軸の誤差。デフォルトは0。
        y_error (int, optional): Y軸の誤差。デフォルトは0。
        t (float, optional): 移動にかかる時間。Noneの場合、自動的に計算されます。デフォルトはNone。
        learning_probability (int): 1/learning_probabilityの割合で画像を保存。デフォルトは0。

    Returns:
        None
    """
    # ランダムな位置を計算
    x = np.random.randint(x_position - x_error, x_position + 1 + x_error)
    y = np.random.randint(y_position - y_error, y_position + 1 + y_error)

    # 現在のマウス位置を取得
    x2, y2 = pyautogui.position()
    # 移動距離を計算
    distance = int(np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2))

    # 時間が指定されていない場合、自動で計算
    if t is None:
        t = distance / 800
        if t > 0.5:
            t = 0.35
        elif t < 0:
            t = 0.1

    # マウスを指定された位置に移動し、クリックする
    pyautogui.moveTo(x, y, duration=t)
    if learning_probability:
        learning_materials(x_position, y_position, probability=learning_probability)

    pyautogui.click(x, y)


def image_cut(
    image: Any, start_col: int, end_col: int, start_row: int, end_row: int
) -> Any:
    """
    画像の一部を切り取る関数。

    Args:
        image (Any): 切り取る対象の画像。NumPy配列として表現されることが想定される。
        start_col (int): 切り取りの開始列のインデックス。
        end_col (int): 切り取りの終了列のインデックス。
        start_row (int): 切り取りの開始行のインデックス。
        end_row (int): 切り取りの終了行のインデックス。

    Returns:
        Any: 指定された領域が切り取られた部分の画像。

    Example:
        >>> image = np.array([[1, 2, 3],
        ...                    [4, 5, 6],
        ...                    [7, 8, 9]])
        >>> image_cut(image, 0, 2, 0, 2)
        array([[1, 2],
               [4, 5]])
    """
    return image[start_row:end_row, start_col:end_col]
    """
    指定された色が含まれる画像上のランダムな位置をクリックします。

    Args:
        R (int): 色の赤成分（0から255の整数）
        G (int): 色の緑成分（0から255の整数）
        B (int): 色の青成分（0から255の整数）
        image (str, optional): 入力画像のファイルパス。デフォルトは空文字列。
        left_right_upper_lower (tuple, optional): 画像の切り取り範囲 (start_col, end_col, start_row, end_row)。デフォルトは空のタプル。
        exclude_radius (int, optional): マウスカーソル周辺の除外半径。デフォルトは70。
        min_size (int, optional): 色の塊として認識する最小サイズ。デフォルトは100。
        bottom (bool, optional): 最下部の塊からランダムにクリックするかどうか。デフォルトはFalse。
        save (str, optional): 抽出された色の塊の保存先フォルダのパス。デフォルトは空文字列。

    Returns:
        tuple[int, int]: クリックする座標 (x, y) のタプル。
    """


def specified_color(
    RGB: tuple,
    image: str = "",
    left_right_upper_Lower: tuple = (),
    label_count=0,
    threshold=3,  # 適宜調整
    exclude_radius: int = 70,
    min_size: int = 100,
    bottom: bool = False,
    save: str = "",
) -> tuple[int, int]:

    R, G, B = RGB
    # 目標色をNumPy配列に変換
    target_color = np.array([B, G, R], dtype=np.uint8)

    # 画像読み込み
    if not image:
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str):
        image = cv2.imread(image)

    if left_right_upper_Lower:
        image = image_cut(image, *left_right_upper_Lower)
        plus_x = left_right_upper_Lower[0]
        plus_y = left_right_upper_Lower[2]
    else:
        plus_x = 0
        plus_y = 0

    # 画像と目標色との距離を計算
    dist = np.linalg.norm(image - target_color, axis=2)

    # 一定距離以下のピクセルをマスク
    mask = dist < threshold

    # マウスカーソル周辺の座標を除外
    if exclude_radius:
        target_x, target_y = pyautogui.position()
        min_x = max(0, target_x - exclude_radius)
        max_x = min(image.shape[1], target_x + exclude_radius)
        min_y = max(0, target_y - exclude_radius)
        max_y = min(image.shape[0], target_y + exclude_radius)
        mask[min_y:max_y, min_x:max_x] = False

    # マスクを用いて元の画像から指定した色を抽出
    detected_color = np.zeros_like(image)
    detected_color[mask] = image[mask]

    # グレースケールに変換
    gray = cv2.cvtColor(detected_color, cv2.COLOR_BGR2GRAY)

    # 二値化
    ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 連結成分のラベリング
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8
    )

    # 面積が最小サイズ未満の塊を削除
    for i, stat in enumerate(stats):
        if stat[4] < min_size:
            labels[labels == i] = 0

    # 削除後の塊を抽出
    result = np.zeros_like(detected_color)
    result[labels != 0] = detected_color[labels != 0]

    # クリック位置をランダムに選択
    dist = np.linalg.norm(result - target_color, axis=2)
    rows, cols = np.where(dist < threshold)
    if save:
        cv2.imwrite(
            str(unique_path(rf"{save}\mask_{{}}.png")),
            result,
        )

    if not len(cols):
        logger.error("not found")
        x, y = pyautogui.position()
        if label_count:
            return [(x, y)]
        else:
            return x, y

    if label_count:
        logger.debug(f"label:{label_count} label数: {num_labels}")
        xy_list = []
        # 各連結成分のエリアとラベルをリストに格納
        areas_and_labels = [
            (stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)
        ]  # ラベル0は背景なので無視

        for idx, (area, label) in enumerate(areas_and_labels):
            if idx >= label_count:
                break
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            logger.info(f"  座標 (x, y) = ({x + w / 2 + plus_x}, {y + h / 2 + plus_y})")
            logger.info(f"  幅 = {w}, 高さ = {h}, 面積 = {area}")
            xy_list.append([x + w / 2 + plus_x, y + h / 2 + plus_y])

        return xy_list

    if bottom:
        bottom_row = np.max(rows)
        bottom_row_indices = np.where(rows == bottom_row)
        Ransom = np.random.randint(0, len(bottom_row_indices))
        bottom_cols = cols[bottom_row_indices[Ransom]][0]
        x = bottom_cols[np.random.randint(0, len(bottom_cols))]
        y = bottom_row
    else:
        x = cols[np.random.randint(0, len(cols))]
        y = rows[np.random.randint(0, len(rows))]
    return x + plus_x, y + plus_y


def specified_labelcolor():
    # スクリーンショットを撮影し、BGR形式に変換
    image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

    # 目標とするRGB値
    target_color = np.array([13, 46, 165], dtype=np.uint8)

    # RGB値と画像の各ピクセルとの距離を計算
    dist = np.linalg.norm(image - target_color, axis=2)

    # 一定距離以下のピクセルをマスク
    threshold = 30  # 適宜調整
    mask = dist < threshold

    # マスクを使用して元の画像から特定の色を抽出
    result = cv2.bitwise_and(image, image, mask=mask.astype(np.uint8))

    # マスクを使用して抽出された領域を白で塗りつぶし、ラベリングを実施
    _, binary_mask = cv2.threshold(mask.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

    # 各ラベルの情報を表示
    for i in range(1, num_labels):  # ラベル0は背景なので無視
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        centroid = centroids[i]


def templates_matching(
    templates,
    image=cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR),
):
    if isinstance(templates, str):
        obj = cv2.imread(templates)

    elif isinstance(templates, np.ndarray):
        obj = templates

    # テンプレートマッチングアルゴリズム
    map_cc = cv2.matchTemplate(image, obj, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(map_cc)

    if max_val > 0.4:
        x = int(max_loc[0] + obj.shape[1] / 2)
        y = int(max_loc[1] + obj.shape[0] / 2)

    else:
        logger.error("not found")
        return pyautogui.position()
    return x, y


def full_templatematching(
    templates,
    image=cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR),
    threshold=0.8,
):
    if isinstance(templates, str):
        obj = cv2.imread(templates)

    elif isinstance(templates, np.ndarray):
        obj = templates

    # テンプレートマッチングを実行
    result = cv2.matchTemplate(image, obj, cv2.TM_CCOEFF_NORMED)

    # 一致箇所を取得
    loc = np.where(result >= threshold)

    # 重複を防ぐためのリストを初期化
    unique_loc = []
    xy_list = []

    # 一致箇所を元画像にマーキング
    for pt in zip(*loc[::-1]):
        # 重複チェック
        is_unique = True
        for existing_pt in unique_loc:
            if (
                abs(existing_pt[0] - pt[0]) < obj.shape[1]
                and abs(existing_pt[1] - pt[1]) < obj.shape[0]
            ):
                is_unique = False
                break
        if is_unique:
            pyautogui.keyDown("ctrl")
            cv2.rectangle(
                image,
                pt,
                (pt[0] + obj.shape[1], pt[1] + obj.shape[0]),
                (0, 255, 0),
                2,
            )
            unique_loc.append(pt)
            # 中央値の座標を計算して出力
            x = pt[0] + obj.shape[1] // 2
            y = pt[1] + obj.shape[0] // 2
            xy_list.append([x, y])

    return xy_list


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    move_and_click(1000, 500)
    gold_color = np.array([105, 253, 192], dtype=np.uint8)
    normal_color = np.array([213, 212, 142], dtype=np.uint8)
    xx, yy = specified_color(gold_color)
