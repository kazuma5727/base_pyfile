import os
import sys
import time
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import pyautogui
from pynput.mouse import Button, Controller

from base_pyfile.file_manager import write_file
from base_pyfile.log_setting import get_log_handler, make_logger
from base_pyfile.path_manager import unique_path

# from base_pyfile import get_log_handler, make_logger,unique_path,write_file


logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


# 指定の座標に移動してクリックする関数
def fast_click(position: Union[Tuple[int, int], int], y: int = None) -> None:
    """
    指定された座標(x, y)でマウスの左クリックを行う関数。
    座標はタプル(x, y)またはxとyの別々の引数として渡すことができる。

    Args:
        position (Union[Tuple[int, int], int]): クリックするX座標または(X, Y)のタプル。
        y (int, optional): クリックするY座標。positionがタプルの場合は省略。
    """
    # 座標の形式をチェックして適切な座標を設定
    if isinstance(position, tuple):
        x, y = position
    else:
        x = position
        if y is None:
            raise ValueError("Y座標を指定する必要があります")
    # マウスコントローラのインスタンスを作成
    mouse = Controller()

    # マウスカーソルを指定された座標に移動
    mouse.position = (x, y)

    # マウスの左ボタンをクリック
    mouse.click(Button.left, 1)


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
        text = "{} {:7f} {:7f} {:7f} {:7f}\n".format(
            classes,
            x / w,
            y / h,
            width / w,
            height / h,
        )
        learn = unique_path("learn\materials_{}.png")
        cv2.imwrite(str(learn), image)
        # 拡張子をpngから、txtに変更
        write_file(Path(learn).with_suffix(".txt"), text)

    return x, y


def move_and_click(
    x_position: int | tuple[int, int] = pyautogui.position(),
    y_position: int = None,
    x_error: int = 0,
    y_error: int = 0,
    t: float = None,
    accelerator: float = 0,
    ctrl: bool = False,
    learning_probability: int = 0,
) -> None:
    """
    マウスを指定された位置に移動してクリックします。

    Args:
        x_position (int | tuple[int, int]): X軸の目標位置、または(x, y)のタプル。
        y_position (int, optional): Y軸の目標位置。x_positionがタプルの場合は無視されます。
        x_error (int, optional): X軸の誤差。デフォルトは0。
        y_error (int, optional): Y軸の誤差。デフォルトは0。
        t (float, optional): 移動にかかる時間。Noneの場合、自動的に計算されます。デフォルトはNone。
        accelerator (float, optional): 移動時間の最小時間調整値。デフォルトは0。
        learning_probability (int, optional): 1/learning_probabilityの割合で画像を保存。デフォルトは0。

    Returns:
        None
    """
    # x_positionがタプルの場合、それをxとyに分解
    if isinstance(x_position, tuple) or isinstance(x_position, list):
        x, y = x_position
    else:
        x = x_position
        y = y_position

    # x_positionがタプルでなく、y_positionが指定されていない場合はエラーを発生
    if y is None:
        raise ValueError(
            "x_positionがタプルでない場合、y_positionを指定する必要があります"
        )

    # 誤差を考慮してランダムな位置を計算
    if x_error or y_error:
        x = np.random.randint(x - x_error, x + 1 + x_error)
        y = np.random.randint(y - y_error, y + 1 + y_error)

    # 時間が指定されていない場合、自動で計算
    if t is None:
        # 現在のマウス位置を取得
        x2, y2 = pyautogui.position()
        # 移動距離を計算
        distance = int(np.sqrt((x2 - x) ** 2 + (y2 - y) ** 2))

        # 距離に基づいて時間を計算
        t = distance / 900 - accelerator
        if t > 0.5:
            t = 0.35
        elif t < -3:
            t = 0.11
        elif t < 0:
            t = 0.13
        pyautogui.moveTo(x, y, duration=t)
    elif t:
        # マウスを指定された位置に移動し、クリックする
        pyautogui.moveTo(x, y, duration=t)

    if learning_probability:
        # 学習用の画像を保存する関数を呼び出し
        if isinstance(x_position, tuple):
            learning_materials(*x_position, probability=learning_probability)
        else:
            learning_materials(x_position, y_position, probability=learning_probability)

    # 指定位置をクリック
    logger.debug(f"click({x}, {y})")
    if ctrl:
        pyautogui.keyDown("ctrl")
        fast_click(x, y)
        pyautogui.keyUp("ctrl")
    else:
        fast_click(x, y)


def search_color(
    RGB: Union[Tuple[int, int, int], List[int], int, str],
    G: int = None,
    B: int = None,
    xy: Union[Tuple[int, int], List[int], int] = pyautogui.position(),
    y: int = None,
    click: int = 0,
) -> Union[bool, Tuple[int, int, int]]:
    """
    指定された座標の色が与えられたRGB値と一致するかを確認する。

    パラメータ:
        RGB (Union[Tuple[int, int, int], List[int], int, str]): チェックするRGB値または'found'。
        G (int, オプション): RGBが整数の場合の緑成分。
        B (int, オプション): RGBが整数の場合の青成分。
        xy (Union[Tuple[int, int], List[int], int], オプション): チェックする座標。
        y (int, オプション): xyが整数の場合のy座標。
        click (int, オプション): 色が一致した場合にクリックするまでの待機時間。

    戻り値:
        Union[bool, Tuple[int, int, int]]: RGBが'found'の場合は座標の色、そうでない場合は色が一致するかどうかの真偽値。
    """

    # RGB値の処理
    if RGB == "found":
        R = G = B = None
    elif isinstance(RGB, tuple) or isinstance(RGB, list):
        if len(RGB) != 3:
            raise ValueError(
                "RGBは3つの整数からなるタプルまたはリストでなければなりません。"
            )
        R, G, B = RGB
    elif isinstance(RGB, int):
        if G is None or B is None:
            raise ValueError(
                "RGBを個別の整数として提供する場合、GとBも指定する必要があります。"
            )
        R = RGB
    else:
        raise ValueError(
            "RGBは、タプル、リスト、または赤成分を表す整数でなければなりません。"
        )

    # xy値の処理
    if isinstance(xy, tuple) or isinstance(xy, list):
        if len(xy) != 2:
            raise ValueError(
                "xyは2つの整数からなるタプルまたはリストでなければなりません。"
            )
        x, y = xy
    elif isinstance(xy, int):
        if y is None:
            raise ValueError(
                "xyを個別の整数として提供する場合、yを指定する必要があります。"
            )
        x = xy
    else:
        raise ValueError(
            "xyは、タプル、リスト、またはx座標を表す整数でなければなりません。"
        )

    # スクリーンショットを取得し、BGR形式に変換
    image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

    # 指定された座標の色を取得
    color = image[y, x]

    # BGR形式からRGB形式に変換
    logger.debug(f"R: {color[2]}, G: {color[1]}, B: {color[0]}")

    # 座標の色を返す部分を追加
    if RGB == "found":
        return (color[2], color[1], color[0])

    if click == 0:
        return (R, G, B) == (color[2], color[1], color[0])

    start_time = time.time()

    while (
        not (R, G, B) == (color[2], color[1], color[0])
        and time.time() - start_time <= click
    ):
        time.sleep(0.5)
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        color = image[y, x]

    if time.time() - start_time > click:
        return False

    move_and_click(x, y, t=0)
    return (R, G, B) == (color[2], color[1], color[0])


def specified_color(
    RGB: tuple[int, int, int] | list[int] | int,
    G: int = None,
    B: int = None,
    image: np.ndarray = None,
    left_right_upper_Lower: tuple[int, int, int, int] = (),
    label_count: int = 1,
    near_label: tuple[int, int] = (),
    bottom: bool = False,
    threshold: int = 10,
    exclude_radius: int = 70,
    max_size: int = float("inf"),
    min_size: int = 100,
    found: bool = False,
    save: str = "",
) -> tuple[int, int] | list[tuple[int, int]]:
    if isinstance(RGB, (tuple, list)):
        if len(RGB) != 3:
            raise ValueError("RGBは3つの整数からなるタプルまたはリストでなければなりません。")
        R, G, B = RGB
    elif isinstance(RGB, int):
        if G is None or B is None:
            raise ValueError("RGBを個別の整数として提供する場合、GとBも指定する必要があります。")
        R = RGB
    else:
        raise ValueError("RGBは、タプル、リスト、または赤成分を表す整数でなければなりません。")

    if image is None:
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    elif isinstance(image, (str, Path)):
        image = cv2.imread(str(image))

    if left_right_upper_Lower:
        start_col, end_col, start_row, end_row = left_right_upper_Lower
        image = image[start_row:end_row, start_col:end_col]
        plus_x = start_col
        plus_y = start_row
    else:
        plus_x = 0
        plus_y = 0

    target_bgr = (B, G, R)
    lower_bound = np.array([max(0, target_bgr[i] - threshold) for i in range(3)])
    upper_bound = np.array([min(255, target_bgr[i] + threshold) for i in range(3)])

    mask = cv2.inRange(image, lower_bound, upper_bound)

    # 🎯 exclude_radius処理：マウスカーソル付近の除外
    if exclude_radius > 0:
        cursor_x, cursor_y = pyautogui.position()
        cursor_x -= plus_x
        cursor_y -= plus_y
        h, w = mask.shape
        cv2.circle(mask, (cursor_x, cursor_y), exclude_radius, 0, thickness=-1)

    if save:
        cv2.imwrite(str(Path(save) / "mask.png"), mask)

    # 連結成分ラベリング
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    # ラベルが存在しないとき
    if num_labels <= 1:
        logger.error("not found")
        x, y = pyautogui.position()
        ret = (x, y) if label_count == 1 else [(x, y)]
        return (ret, False) if found else ret

    xy_list = []
    areas_and_labels = [
        (stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)
    ]
    areas_and_labels.sort(reverse=True, key=lambda x: x[0])

    for area, label in areas_and_labels:
        if area > max_size:
            continue
        if len(xy_list) >= label_count or area < min_size:
            break

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        w = stats[label, cv2.CC_STAT_WIDTH]
        h = stats[label, cv2.CC_STAT_HEIGHT]

        logger.info(f"座標 (x, y) = ({x + w // 2 + plus_x}, {y + h // 2 + plus_y})")
        logger.info(f"幅 = {w}, 高さ = {h}, 面積 = {area}")
        if bottom:
            xy_list.append((x + w // 2 + plus_x, y + h + plus_y))
        else:
            xy_list.append((x + w // 2 + plus_x, y + h // 2 + plus_y))

    if label_count == 1 and xy_list:
        return xy_list[0]
    elif label_count == 1 and not xy_list:
        x, y = pyautogui.position()
        ret = (x, y) if label_count == 1 else [(x, y)]
        return (ret, False) if found else ret
    elif label_count > 1:
        return xy_list

    # near_labelが使われたとき
    if near_label:
        target_coord = near_label
        min_distance = float("inf")
        closest_label = None
        for i in range(1, num_labels):
            center = centroids[i]
            distance = np.linalg.norm(np.array(target_coord) - center)
            if distance < min_distance:
                min_distance = distance
                closest_label = i

        if closest_label is not None:
            x = stats[closest_label, cv2.CC_STAT_LEFT]
            y = stats[closest_label, cv2.CC_STAT_TOP]
            w = stats[closest_label, cv2.CC_STAT_WIDTH]
            h = stats[closest_label, cv2.CC_STAT_HEIGHT]
            if bottom:
                return x + w // 2 + plus_x, y + h + plus_y
            else:
                return x + w // 2 + plus_x, y + h // 2 + plus_y
        else:
            logger.error("指定座標に近いラベルが見つかりませんでした。")
            return pyautogui.position()

    # ラベルも近傍も指定されていない場合はランダムに選ぶ
    rows, cols = np.where(mask == 255)
    if bottom:
        bottom_row = np.max(rows)
        bottom_row_indices = np.where(rows == bottom_row)
        rand_index = np.random.randint(0, len(bottom_row_indices[0]))
        bottom_col = cols[bottom_row_indices[0][rand_index]]
        x = bottom_col
        y = bottom_row
    else:
        rand_index = np.random.randint(0, len(rows))
        x = cols[rand_index]
        y = rows[rand_index]

    return x + plus_x, y + plus_y

def specified_color_fast_ver(
    RGB: tuple[int, int, int] | list[int] | int,
    G: int = None,
    B: int = None,
    image: np.ndarray = None,
    left_right_upper_Lower: tuple[int, int, int, int] = (),
    label_count: int = 1,
    near_label: tuple[int, int] = (),
    bottom: bool = False,
    threshold: int = 10,  # 適宜調整
    exclude_radius: int = 70,
    max_size: int = float("inf"),
    min_size: int = 100,
    found: bool = False,
    save: str = "",
) -> tuple[int, int]:
    """
    指定した色を画像内で検出し、その位置を返します。

    Args:
        RGB (tuple[int, int, int] | list[int] | int): 検出する色のRGB値。
        G (int, optional): 色の緑成分。RGBがタプルやリストの場合は省略。
        B (int, optional): 色の青成分。RGBがタプルやリストの場合は省略。
        image (np.ndarray, optional): 読み込む画像。省略時はスクリーンショットを使用。
        left_right_upper_Lower (tuple[int, int, int, int], optional): 画像の一部を切り抜く範囲。
        label_count (int, optional): 検出するラベルの数。
        near_label (tuple[int, int], optional): 指定した座標に最も近いラベルを取得。
        bottom (bool, optional): 最も下にあるピクセルを取得するかどうか。
        threshold (int, optional): 色の距離の閾値。
        exclude_radius (int, optional): マウスカーソル周辺の除外半径。
        max_size (int, optional): 検出する最大サイズ。デフォルトは無限大。
        min_size (int, optional): 検出する最小サイズ。デフォルトは100。
        save (str, optional): マスク画像の保存先パス。

    Returns:
        tuple[int, int]: 検出された色の位置。
    """
    # RGB値の処理
    if isinstance(RGB, tuple) or isinstance(RGB, list):
        if len(RGB) != 3:
            raise ValueError(
                "RGBは3つの整数からなるタプルまたはリストでなければなりません。"
            )
        R, G, B = RGB
    elif isinstance(RGB, int):
        if G is None or B is None:
            raise ValueError(
                "RGBを個別の整数として提供する場合、GとBも指定する必要があります。"
            )
        R = RGB
    else:
        raise ValueError(
            "RGBは、タプル、リスト、または赤成分を表す整数でなければなりません。"
        )

    # 目標色をNumPy配列に変換
    target_color = np.array([B, G, R], dtype=np.uint8)

    # 画像読み込み
    if image is None:
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
    elif isinstance(image, str) or isinstance(image, Path):
        image = cv2.imread(str(image))

    # 画像の一部を切り抜く
    if left_right_upper_Lower:
        start_col, end_col, start_row, end_row = left_right_upper_Lower
        image = image[start_row:end_row, start_col:end_col]
        plus_x = start_col
        plus_y = start_row

    else:
        plus_x = 0
        plus_y = 0

    # 画像と目標色との距離を計算
    dist = np.linalg.norm(image - target_color, axis=2)

    # 一定距離以下のピクセルをマスク
    mask = dist < threshold

    # マウスカーソル周辺の座標を除外
    if exclude_radius:
        x, y = pyautogui.position()
        target_x = x - plus_x
        target_y = y - plus_y
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

        if label_count > 1:
            ret = [(x, y)]
        else:
            ret = x, y

        if found:
            return ret, False
        else:
            return ret

    if label_count:
        logger.debug(f"label:{label_count} label数: {num_labels}")
        xy_list = []
        # 各連結成分のエリアとラベルをリストに格納
        areas_and_labels = [
            (stats[i, cv2.CC_STAT_AREA], i) for i in range(1, num_labels)
        ]  # ラベル0は背景なので無視

        # エリアの大きい順にソート
        areas_and_labels.sort(reverse=True, key=lambda x: x[0])

        for area, label in areas_and_labels:
            if area > max_size:
                continue
            if len(xy_list) >= label_count or area < min_size:
                break

            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            logger.info(
                f"  座標 (x, y) = ({x + w // 2 + plus_x}, {y + h // 2 + plus_y})"
            )
            logger.info(f"幅 = {w}, 高さ = {h}, 面積 = {area}")
            if bottom:
                xy_list.append([x + w // 2 + plus_x, y + h + plus_y])
            else:
                xy_list.append([x + w // 2 + plus_x, y + h // 2 + plus_y])
        if label_count == 1 and xy_list:
            return xy_list[0]

        elif label_count == 1 and not xy_list:
            return pyautogui.position()

        else:
            return xy_list

    if near_label:
        target_coord = pyautogui.position()
        # 指定した座標に最も近いラベルを取得
        min_distance = float("inf")
        closest_label = None
        for i in range(1, num_labels):
            center = centroids[i]
            distance = np.linalg.norm(np.array(target_coord) - center)
            if distance < min_distance:
                min_distance = distance
                closest_label = i

        # 最も近いラベルの領域のみを残し、それ以外を黒く塗りつぶす
        result = np.zeros_like(result)
        if closest_label is not None:
            result[labels == closest_label] = result[labels == closest_label]
            x = stats[closest_label, cv2.CC_STAT_LEFT]
            y = stats[closest_label, cv2.CC_STAT_TOP]
            width = stats[closest_label, cv2.CC_STAT_WIDTH]
            height = stats[closest_label, cv2.CC_STAT_HEIGHT]
            center_x = x + width // 2
            center_y = y + height // 2

            logger.debug(f"中央座標: ({center_x}, {center_y})")
            logger.debug(f"幅: {width}")
            logger.debug(f"高さ: {height}")
        else:
            logger.error("指定した座標に有効なラベルが見つかりませんでした。")
        if bottom:
            return center_x + plus_x, y + height + plus_y
        else:
            return center_x + plus_x, center_y + plus_y

    if bottom:
        bottom_row = np.max(rows)
        bottom_row_indices = np.where(rows == bottom_row)
        rand_index = np.random.randint(0, len(bottom_row_indices))
        bottom_col = cols[bottom_row_indices[rand_index]][0]
        x = bottom_col
        y = bottom_row
    else:
        rand_index = np.random.randint(0, len(rows))
        x = cols[rand_index]
        y = rows[rand_index]
    return x + plus_x, y + plus_y


def templates_matching(
    templates: str | np.ndarray,
    image: np.ndarray = None,
    left_right_upper_Lower: tuple = (),
    threshold: float = 0.4,
    found: bool = False,
) -> tuple[int, int]:
    """
    テンプレートマッチングを用いて画像内のテンプレートの位置を検出します。

    Args:
        templates (str | np.ndarray): テンプレート画像のパスまたはテンプレート画像。
        image (np.ndarray, optional): 検出対象の画像。省略時はスクリーンショットを使用。
        left_right_upper_Lower (tuple, optional): 画像の切り取り範囲 (start_col, end_col, start_row, end_row)。デフォルトは空のタプル。

    Returns:
        tuple[int, int]: 検出されたテンプレートの中心位置。
    """
    if image is None:
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

    if isinstance(templates, str):
        obj = cv2.imread(templates)
    elif isinstance(templates, np.ndarray):
        obj = templates
    else:
        raise ValueError("Invalid template type. Must be a file path or a NumPy array.")

    if left_right_upper_Lower:
        start_col, end_col, start_row, end_row = left_right_upper_Lower
        image = image[start_row:end_row, start_col:end_col]
        plus_x = start_col
        plus_y = start_row
    else:
        plus_x = 0
        plus_y = 0

    # テンプレートマッチングアルゴリズム
    map_cc = cv2.matchTemplate(image, obj, cv2.TM_CCOEFF_NORMED)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(map_cc)

    if max_val > threshold:
        x = int(max_loc[0] + obj.shape[1] // 2) + plus_x
        y = int(max_loc[1] + obj.shape[0] // 2) + plus_y
    else:
        logger.error("not found")
        if found:
            return pyautogui.position(), False
        else:
            return pyautogui.position()
    return x, y


def full_templatematching(
    templates: str | np.ndarray,
    image: np.ndarray = cv2.cvtColor(
        np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR
    ),
    threshold: float = 0.8,
) -> list[list[int]]:
    """
    テンプレートマッチングを用いて画像内のテンプレートの全ての位置を検出します。

    Args:
        templates (str | np.ndarray): テンプレート画像のパスまたはテンプレート画像。
        image (np.ndarray): 検出対象の画像。省略時はスクリーンショットを使用。
        threshold (float): マッチングの閾値。これ以上の値を持つ場所を一致箇所とする。

    Returns:
        list[list[int]]: 検出されたテンプレートの中心位置のリスト。
    """
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
    time.sleep(0.5)

    move_and_click()
    gold_color = (105, 253, 192)
    normal_color = (213, 212, 142)
    xx, yy = specified_color(109, 69, 59, label_count=1)
