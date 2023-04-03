import json
import os
import re
import site
import subprocess
import time
import wave
from functools import cache
from logging import NullHandler, getLogger
from typing import Dict, List, Optional

import requests

try:
    from tqdm import tqdm

    progress = lambda iterable: tqdm(iterable)
except ImportError:
    tqdm = lambda iterable, *args, **kwargs: iterable
    progress = tqdm


from base_pyfile import (
    get_files,
    get_log_handler,
    logger_timer,
    make_logger,
    read_text_file,
    unique_path,
    write_file,
)

logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())


# speakerデータを取得する関数
def _get_speaker_data() -> List[Dict]:
    """スピーカーデータを取得する"""
    if not hasattr(_get_speaker_data, "cache"):
        for i in range(10):
            try:
                host = "localhost"
                port = 50021
                # APIからデータを取得する
                response = requests.get(f"http://{host}:{port}/speakers")
                response.raise_for_status()
                _get_speaker_data.cache = response.json()
                # 取得したデータを返す
                return _get_speaker_data.cache

            except:
                # エラーが発生した場合はリトライする
                interval = 1 + i / 10
                logger.info(f"出力失敗{interval}秒待ちます")
                time.sleep(interval)
                continue
        else:
            logger.error("出力できなかったため、終了します")
            raise ValueError
    else:
        # キャッシュに保存されたデータを返す
        return _get_speaker_data.cache


# 全スピーカーの名前のリストを取得する関数
def get_all_speaker_names() -> List[str]:
    """全スピーカーの名前のリストを取得する"""
    speaker_data = _get_speaker_data()
    # スピーカーデータから名前を取得し、リストにして返す
    return [speaker["name"] for speaker in speaker_data]


# スピーカーのIDから名前を取得する関数
def get_speaker_name_by_id(speaker_id: int = 3) -> Optional[str]:
    """スピーカーのIDから名前を取得する"""
    speaker_data = _get_speaker_data()
    for speaker in speaker_data:
        for style in speaker.get("styles", []):
            if style.get("id") == speaker_id:
                # IDが一致したスピーカーの名前を返す
                return speaker["name"]
    # 見つからなかった場合はNoneを返す
    return None


# ノーマルスピーカーのIDと名前の辞書を取得する関数
def get_normal_speaker_names() -> Dict[int, str]:
    """ノーマルスピーカーのIDと名前の辞書を取得する"""
    speaker_data = _get_speaker_data()
    # スピーカーデータから"ノーマル"という名前のスタイルを持つスピーカーのIDと名前を辞書にして返す
    normal_speakers = {
        style.get("id"): speaker.get("name")
        for speaker in speaker_data
        for style in speaker.get("styles", [])
        if style.get("name") == "ノーマル"
    }
    return normal_speakers


def get_speaker_name_by_normal_id(speaker_id: int = 3) -> Optional[str]:
    """ノーマルスピーカーのIDから名前を取得する"""
    normal_speaker_names = get_normal_speaker_names()
    name = normal_speaker_names.get(speaker_id)
    if name is not None:
        return name

    max_speaker_id = max(normal_speaker_names.keys())
    while speaker_id <= max_speaker_id:
        speaker_id += 1
        name = normal_speaker_names.get(speaker_id)
        if name is not None:
            return name
    return None


@cache
def call_speaker_name_by_id(speaker_id: int = 3) -> int:
    """スピーカーのIDから名前を取得する"""
    # キャラクターデータを取得し、ノーマルスピーカーのIDと名前を取得する
    normal_speakers = get_normal_speaker_names()

    while True:
        # 検索したIDがノーマルスピーカーのIDの範囲内にある場合、そのIDの名前を返す
        if speaker_id in normal_speakers and speaker_id <= max(normal_speakers.keys()):
            logger.info(f"ノーマルスピーカーのID {speaker_id} が見つかりました。")
            return speaker_id
        # 検索したIDがノーマルスピーカーのIDの範囲外の場合、最初のIDに戻って再検索する
        elif speaker_id > max(normal_speakers.keys()):
            logger.warning("全てのキャラクターを検索しました。最初のキャラクターに戻ります。")
            speaker_id = 1
        # 検索したIDがノーマルスピーカーのIDではない場合、次のIDを検索する
        else:
            speaker_id += 1
            logger.debug(f"{speaker_id} はノーマルスピーカーのIDではありません。次のIDに移動します。")


@logger_timer()
def generate_wav(text: str, speaker: int = 3, output_path: str = "audio.wav") -> bool:
    """
    テキストから音声を生成してwavファイルを出力する
    Args:
        text (str): 音声に変換するテキスト
        speaker (int, optional): 声の種類を表す数字。デフォルトは1。
        output_path (str, optional): wavファイルの出力先パス。デフォルトは"audio"。

    Returns:
        bool: wavファイルの出力に成功したらTrue, 失敗したらFalseを返す
    """

    host = "localhost"
    port = 50021
    params = (
        ("text", text),
        ("speaker", speaker),
    )
    for i in range(10):
        try:
            with requests.Session() as session:
                response1 = session.post(f"http://{host}:{port}/audio_query", params=params)
                response2 = session.post(
                    f"http://{host}:{port}/synthesis",
                    headers={"Content-Type": "application/json"},
                    params=params,
                    data=json.dumps(response1.json()),
                )
                with wave.open(output_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(response2.content)
            return True

        except:
            interval = 1 + i / 10
            logger.info(f"出力失敗{interval}秒待ちます")
            time.sleep(interval)
            continue
    else:
        logger.error("出力できなかったため、終了します")
        return False


def VOICEVOX_output(
    path_or_text: str, speaker: int = 8, output_dir: str = "", delimiter="\n", progressbar=progress
) -> bool:
    """
    テキストファイルを読み込み、1行ずつVOICEVOXで音声を生成します。

    Args:
        path_or_text (str): テキストファイルのパスまたは文字列。
        speaker (int, optional): 声の種類を表す数字。デフォルトは8。
        output_dir (Optional[str], optional): 音声ファイルの出力先ディレクトリ。指定しない場合はカレントディレクトリに保存されます。
        delimiter (str, optional): テキストファイルの区切り文字。デフォルトは"\\n"。

    Returns:
        bool: すべての行が正常に処理された場合はTrue、それ以外の場合はFalseを返します。
    """

    if os.path.exists(path_or_text):
        text = read_text_file(path_or_text)

    else:
        text = path_or_text

    if not output_dir and os.path.exists(path_or_text):
        output_dir = f"{os.path.splitext(path_or_text)[0] }_{{}}"

    elif not output_dir:
        output_dir = f"音声{{}}"

    elif os.path.isfile(output_dir):
        output_dir = os.path.splitext(output_dir)[0] + "_{}"

    voice_number = 0

    if isinstance(text, str):
        text = text.replace("＝", "、")
        text = text.replace("》", "")
        huri = "｜.*?《"
        for rub in re.findall(huri, text):
            text = text.replace(rub, "")

        text = text.split(delimiter)
    # 桁数を取得
    zero_count = r"{:0" + str(len(str(len(text)))) + r"}_{}.wav"
    subprocess.Popen([r"D:\0soft\VOICEVOX\VOICEVOX.exe"], shell=True)
    _e = 0
    for e, t in progressbar(enumerate(text)):
        t = t.split()
        if not t:
            _e -= 1
            continue

        # もし、文字列tに「、「（」または「(」のいずれかが含まれている場合、voice変更
        if any(c in "「（(" for c in t):
            voice_number += 1

        else:
            voice_number = 0

        voice_number = call_speaker_name_by_id(speaker + voice_number) - speaker

        success = generate_wav(
            t,
            speaker + voice_number,
            unique_path(
                os.path.join(
                    output_dir,
                    zero_count.format(e + _e, get_speaker_name_by_id(speaker + voice_number)),
                )
            ),
        )
        if not success:
            logger.error(f'"{text}" の音声生成に失敗しました。')
            return success
    return success


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    path = r"E:\Lexar\短編.txt"

    VOICEVOX_output(path, delimiter="\n")


# VOICEBOX_global
# C:\python\vpython\Lib\site-packages\urllib3\response.py
# 630行目
# try:
#     import global_value as Vg
#     kana=data.decode('utf-8')
#     Vg.g_kana=kana[kana[:-2].find("kana")+7:-2]
# except:
#     pass
