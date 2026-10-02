"""クリップボードのコピー内容をAIへ振り分け、結果を貼り付けられるようにする補助ツール。

起動後、新しくコピーされたテキストを検知すると、内容に応じた処理を行い、
結果をクリップボードへ書き戻します。ユーザーはそのまま Ctrl+V で貼り付けられます。

モードは2つあります（`mode` 引数）。
    - "answer"（既定）: `ai_router` の振り分けで最適なモデルを選び、回答を生成する。
    - "correct": 誤字脱字を文字数に応じたモデルで校正し、**元のテキストを修正したもの**
      をクリップボードへ書き戻す（コピー → 貼り付けで直っている）。

自動で前面アプリへ貼り付けたい場合は `auto_paste=True`（pyautoguiが必要）に
してください。
"""

# --- 標準ライブラリのインポート ---
import sys
import time
from logging import NullHandler, getLogger
from typing import Any, Dict, Optional

# --- 外部ライブラリのインポート ---
import pyperclip

# --- 独自モジュールのインポート ---
from base_pyfile.ai_corrector import correct_text
from base_pyfile.ai_router import dispatch
from base_pyfile.log_setting import get_log_handler, make_logger

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

VALID_MODES = ("answer", "correct")


def _safe_paste() -> str:
    """クリップボードの内容を取得します。失敗した場合は空文字を返します。"""
    try:
        return pyperclip.paste()
    except Exception as e:
        logger.error(f"クリップボードの読み取りに失敗しました: {e}")
        return ""


def _safe_copy(text: str) -> bool:
    """クリップボードへ内容を書き込みます。成功した場合はTrueを返します。"""
    try:
        pyperclip.copy(text)
        return True
    except Exception as e:
        logger.error(f"クリップボードへの書き込みに失敗しました: {e}")
        return False


def answer_text(text: str, use_labels: bool = True, **kwargs: Any) -> Dict[str, Any]:
    """テキストを最適なモデルへ振り分けて回答を生成します。

    Args:
        text (str): 処理させたいテキスト（コピーされた内容）。
        use_labels (bool, optional): Trueならラベル方式、Falseなら名前方式で振り分け。
            デフォルトはTrue。
        **kwargs: dispatch() へ渡す追加のキーワード引数。

    Returns:
        Dict[str, Any]: dispatch() と同じ結果の辞書。
    """
    return dispatch(text, use_labels=use_labels, **kwargs)


def process_text(
    text: str, mode: str = "answer", use_labels: bool = True, **kwargs: Any
) -> Dict[str, Any]:
    """モードに応じてテキストを処理します。

    Args:
        text (str): 処理させたいテキスト。
        mode (str, optional): "answer"（回答生成）または "correct"（誤字脱字の校正）。
        use_labels (bool, optional): answer モード時の振り分け方式。
        **kwargs: 各処理へ渡す追加のキーワード引数。

    Returns:
        Dict[str, Any]: 処理結果の辞書。
    """
    if mode == "correct":
        return correct_text(text, **kwargs)
    if mode == "answer":
        return answer_text(text, use_labels=use_labels, **kwargs)
    raise ValueError(f"mode は {VALID_MODES} のいずれかを指定してください: {mode!r}")


def _replacement_of(result: Dict[str, Any], mode: str) -> Optional[str]:
    """処理結果から、クリップボードへ書き戻すテキストを取り出します。"""
    if mode == "correct":
        return result.get("corrected")
    return result.get("answer")


def paste_answer() -> None:
    """現在のクリップボード内容を、前面のアプリへ貼り付けます。

    pyautogui を利用するため、遅延インポートしています。
    """
    import pyautogui

    pyautogui.hotkey("ctrl", "v")


def process_clipboard_once(
    mode: str = "answer",
    use_labels: bool = True,
    auto_paste: bool = False,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """現在のクリップボード内容を1回だけ処理します。

    Args:
        mode (str, optional): "answer" または "correct"。デフォルトは "answer"。
        use_labels (bool, optional): answer モード時の振り分け方式。デフォルトはTrue。
        auto_paste (bool, optional): Trueなら結果を前面アプリへ自動貼り付けします。
            デフォルトはFalse（クリップボードへ書き戻すのみ）。
        **kwargs: 各処理へ渡す追加のキーワード引数。

    Returns:
        Optional[Dict[str, Any]]: 生成結果の辞書。クリップボードが空の場合はNone。
    """
    text = _safe_paste()
    if not text or not text.strip():
        logger.info("クリップボードが空のため処理をスキップしました。")
        return None

    result = process_text(text, mode=mode, use_labels=use_labels, **kwargs)
    replacement = _replacement_of(result, mode)
    if replacement:
        _safe_copy(replacement)
        if auto_paste:
            paste_answer()
    return result


def watch_clipboard(
    interval: float = 0.5,
    mode: str = "answer",
    use_labels: bool = True,
    auto_paste: bool = False,
    stop_after: Optional[int] = None,
    **kwargs: Any,
) -> int:
    """クリップボードを監視し、新しくコピーされた内容を処理し続けます。

    Args:
        interval (float, optional): クリップボードを確認する間隔(秒)。デフォルトは0.5。
        mode (str, optional): "answer" または "correct"。デフォルトは "answer"。
        use_labels (bool, optional): answer モード時の振り分け方式。デフォルトはTrue。
        auto_paste (bool, optional): Trueなら結果を前面アプリへ自動貼り付けします。
            デフォルトはFalse（クリップボードへ書き戻すのみ）。
        stop_after (Optional[int], optional): 指定した回数だけ処理したら終了します。
            Noneの場合は Ctrl+C まで監視し続けます。
        **kwargs: 各処理へ渡す追加のキーワード引数。

    Returns:
        int: 処理した回数。
    """
    last = _safe_paste()
    processed = 0
    logger.info(f"クリップボードの監視を開始します（mode={mode}, Ctrl+Cで停止）。")

    try:
        while True:
            current = _safe_paste()
            # 直前と異なる内容がコピーされたときだけ処理する
            if current and current != last:
                last = current
                logger.info(f"コピーを検知しました（{len(current)}文字）。")
                result = process_text(
                    current, mode=mode, use_labels=use_labels, **kwargs
                )
                replacement = _replacement_of(result, mode)
                if replacement:
                    _safe_copy(replacement)
                    # 自分の出力を新規コピーとして再処理しないよう記録
                    last = replacement
                    if auto_paste:
                        paste_answer()
                    processed += 1
                    logger.info(
                        f"[{processed}] {result.get('model')} が処理しました。"
                    )
                else:
                    logger.warning("結果を生成できませんでした。")

                if stop_after is not None and processed >= stop_after:
                    break

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("監視を停止しました。")

    return processed


def main() -> None:
    # 使い方: python -m base_pyfile.ai_clipboard [answer|correct]
    mode = sys.argv[1] if len(sys.argv) > 1 else "answer"
    if mode not in VALID_MODES:
        print(f"mode は {VALID_MODES} のいずれかを指定してください。")
        return

    if mode == "correct":
        print("=== クリップボード 誤字脱字 校正ツール ===")
        print("起動後、何かをコピーすると、校正した内容をクリップボードへ書き戻します。")
        print("そのまま Ctrl+V で、直ったテキストが貼り付けられます。停止は Ctrl+C。")
    else:
        print("=== クリップボードAI 振り分けツール ===")
        print("起動後、何かをコピーすると、内容を判断して回答をクリップボードへ書き戻します。")
        print("そのまま Ctrl+V で貼り付けられます。停止は Ctrl+C。")

    watch_clipboard(mode=mode)


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))

    main()
