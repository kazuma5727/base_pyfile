"""クリップボードのコピー内容をAIへ振り分け、結果を貼り付けられるようにする補助ツール。

起動後、新しくコピーされたテキストを検知すると、内容に応じた処理を行い、
結果をクリップボードへ書き戻します。ユーザーはそのまま Ctrl+V で貼り付けられます。

モードは3つあります（`mode` 引数）。
    - "answer"（既定）: `ai_router` の振り分けで最適なモデルを選び、回答を生成する。
    - "correct": 誤字脱字を文字数に応じたモデルで校正し、**元のテキストを修正したもの**
      をクリップボードへ書き戻す（コピー → 貼り付けで直っている）。
    - "translate": 文字数に応じたモデルで **日本語へ翻訳**し、翻訳結果を
      クリップボードへ書き戻す（`target_language` で翻訳先を変更可能）。

安全に試すための機能:
    - ``--preview``: 結果をクリップボードへ書き戻さず、差分を表示する。
    - ``--undo``: 直前の処理前のクリップボード内容へ戻す。
    - ``--history``: 最近の処理履歴を表示する（既定で最新100件を保存）。
    - ``--hotkey``: クリップボード監視の代わりに、グローバルホットキーで1回だけ処理する。

自動で前面アプリへ貼り付けたい場合は `auto_paste=True`（pyautoguiが必要）に
してください。
"""

# --- 標準ライブラリのインポート ---
import argparse
import datetime
import difflib
import json
import os
import sys
import threading
import time
from logging import NullHandler, getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# --- 外部ライブラリのインポート ---
import pyperclip

# --- 独自モジュールのインポート ---
from base_pyfile.dev.ai_corrector import correct_text
from base_pyfile.dev.ai_router import MODEL_PROFILE_ENV, dispatch
from base_pyfile.dev.ai_translator import translate_text
from base_pyfile.file_manager import atomic_write
from base_pyfile.log_setting import get_log_handler, make_logger

# --- ロガーの初期設定 ---
logger = getLogger("log").getChild(__name__)
logger.addHandler(NullHandler())

VALID_MODES = ("answer", "correct", "translate")
HISTORY_LIMIT = 100
DEFAULT_HOTKEY = "<ctrl>+<alt>+c"


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


def _default_history_path() -> Path:
    """OSごとの既定の履歴ファイルパスを返します。"""
    if os.name == "nt":
        root = os.environ.get("LOCALAPPDATA")
        base = Path(root) if root else Path.home() / "AppData" / "Local"
    else:
        root = os.environ.get("XDG_DATA_HOME")
        base = Path(root) if root else Path.home() / ".local" / "share"
    return base / "base_pyfile" / "clipboard_history.jsonl"


def _resolve_history_path(history_path: Optional[Any] = None) -> Path:
    return Path(history_path) if history_path is not None else _default_history_path()


def _read_history(path: Path) -> List[Dict[str, Any]]:
    """JSON Lines形式の履歴を読みます。壊れた行は無視します。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    except OSError as e:
        logger.error(f"履歴の読み込みに失敗しました: {e}")
        return []

    entries: List[Dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("履歴に壊れた行があるためスキップしました。")
            continue
        if isinstance(entry, dict):
            entries.append(entry)
    return entries


def _write_history(path: Path, entries: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
    atomic_write(path, text)


def _record_history(
    text: str,
    result: str,
    mode: str,
    model: Optional[str],
    history_path: Optional[Any] = None,
) -> None:
    """処理前後のテキストを履歴へ1件追加します。"""
    path = _resolve_history_path(history_path)
    entries = _read_history(path)
    entries.append(
        {
            "timestamp": datetime.datetime.now()
            .astimezone()
            .isoformat(timespec="seconds"),
            "mode": mode,
            "model": model,
            "original": text,
            "result": result,
        }
    )
    if len(entries) > HISTORY_LIMIT:
        entries = entries[-HISTORY_LIMIT:]
    try:
        _write_history(path, entries)
    except OSError as e:
        # 履歴保存の失敗で、クリップボードへの書き戻し自体を失敗させない。
        logger.error(f"履歴の保存に失敗しました: {e}")


def load_history(
    limit: int = 10, history_path: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """最近の処理履歴を新しい順ではなく、古い順のまま返します。

    Args:
        limit (int, optional): 取得する最大件数。0以下なら空リスト。
        history_path (str | Path, optional): 履歴ファイル。省略時はOS既定の場所。

    Returns:
        List[Dict[str, Any]]: ``timestamp`` / ``mode`` / ``model`` / ``original`` /
        ``result`` を持つ辞書のリスト（最大 ``limit`` 件）。
    """
    if limit <= 0:
        return []
    entries = _read_history(_resolve_history_path(history_path))
    return entries[-limit:]


def undo_last(history_path: Optional[Any] = None) -> Optional[str]:
    """直前の処理前のテキストをクリップボードへ戻し、その履歴を削除します。

    Args:
        history_path (str | Path, optional): 履歴ファイル。省略時はOS既定の場所。

    Returns:
        Optional[str]: クリップボードへ戻したテキスト。履歴がない場合はNone。
    """
    path = _resolve_history_path(history_path)
    entries = _read_history(path)
    if not entries:
        logger.info("戻せる履歴がありません。")
        return None

    original = entries[-1].get("original")
    if not isinstance(original, str) or not original:
        logger.warning("直前の履歴に元のテキストがないため戻せません。")
        return None

    if not _safe_copy(original):
        return None

    entries.pop()
    try:
        _write_history(path, entries)
    except OSError as e:
        logger.error(f"履歴の更新に失敗しました: {e}")
    logger.info("クリップボードを直前の処理前の内容へ戻しました。")
    return original


def format_diff(original: str, result: str) -> str:
    """変更前後のテキストをunified diff形式にします。"""
    diff = difflib.unified_diff(
        original.splitlines(),
        result.splitlines(),
        fromfile="変更前",
        tofile="変更後",
        lineterm="",
    )
    return "\n".join(diff)


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
        mode (str, optional): "answer"（回答生成）、"correct"（誤字脱字の校正）、
            "translate"（翻訳）。
        use_labels (bool, optional): answer モード時の振り分け方式。
        **kwargs: 各処理へ渡す追加のキーワード引数（translate なら target_language など）。

    Returns:
        Dict[str, Any]: 処理結果の辞書。
    """
    if mode == "correct":
        return correct_text(text, **kwargs)
    if mode == "translate":
        return translate_text(text, **kwargs)
    if mode == "answer":
        return answer_text(text, use_labels=use_labels, **kwargs)
    raise ValueError(f"mode は {VALID_MODES} のいずれかを指定してください: {mode!r}")


def _replacement_of(result: Dict[str, Any], mode: str) -> Optional[str]:
    """処理結果から、クリップボードへ書き戻すテキストを取り出します。"""
    if mode == "correct":
        return result.get("corrected")
    if mode == "translate":
        return result.get("translated")
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
    copy_result: bool = True,
    save_history: bool = True,
    history_path: Optional[Any] = None,
    source_text: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """現在のクリップボード内容を1回だけ処理します。

    Args:
        mode (str, optional): "answer"、"correct"、"translate" のいずれか。
            デフォルトは "answer"。
        use_labels (bool, optional): answer モード時の振り分け方式。デフォルトはTrue。
        auto_paste (bool, optional): Trueなら結果を前面アプリへ自動貼り付けします。
            デフォルトはFalse（クリップボードへ書き戻すのみ）。
        copy_result (bool, optional): Falseなら結果をクリップボードへ書き戻さず、
            プレビューだけ行います。この場合、履歴にも保存しません。
        save_history (bool, optional): Trueなら処理前後のテキストを履歴へ保存します。
        history_path (str | Path, optional): 履歴ファイル。省略時はOS既定の場所。
        source_text (str, optional): 処理対象のテキスト。省略時はクリップボードから
            読み取ります。プレビューで読み取り回数を1回にしたい場合に使います。
        **kwargs: 各処理へ渡す追加のキーワード引数。

    Returns:
        Optional[Dict[str, Any]]: 生成結果の辞書。クリップボードが空の場合はNone。
    """
    text = _safe_paste() if source_text is None else source_text
    if not text or not text.strip():
        logger.info("クリップボードが空のため処理をスキップしました。")
        return None

    result = process_text(text, mode=mode, use_labels=use_labels, **kwargs)
    replacement = _replacement_of(result, mode)
    if not replacement:
        logger.warning("結果を生成できませんでした。")
        return result

    if not copy_result:
        logger.info("プレビューのため、クリップボードは変更しません。")
        return result

    if not _safe_copy(replacement):
        logger.warning("結果をクリップボードへ書き戻せませんでした。")
        return result

    if auto_paste:
        try:
            paste_answer()
        except Exception as e:
            logger.error(f"自動貼り付けに失敗しました: {e}")

    if save_history:
        _record_history(
            text,
            replacement,
            mode,
            result.get("model"),
            history_path=history_path,
        )
    return result


def watch_clipboard(
    interval: float = 0.5,
    mode: str = "answer",
    use_labels: bool = True,
    auto_paste: bool = False,
    stop_after: Optional[int] = None,
    copy_result: bool = True,
    save_history: bool = True,
    history_path: Optional[Any] = None,
    **kwargs: Any,
) -> int:
    """クリップボードを監視し、新しくコピーされた内容を処理し続けます。

    Args:
        interval (float, optional): クリップボードを確認する間隔(秒)。デフォルトは0.5。
        mode (str, optional): "answer"、"correct"、"translate" のいずれか。
            デフォルトは "answer"。
        use_labels (bool, optional): answer モード時の振り分け方式。デフォルトはTrue。
        auto_paste (bool, optional): Trueなら結果を前面アプリへ自動貼り付けします。
            デフォルトはFalse（クリップボードへ書き戻すのみ）。
        stop_after (Optional[int], optional): 指定した回数だけ処理したら終了します。
            Noneの場合は Ctrl+C まで監視し続けます。
        copy_result (bool, optional): Falseなら結果をクリップボードへ書き戻しません。
        save_history (bool, optional): Trueなら処理前後のテキストを履歴へ保存します。
        history_path (str | Path, optional): 履歴ファイル。省略時はOS既定の場所。
        **kwargs: 各処理へ渡す追加のキーワード引数。

    Returns:
        int: 処理した回数。
    """
    if interval <= 0:
        raise ValueError("interval は 0 より大きい値を指定してください。")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after は 1 以上を指定してください。")

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
                    if not copy_result:
                        logger.info(
                            f"[{processed + 1}] プレビューのみ（クリップボードは変更しません）。"
                        )
                    elif _safe_copy(replacement):
                        # 自分の出力を新規コピーとして再処理しないよう記録
                        last = replacement
                        if auto_paste:
                            try:
                                paste_answer()
                            except Exception as e:
                                logger.error(f"自動貼り付けに失敗しました: {e}")
                        if save_history:
                            _record_history(
                                current,
                                replacement,
                                mode,
                                result.get("model"),
                                history_path=history_path,
                            )
                        logger.info(
                            f"[{processed + 1}] {result.get('model')} が処理しました。"
                        )
                    else:
                        # 書き戻しに失敗した場合、同じクリップボード内容を
                        # 無限に再処理しないよう last は current のままにする。
                        logger.warning("結果をクリップボードへ書き戻せませんでした。")
                    # 書き戻しに失敗しても stop_after が機能するよう、
                    # 生成に成功した時点で処理回数を数える。
                    processed += 1
                else:
                    logger.warning("結果を生成できませんでした。")

                if stop_after is not None and processed >= stop_after:
                    break

            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("監視を停止しました。")

    return processed


def watch_hotkey(
    hotkey: str = DEFAULT_HOTKEY,
    mode: str = "answer",
    use_labels: bool = True,
    auto_paste: bool = False,
    copy_result: bool = True,
    save_history: bool = True,
    history_path: Optional[Any] = None,
    stop_after: Optional[int] = None,
    on_result: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    **kwargs: Any,
) -> int:
    """グローバルホットキーが押されたときだけクリップボードを処理します。

    クリップボードを常時監視しないため、コピーのたびに勝手に書き換わるのを
    避けたい場合に使います。``pynput`` が必要です（省略可能な依存）。

    Args:
        hotkey (str, optional): pynput形式のホットキー。既定は
            ``"<ctrl>+<alt>+c"``。
        mode (str, optional): "answer"、"correct"、"translate" のいずれか。
        use_labels (bool, optional): answer モード時の振り分け方式。
        auto_paste (bool, optional): Trueなら結果を前面アプリへ自動貼り付けします。
        copy_result (bool, optional): Falseなら結果をクリップボードへ書き戻しません。
        save_history (bool, optional): Trueなら処理前後のテキストを履歴へ保存します。
        history_path (str | Path, optional): 履歴ファイル。省略時はOS既定の場所。
        stop_after (Optional[int], optional): 指定回数処理したら終了します。
            Noneの場合は Ctrl+C まで待ち受けます。
        on_result (callable, optional): 処理後に ``on_result(元のテキスト, 結果)``
            として呼ばれるコールバック。プレビュー表示などに使います。
        **kwargs: 各処理へ渡す追加のキーワード引数。

    Returns:
        int: 処理した回数。

    Raises:
        RuntimeError: ``pynput`` がインストールされていない場合。
    """
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after は 1 以上を指定してください。")

    try:
        from pynput import keyboard
    except ImportError as e:
        raise RuntimeError(
            "ホットキー機能には pynput が必要です。pip install pynput を実行してください。"
        ) from e

    processed = 0
    stop_event = threading.Event()
    lock = threading.Lock()

    def on_activate() -> Optional[bool]:
        nonlocal processed
        with lock:
            text = _safe_paste()
            try:
                result = process_clipboard_once(
                    mode=mode,
                    use_labels=use_labels,
                    auto_paste=auto_paste,
                    copy_result=copy_result,
                    save_history=save_history,
                    history_path=history_path,
                    source_text=text,
                    **kwargs,
                )
            except Exception as e:
                logger.exception(f"ホットキー処理に失敗しました: {e}")
                return None
            if result is None:
                return None
            if on_result is not None:
                try:
                    on_result(text, result)
                except Exception as e:
                    logger.exception(f"ホットキーの結果表示に失敗しました: {e}")
            processed += 1
            logger.info(f"[{processed}] ホットキーで処理しました。")
            if stop_after is not None and processed >= stop_after:
                stop_event.set()
                # pynput ではコールバックが False を返すとリスナーが停止する。
                return False
        return None

    listener = keyboard.GlobalHotKeys({hotkey: on_activate})
    listener.start()
    logger.info(f"ホットキー {hotkey} を待ち受けています（Ctrl+Cで停止）。")

    try:
        while listener.running and not stop_event.is_set():
            stop_event.wait(0.2)
    except KeyboardInterrupt:
        logger.info("ホットキーの待ち受けを停止しました。")
    finally:
        listener.stop()

    return processed


def _print_result(text: str, result: Dict[str, Any], mode: str, as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    replacement = _replacement_of(result, mode)
    if mode == "answer":
        print(replacement or "（回答を生成できませんでした）")
        return

    print("--- 変更前 ---")
    print(text)
    print("--- 変更後 ---")
    print(replacement or "（結果を生成できませんでした）")
    if replacement:
        diff = format_diff(text, replacement)
        if diff:
            print("--- diff ---")
            print(diff)


def _print_history(entries: List[Dict[str, Any]], as_json: bool) -> None:
    if as_json:
        print(json.dumps(entries, ensure_ascii=False, indent=2))
        return
    if not entries:
        print("履歴はありません。")
        return
    for index, entry in enumerate(entries, 1):
        original = str(entry.get("original", "")).replace("\n", " ")
        result = str(entry.get("result", "")).replace("\n", " ")
        if len(original) > 60:
            original = original[:57] + "..."
        if len(result) > 60:
            result = result[:57] + "..."
        print(
            f"[{index}] {entry.get('timestamp', '')} "
            f"{entry.get('mode', '')} "
            f"model={entry.get('model') or '-'}"
        )
        print(f"    {original} -> {result}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m base_pyfile.dev.ai_clipboard",
        description=(
            "クリップボードの内容をAIで処理します。"
            "既定ではコピーを監視し、結果をクリップボードへ書き戻します。"
        ),
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="answer",
        choices=VALID_MODES,
        help="answer（回答）/ correct（校正）/ translate（翻訳）",
    )
    parser.add_argument(
        "language",
        nargs="?",
        default=None,
        help="translate の翻訳先（例: en）。省略時は設定ファイルの既定値。",
    )
    parser.add_argument(
        "--profile",
        choices=("cpu", "5060ti"),
        default=None,
        help="使用するモデル構成（cpu=既定 / 5060ti=RTX 5060 Ti向け）",
    )
    parser.add_argument(
        "--hotkey",
        nargs="?",
        const=DEFAULT_HOTKEY,
        default=None,
        metavar="KEYS",
        help=f"クリップボード監視の代わりにグローバルホットキーで処理する（既定: {DEFAULT_HOTKEY}）",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="結果を書き戻さず、変更前後とdiffを表示する",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="監視せず1回だけ処理する（--hotkey と併用時は1回で終了）",
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--undo", action="store_true", help="直前の処理前の内容をクリップボードへ戻す"
    )
    actions.add_argument(
        "--history", action="store_true", help="最近の処理履歴を表示する"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="--history で表示する件数（既定: 10）"
    )
    parser.add_argument("--json", action="store_true", help="JSONで出力する")
    parser.add_argument(
        "--auto-paste",
        action="store_true",
        help="結果をクリップボードへ書き戻した後に Ctrl+V を送る（pyautoguiが必要）",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="クリップボードを確認する間隔（秒、既定: 0.5）",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="指定回数処理したら終了する",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="処理前後のテキストを履歴へ保存しない",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help="履歴ファイルのパス（省略時はOS既定の場所）",
    )
    return parser


def _intro(mode: str, language: Optional[str]) -> None:
    if mode == "correct":
        print("=== クリップボード 誤字脱字 校正ツール ===")
        print(
            "起動後、何かをコピーすると、校正した内容をクリップボードへ書き戻します。"
        )
        print("そのまま Ctrl+V で、直ったテキストが貼り付けられます。停止は Ctrl+C。")
    elif mode == "translate":
        display_language = language or "設定ファイルの既定値"
        print(f"=== クリップボード 翻訳ツール（{display_language} へ） ===")
        print(
            "起動後、何かをコピーすると、翻訳した内容をクリップボードへ書き戻します。"
        )
        print("そのまま Ctrl+V で、翻訳結果が貼り付けられます。停止は Ctrl+C。")
    else:
        print("=== クリップボードAI 振り分けツール ===")
        print(
            "起動後、何かをコピーすると、内容を判断して回答をクリップボードへ書き戻します。"
        )
        print("そのまま Ctrl+V で貼り付けられます。停止は Ctrl+C。")
    print("安全機能: --preview（差分確認）、--undo（直前を戻す）、--history（履歴）。")
    print(f"ホットキーで1回だけ処理する場合: --hotkey（既定 {DEFAULT_HOTKEY}）")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLIのエントリポイント。終了コードを返します。"""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit は 1 以上を指定してください。")
    if args.interval <= 0:
        parser.error("--interval は 0 より大きい値を指定してください。")
    if args.stop_after is not None and args.stop_after < 1:
        parser.error("--stop-after は 1 以上を指定してください。")
    if args.language is not None and args.mode != "translate":
        parser.error("language は mode=translate のときだけ指定できます。")

    if args.profile:
        os.environ[MODEL_PROFILE_ENV] = args.profile

    history_path = args.history_path

    if args.undo:
        original = undo_last(history_path=history_path)
        if original is None:
            print(
                "戻せませんでした（履歴がないか、クリップボードへ書き込めませんでした）。",
                file=sys.stderr,
            )
            return 1
        if args.json:
            print(json.dumps({"original": original}, ensure_ascii=False, indent=2))
        else:
            print("クリップボードを直前の処理前の内容へ戻しました。")
            print(original)
        return 0

    if args.history:
        entries = load_history(limit=args.limit, history_path=history_path)
        _print_history(entries, args.json)
        return 0

    kwargs: Dict[str, Any] = {}
    if args.mode == "translate":
        kwargs["target_language"] = args.language

    common: Dict[str, Any] = {
        "mode": args.mode,
        "auto_paste": args.auto_paste,
        "copy_result": not args.preview,
        "save_history": not args.no_history,
        "history_path": history_path,
    }

    if args.hotkey is not None:
        stop_after = 1 if args.once else args.stop_after

        def on_result(text: str, result: Dict[str, Any]) -> None:
            if args.preview:
                _print_result(text, result, args.mode, args.json)
            elif args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif _replacement_of(result, args.mode):
                print(
                    f"ホットキーで処理しました（model={result.get('model') or '-'}）。"
                )
            else:
                print("結果を生成できませんでした。", file=sys.stderr)

        try:
            processed = watch_hotkey(
                hotkey=args.hotkey,
                stop_after=stop_after,
                on_result=on_result,
                **common,
                **kwargs,
            )
        except (RuntimeError, ValueError, OSError) as e:
            print(f"ホットキーを開始できませんでした: {e}", file=sys.stderr)
            return 1
        print(f"ホットキー処理を終了しました（{processed}回）。")
        return 0

    if args.preview or args.once:
        source_text = _safe_paste()
        result = process_clipboard_once(source_text=source_text, **common, **kwargs)
        if result is None:
            print("クリップボードが空です。", file=sys.stderr)
            return 1
        if args.preview:
            _print_result(source_text, result, args.mode, args.json)
        elif args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif not _replacement_of(result, args.mode):
            print("結果を生成できませんでした。", file=sys.stderr)
            return 1
        else:
            print(
                f"クリップボードを更新しました（model={result.get('model') or '-'}）。"
            )
        return 0

    _intro(args.mode, args.language)
    try:
        watch_clipboard(
            interval=args.interval,
            stop_after=args.stop_after,
            **common,
            **kwargs,
        )
    except ValueError as e:
        print(f"監視を開始できませんでした: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    logger = make_logger(handler=get_log_handler(10))
    raise SystemExit(main())
