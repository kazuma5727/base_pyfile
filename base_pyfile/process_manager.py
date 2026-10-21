"""外部コマンドを安全に実行するためのユーティリティ。

Windows の日本語環境では、子プロセスの標準出力が UTF-8 だったり
CP932 だったりする。``subprocess.run(..., text=True)`` だけで済ませると
文字化けや ``UnicodeDecodeError`` になりやすいため、このモジュールでは
バイト列で受け取ってから自動判別する。
"""

from __future__ import annotations

import locale
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

# subprocess.run に渡す引数の型（PathLike を含む）
CommandArgs = str | Sequence[str | os.PathLike[str]]


@dataclass(frozen=True)
class CommandResult:
    """``run_command`` の実行結果。"""

    args: str | Tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """正常終了（returncode == 0 かつタイムアウトしていない）か。"""
        return self.returncode == 0 and not self.timed_out


def _normalize_args(args: CommandArgs) -> str | Tuple[str, ...]:
    """引数を subprocess と結果表示の両方で扱いやすい形に揃える。"""
    if isinstance(args, str):
        return args
    return tuple(os.fsdecode(os.fspath(arg)) for arg in args)


def _decode_output(
    data: bytes | str | None,
    encoding: str | None,
    errors: str,
) -> str:
    """子プロセスの出力をデコードする。

    ``encoding`` が指定された場合はそれを使う。指定がない場合は
    UTF-8 (BOM 付きを含む) -> CP932 (Windows) -> OS の推奨エンコーディング
    の順に試す。
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if encoding:
        return data.decode(encoding, errors=errors)

    # UTF-16 は BOM がある場合だけ試す。BOM なしで decode すると
    # CP932 のバイト列を誤って UTF-16 として読めてしまうため。
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass

    preferred_encoding = locale.getpreferredencoding(False)
    candidates = ["utf-8-sig", "utf-8"]
    if os.name == "nt":
        # 日本語Windowsのcp932を、英語Windowsのcp1252より先に試す。
        # cp1252 は多くのバイト列をデコードできてしまい、cp932 の出力が
        # 文字化けしたまま採用されるのを防ぐ。
        candidates.append("cp932")
    candidates.append(preferred_encoding)
    if os.name != "nt":
        candidates.append("cp932")
    seen = set()
    for candidate in candidates:
        if not candidate or candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        try:
            return data.decode(candidate)
        except (UnicodeDecodeError, LookupError):
            continue

    # 最後の手段。診断用途なので、例外で落とさず置換文字を返す。
    return data.decode("utf-8", errors="replace")


def _encode_input(
    input_data: str | bytes | None,
    encoding: str | None,
) -> bytes | None:
    if input_data is None or isinstance(input_data, bytes):
        return input_data
    if not isinstance(input_data, str):
        raise TypeError("input は str または bytes、あるいは None です。")
    # 子プロセスの標準入力は、通常は OS の推奨エンコーディングで読まれる。
    input_encoding = encoding or locale.getpreferredencoding(False)
    return input_data.encode(input_encoding)


def run_command(
    args: CommandArgs,
    *,
    timeout: float | None = None,
    cwd: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
    input: str | bytes | None = None,
    encoding: str | None = None,
    errors: str = "replace",
    check: bool = False,
    shell: bool = False,
    hide_window: bool = True,
) -> CommandResult:
    """外部コマンドを実行し、標準出力・標準エラーを文字列で返す。

    Args:
        args: コマンドと引数。``shell=False`` ならリスト/タプルを推奨。
        timeout: タイムアウト秒。``None`` なら無制限。
        cwd: 子プロセスの作業ディレクトリ。
        env: 子プロセスの環境変数。``None`` なら現在の環境を引き継ぐ。
        input: 標準入力に渡す文字列またはバイト列。
        encoding: 入出力の文字コード。``None`` なら出力を自動判別する。
        errors: ``encoding`` 指定時のデコードエラー処理。既定は ``"replace"``。
        check: True なら終了コードが 0 以外、またはタイムアウト時に例外を送出する。
        shell: True ならシェル経由で実行する。シェル注入に注意。
        hide_window: Windows でコンソールウィンドウを出さない。既定は True。

    Returns:
        CommandResult: 終了コード、標準出力、標準エラー、実行時間を含む結果。
        タイムアウト時は ``timed_out=True``、``returncode=-1`` になる
        （``check=True`` の場合は ``subprocess.TimeoutExpired`` を送出）。

    Raises:
        subprocess.CalledProcessError: ``check=True`` で終了コードが 0 以外。
        subprocess.TimeoutExpired: ``check=True`` でタイムアウト。
        OSError: 実行ファイルが見つからない、権限がないなど。
    """
    normalized_args = _normalize_args(args)
    input_bytes = _encode_input(input, encoding)

    kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": timeout,
        "shell": shell,
        "check": False,
    }
    if cwd is not None:
        kwargs["cwd"] = os.fspath(cwd)
    if env is not None:
        kwargs["env"] = dict(env)
    if input_bytes is None:
        # 対話入力を待って固まらないよう、既定では標準入力を閉じる。
        kwargs["stdin"] = subprocess.DEVNULL
    else:
        kwargs["input"] = input_bytes
    if hide_window and os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    started_at = time.perf_counter()
    try:
        completed = subprocess.run(normalized_args, **kwargs)
    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - started_at
        stdout = _decode_output(exc.stdout, encoding, errors)
        stderr = _decode_output(exc.stderr, encoding, errors)
        if check:
            raise subprocess.TimeoutExpired(
                normalized_args,
                timeout,
                output=stdout,
                stderr=stderr,
            ) from exc
        return CommandResult(
            args=normalized_args,
            returncode=-1,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            timed_out=True,
        )

    duration = time.perf_counter() - started_at
    stdout = _decode_output(completed.stdout, encoding, errors)
    stderr = _decode_output(completed.stderr, encoding, errors)

    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            normalized_args,
            output=stdout,
            stderr=stderr,
        )

    return CommandResult(
        args=normalized_args,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        duration=duration,
    )


__all__ = ["CommandResult", "run_command"]
