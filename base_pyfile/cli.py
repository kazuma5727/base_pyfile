"""base_pyfile の統合CLI。

``python -m base_pyfile`` または ``base-pyfile`` で呼び出す。
重い依存は各サブコマンドの実行時だけ import する。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence, TextIO

from base_pyfile._version import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="base-pyfile",
        description="base_pyfile の便利ツールをまとめて呼び出すCLI。",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="実行環境・任意依存・利用可能な機能を診断する"
    )
    doctor_parser.add_argument("--json", action="store_true", help="JSONで出力する")

    run_parser = subparsers.add_parser("run", help="外部コマンドを実行する")
    run_parser.add_argument("--timeout", type=float, default=None, help="タイムアウト秒")
    run_parser.add_argument("--cwd", type=Path, default=None, help="作業ディレクトリ")
    run_parser.add_argument("--encoding", default=None, help="入出力の文字コード")
    run_parser.add_argument(
        "--shell", action="store_true", help="シェル経由で実行する（注入に注意）"
    )
    run_parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="実行するコマンド（例: base-pyfile run -- git status）",
    )

    open_parser = subparsers.add_parser(
        "open", help="ファイルやフォルダを既定アプリで開く"
    )
    open_parser.add_argument("path", type=Path)

    pdf_parser = subparsers.add_parser("pdf-to-png", help="PDFをPNGに変換する")
    pdf_parser.add_argument("input", type=Path)
    pdf_parser.add_argument("-o", "--output", type=Path, default=None, help="出力フォルダ")
    pdf_parser.add_argument(
        "--dpi", type=int, default=None, help="解像度（例: 200。既定はPyMuPDFの既定値）"
    )

    tiff_parser = subparsers.add_parser(
        "tiff-to-pdf", help="TIFFをPDFに変換する"
    )
    tiff_parser.add_argument("input", type=Path)
    tiff_parser.add_argument("-o", "--output", type=Path, default=None, help="出力PDF")

    image_parser = subparsers.add_parser(
        "image-to-pdf", help="複数の画像を1つのPDFに変換する"
    )
    image_parser.add_argument("images", nargs="+", type=Path)
    image_parser.add_argument(
        "-o", "--output", type=Path, required=True, help="出力PDF"
    )

    ai_parser = subparsers.add_parser(
        "ai",
        help="開発中AI（editableインストール・リポジトリ直下でのみ利用可能）",
        add_help=False,
    )
    ai_parser.add_argument("args", nargs=argparse.REMAINDER)

    subparsers.add_parser("version", help="バージョンを表示する")
    return parser


def _write(stream: TextIO, text: str) -> None:
    """コンソールの文字コードで表現できない文字を置換して書き込む。"""
    if not text:
        return
    try:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
    except UnicodeEncodeError:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        stream.write(safe_text)
        if not safe_text.endswith("\n"):
            stream.write("\n")
    stream.flush()


def _cmd_doctor(args: argparse.Namespace) -> int:
    from base_pyfile.diagnostics import doctor

    report = doctor()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    python_info = report["python"]
    platform_info = report["platform"]
    bitness = "64bit" if python_info["64bit"] else "32bit"
    print(f"base_pyfile {report['base_pyfile']}")
    print(f"Package {report['package_path']}")
    print(f"Python {python_info['version']} ({bitness})")
    print(
        "Platform "
        f"{platform_info['system']} {platform_info['release']} "
        f"({platform_info['machine']})"
    )
    print("Features:")
    for name, info in report["features"].items():
        if info["available"]:
            print(f"  {name}: OK")
        else:
            print(f"  {name}: missing {', '.join(info['missing'])}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    # argparse の REMAINDER は先頭の "--" を含むことがある。
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print(
            "コマンドを指定してください。例: base-pyfile run -- git status",
            file=sys.stderr,
        )
        return 2

    from base_pyfile.process_manager import run_command

    try:
        result = run_command(
            command,
            timeout=args.timeout,
            cwd=args.cwd,
            encoding=args.encoding,
            shell=args.shell,
        )
    except (OSError, ValueError, LookupError) as exc:
        print(f"実行できませんでした: {exc}", file=sys.stderr)
        return 1

    _write(sys.stdout, result.stdout)
    _write(sys.stderr, result.stderr)
    if result.timed_out:
        print(f"タイムアウトしました（{args.timeout}秒）。", file=sys.stderr)
        return 124
    return result.returncode if result.returncode >= 0 else 1


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=True)
    else:
        subprocess.run(["xdg-open", str(path)], check=True)


def _cmd_open(args: argparse.Namespace) -> int:
    try:
        _open_path(args.path)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"開けませんでした: {exc}", file=sys.stderr)
        return 1
    return 0


def _pdf_to_png(input_path: Path, output: Path | None, dpi: int | None) -> None:
    from base_pyfile.pdf_tiff_converter import pdf_to_png

    pdf_to_png(input_path, output_folder=output, dpi=dpi)


def _tiff_to_pdf(input_path: Path, output: Path | None) -> None:
    from base_pyfile.pdf_tiff_converter import tiff_to_pdf

    tiff_to_pdf(input_path, output_pdf=output)


def _image_to_pdf(images: Sequence[Path], output: Path) -> None:
    from base_pyfile.pdf_tiff_converter import image_to_pdf

    image_to_pdf(list(images), output)


def _handle_pdf_error(exc: Exception) -> int:
    if isinstance(exc, ImportError):
        print(
            'PDF機能には PyMuPDF と Pillow が必要です: pip install "base_pyfile[pdf]"',
            file=sys.stderr,
        )
    else:
        print(f"変換に失敗しました: {exc}", file=sys.stderr)
    return 1


def _cmd_pdf_to_png(args: argparse.Namespace) -> int:
    try:
        _pdf_to_png(args.input, args.output, args.dpi)
    except Exception as exc:
        return _handle_pdf_error(exc)
    output_folder = args.output or args.input.parent / args.input.stem
    print(f"PNGを出力しました: {output_folder}")
    return 0


def _cmd_tiff_to_pdf(args: argparse.Namespace) -> int:
    try:
        _tiff_to_pdf(args.input, args.output)
    except Exception as exc:
        return _handle_pdf_error(exc)
    output_pdf = args.output or args.input.with_suffix(".pdf")
    print(f"PDFを出力しました: {output_pdf}")
    return 0


def _cmd_image_to_pdf(args: argparse.Namespace) -> int:
    try:
        _image_to_pdf(args.images, args.output)
    except Exception as exc:
        return _handle_pdf_error(exc)
    print(f"PDFを出力しました: {args.output}")
    return 0


def _cmd_ai(args: argparse.Namespace) -> int:
    try:
        from base_pyfile.dev.ai_clipboard import main as ai_main
    except ImportError:
        print(
            "AI機能は開発中のため、リポジトリ直下または editable インストールでのみ使えます。",
            file=sys.stderr,
        )
        return 1

    ai_args = list(args.args)
    if not ai_args:
        ai_args = ["--help"]
    return ai_main(ai_args)


def _cmd_version(args: argparse.Namespace) -> int:
    print(__version__)
    return 0


_HANDLERS = {
    "doctor": _cmd_doctor,
    "run": _cmd_run,
    "open": _cmd_open,
    "pdf-to-png": _cmd_pdf_to_png,
    "tiff-to-pdf": _cmd_tiff_to_pdf,
    "image-to-pdf": _cmd_image_to_pdf,
    "ai": _cmd_ai,
    "version": _cmd_version,
}


def main(argv: Sequence[str] | None = None) -> int:
    """CLIのエントリポイント。終了コードを返す。"""
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    # ``ai`` は引数全体を開発中AIのCLIへそのまま渡す。
    # 先に外側の argparse へ通すと ``--help`` などが消費されてしまうため、
    # ここで分岐する。
    if actual_argv[:1] == ["ai"]:
        return _cmd_ai(argparse.Namespace(args=actual_argv[1:]))

    parser = _build_parser()
    args = parser.parse_args(actual_argv)
    handler = _HANDLERS[args.subcommand]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
