"""実行環境と任意依存の診断情報を返すモジュール。

このモジュールは標準ライブラリだけで動く。``cv2`` や ``fitz`` などの
任意依存は ``importlib.util.find_spec`` で存在だけを確認し、実際には
import しない。そのため、``from base_pyfile import doctor`` を呼んでも
重い依存は読み込まれない。

CLI としても使える::

    python -m base_pyfile.diagnostics
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import locale
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from base_pyfile._version import __version__

# import 名 -> 配布名の候補。cv2 のように import 名と配布名が違うものや、
# headless 版でも同じ import 名になるものを吸収する。
_DEPENDENCIES: Dict[str, Tuple[str, ...]] = {
    "natsort": ("natsort",),
    "requests": ("requests",),
    "pyperclip": ("pyperclip",),
    "numpy": ("numpy",),
    "cv2": (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python",
        "opencv-contrib-python-headless",
    ),
    "pyautogui": ("pyautogui",),
    "pynput": ("pynput",),
    "bs4": ("beautifulsoup4",),
    "tqdm": ("tqdm",),
    "fitz": ("PyMuPDF",),
    "PIL": ("Pillow",),
}

# 機能名 -> 必要な import 名。core は常に利用可能。
_FEATURES: Dict[str, Tuple[str, ...]] = {
    "core": (),
    "automation": ("numpy", "cv2", "pyautogui", "pynput"),
    "web": (
        "requests",
        "bs4",
        "tqdm",
        "pyperclip",
        "numpy",
        "cv2",
        "pyautogui",
        "pynput",
    ),
    "pdf": ("fitz", "PIL"),
    "ai_dev": ("requests",),
}


def _module_available(module_name: str) -> bool:
    """モジュールが import 可能かを、実際に import せずに調べる。"""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, AttributeError, ValueError):
        # 親パッケージが壊れている場合など。診断用途なので False に倒す。
        return False


def _distribution_version(module_name: str) -> str | None:
    """import 名から配布パッケージのバージョンを引く。見つからなければ None。"""
    for distribution_name in _DEPENDENCIES.get(module_name, ()):
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _dependency_report() -> Dict[str, Dict[str, Any]]:
    report: Dict[str, Dict[str, Any]] = {}
    for module_name in _DEPENDENCIES:
        available = _module_available(module_name)
        report[module_name] = {
            "available": available,
            "version": _distribution_version(module_name) if available else None,
        }
    return report


def _feature_report(
    dependencies: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    features: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []

    for feature_name, required_modules in _FEATURES.items():
        missing = [
            module_name
            for module_name in required_modules
            if not dependencies[module_name]["available"]
        ]

        # 開発中AIは安定版 wheel に含まれないため、モジュール自体の有無も見る。
        if feature_name == "ai_dev" and not _module_available("base_pyfile.dev"):
            missing.append("base_pyfile.dev")

        features[feature_name] = {
            "available": not missing,
            "missing": missing,
        }

        # ai_dev は安定版では意図的に無効なので警告にはしない。
        if missing and feature_name != "ai_dev":
            warnings.append(
                f"{feature_name}: 利用できません（不足: {', '.join(missing)}）"
            )

    return features, warnings


def _stream_encoding(stream: Any) -> str | None:
    return getattr(stream, "encoding", None) if stream is not None else None


def doctor() -> Dict[str, Any]:
    """実行環境、任意依存、利用可能な機能をまとめて返す。

    Returns:
        dict: 以下のキーを持つ診断情報。

        - ``base_pyfile``: パッケージのバージョン
        - ``python``: Python のバージョン・実行ファイル・64bit かどうか
        - ``platform``: OS 名・リリース・マシン種別
        - ``encoding``: 既定・推奨・標準出力・ファイルシステムの文字コード
        - ``dependencies``: 各任意依存の導入状況とバージョン
        - ``features``: ``core`` / ``automation`` / ``web`` / ``pdf`` / ``ai_dev``
          が利用可能かと、不足しているモジュール
        - ``warnings``: 不足している任意機能の説明

    この関数は副作用としてファイルやクリップボードを変更しない。
    """
    dependencies = _dependency_report()
    features, warnings = _feature_report(dependencies)
    package_dir = Path(__file__).resolve().parent

    return {
        "base_pyfile": __version__,
        "package_path": str(package_dir),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "64bit": sys.maxsize > 2**32,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "windows": os.name == "nt",
        },
        "cwd": str(Path.cwd()),
        "encoding": {
            "default": sys.getdefaultencoding(),
            "preferred": locale.getpreferredencoding(False),
            "stdout": _stream_encoding(sys.stdout),
            "stderr": _stream_encoding(sys.stderr),
            "filesystem": sys.getfilesystemencoding(),
        },
        "dependencies": dependencies,
        "features": features,
        "warnings": warnings,
    }


def main() -> None:
    """``python -m base_pyfile.diagnostics`` のエントリポイント。"""
    import json

    print(json.dumps(doctor(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
