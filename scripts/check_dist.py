"""ビルド済みの sdist / wheel を検証する。

`python -m build` の後に実行し、成果物の同梱内容とバージョンの一致を確認する。
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_SDIST = [
    "README.md",
    "VERSIONING.md",
    "CHANGELOG.md",
    "AGENTS.md",
    "MANIFEST.in",
    "pyproject.toml",
    "scripts/check_dist.py",
    "tests/test_versioning.py",
    "base_pyfile/settings.ini",
    "base_pyfile/dev/ai_models.json",
]
REQUIRED_WHEEL = [
    "base_pyfile/_version.py",
    "base_pyfile/settings.ini",
]
FORBIDDEN_WHEEL_PREFIXES = ["base_pyfile/dev/"]


def _single(dist: Path, pattern: str) -> Path:
    matches = sorted(dist.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"{pattern} は1つだけ必要です: {matches}")
    return matches[0]


def _version_from_source() -> str:
    text = (ROOT / "base_pyfile" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', text, re.MULTILINE)
    if match is None:
        raise SystemExit("_version.py から __version__ を取得できません")
    return match.group(1)


def _check_sdist(path: Path, version: str) -> None:
    with tarfile.open(path) as archive:
        names = archive.getnames()

        top_level = {name.split("/", 1)[0] for name in names}
        expected_top = f"base_pyfile-{version}"
        if top_level != {expected_top}:
            raise SystemExit(f"sdist のトップレベルが想定外です: {sorted(top_level)}")

        contents = {name.split("/", 1)[1] for name in names if "/" in name}
        missing = [name for name in REQUIRED_SDIST if name not in contents]
        if missing:
            raise SystemExit(f"sdist に不足しているファイルがあります: {missing}")

        pkg_info = archive.extractfile(f"{expected_top}/PKG-INFO")
        if pkg_info is None:
            raise SystemExit("sdist に PKG-INFO がありません")
        metadata = pkg_info.read().decode("utf-8")

    if f"Version: {version}" not in metadata.splitlines():
        raise SystemExit("sdist の PKG-INFO のバージョンが _version.py と一致しません")


def _check_wheel(path: Path, version: str) -> None:
    if not path.name.startswith(f"base_pyfile-{version}-"):
        raise SystemExit(f"wheel のファイル名がバージョンと一致しません: {path.name}")

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")), None
        )
        if metadata_name is None:
            raise SystemExit("wheel に METADATA がありません")
        metadata = archive.read(metadata_name).decode("utf-8")

    missing = [name for name in REQUIRED_WHEEL if name not in names]
    if missing:
        raise SystemExit(f"wheel に不足しているファイルがあります: {missing}")

    leaked = [
        name
        for name in names
        if any(name.startswith(prefix) for prefix in FORBIDDEN_WHEEL_PREFIXES)
    ]
    if leaked:
        raise SystemExit(f"wheel に base_pyfile.dev が混入しています: {leaked}")

    if f"Version: {version}" not in metadata.splitlines():
        raise SystemExit("wheel の METADATA のバージョンが _version.py と一致しません")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dist",
        type=Path,
        default=ROOT / "dist",
        help="ビルド成果物のディレクトリ（既定: dist）",
    )
    args = parser.parse_args()

    dist = args.dist.resolve()
    version = _version_from_source()
    sdist = _single(dist, "*.tar.gz")
    wheel = _single(dist, "*.whl")

    _check_sdist(sdist, version)
    _check_wheel(wheel, version)

    print(f"OK: {sdist.name}, {wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
