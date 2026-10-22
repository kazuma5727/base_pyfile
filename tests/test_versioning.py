"""バージョン方針と changelog の運用を確認するテスト。

バージョンだけ上げて CHANGELOG.md の更新を忘れた場合や、
見出しの形式・並び順が崩れた場合に CI で検出する。
"""

import re
from datetime import date
from pathlib import Path

import base_pyfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE_HEADING = re.compile(
    r"^## \[(?P<version>\d+\.\d+\.\d+)\] - (?P<date>\d{4}-\d{2}-\d{2})$",
    re.MULTILINE,
)
CHANGELOG_HEADING = re.compile(r"^## \[[^\]]+\].*$", re.MULTILINE)


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _release_entries(changelog: str) -> list[tuple[str, str]]:
    return [
        (match.group("version"), match.group("date"))
        for match in RELEASE_HEADING.finditer(changelog)
    ]


def _version_key(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def test_versioning_docs_exist_and_are_in_sdist():
    assert (ROOT / "VERSIONING.md").is_file()
    assert (ROOT / "CHANGELOG.md").is_file()
    assert (ROOT / "scripts" / "check_dist.py").is_file()
    manifest = _read("MANIFEST.in")
    for name in ("README.md", "VERSIONING.md", "CHANGELOG.md", "AGENTS.md"):
        assert re.search(rf"^include {re.escape(name)}$", manifest, re.MULTILINE), name


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", base_pyfile.__version__)


def test_changelog_has_unreleased_and_current_version():
    changelog = _read("CHANGELOG.md")
    entries = _release_entries(changelog)
    assert entries
    assert len(re.findall(r"^## \[Unreleased\]$", changelog, re.MULTILINE)) == 1
    assert f"## [{base_pyfile.__version__}]" in changelog
    assert changelog.index("## [Unreleased]") < changelog.index(
        f"## [{entries[0][0]}] - {entries[0][1]}"
    )


def test_changelog_headings_are_well_formed():
    headings = CHANGELOG_HEADING.findall(_read("CHANGELOG.md"))
    assert headings
    for heading in headings:
        assert heading == "## [Unreleased]" or RELEASE_HEADING.fullmatch(heading), heading


def test_changelog_releases_are_unique_and_ordered():
    entries = _release_entries(_read("CHANGELOG.md"))
    assert entries

    versions = [version for version, _ in entries]
    assert len(versions) == len(set(versions)), f"バージョンが重複しています: {versions}"
    version_keys = [_version_key(version) for version in versions]
    assert version_keys == sorted(
        version_keys, reverse=True
    ), f"バージョンの並びが新しい順ではありません: {versions}"

    release_dates = [date.fromisoformat(value) for _, value in entries]
    assert release_dates == sorted(
        release_dates, reverse=True
    ), f"日付の並びが新しい順ではありません: {release_dates}"


def test_current_version_is_the_latest_release():
    entries = _release_entries(_read("CHANGELOG.md"))
    assert entries
    assert entries[0][0] == base_pyfile.__version__


def test_agents_points_to_versioning_policy():
    agents = _read("AGENTS.md")
    assert "VERSIONING.md" in agents
    assert "CHANGELOG.md" in agents
    assert "MAJOR" in agents and "MINOR" in agents and "PATCH" in agents


def test_versioning_policy_has_ai_prompt_and_release_steps():
    policy = _read("VERSIONING.md")
    assert "Semantic Versioning" in policy
    assert "AI・エージェント向け指示" in policy
    assert "外部AIに渡すプロンプト例" in policy
    assert "リリース手順" in policy
    assert "tests/test_versioning.py" in policy
