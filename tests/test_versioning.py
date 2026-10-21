"""バージョン方針と changelog の運用を確認するテスト。

バージョンだけ上げて CHANGELOG.md の更新を忘れた場合に CI で検出する。
"""

import re
from pathlib import Path

import base_pyfile


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_versioning_docs_exist_and_are_in_sdist():
    assert (ROOT / "VERSIONING.md").is_file()
    assert (ROOT / "CHANGELOG.md").is_file()
    manifest = _read("MANIFEST.in")
    assert "VERSIONING.md" in manifest
    assert "CHANGELOG.md" in manifest
    assert "AGENTS.md" in manifest


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", base_pyfile.__version__)


def test_changelog_has_unreleased_and_current_version():
    changelog = _read("CHANGELOG.md")
    assert "## [Unreleased]" in changelog
    assert f"## [{base_pyfile.__version__}]" in changelog


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
