"""統合CLI（``base-pyfile`` / ``python -m base_pyfile``）の回帰テスト。

外部サービスや重い依存（PyMuPDF / OpenCV / pyautogui）は使わない。
"""

import json
import os
import subprocess
import sys

import pytest

import base_pyfile
from base_pyfile import cli
from base_pyfile.dev.ai_router import MODEL_PROFILE_ENV


def test_version_subcommand(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == base_pyfile.__version__


def test_doctor_json(capsys):
    assert cli.main(["doctor", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["base_pyfile"] == base_pyfile.__version__
    assert "features" in report
    assert report["features"]["core"]["available"] is True


def test_run_success(capsys):
    assert cli.main(["run", sys.executable, "-c", "print('hello')"]) == 0
    assert "hello" in capsys.readouterr().out


def test_run_with_separator(capsys):
    assert (
        cli.main(
            ["run", "--timeout", "30", "--", sys.executable, "-c", "print('separated')"]
        )
        == 0
    )
    assert "separated" in capsys.readouterr().out


def test_run_propagates_exit_code():
    assert cli.main(["run", sys.executable, "-c", "import sys; sys.exit(3)"]) == 3


def test_run_without_command(capsys):
    assert cli.main(["run"]) == 2
    assert "コマンドを指定" in capsys.readouterr().err


def test_open_missing_path(capsys, tmp_path):
    assert cli.main(["open", str(tmp_path / "missing.txt")]) == 1
    assert "開けませんでした" in capsys.readouterr().err


def test_open_dispatches_path(monkeypatch, tmp_path):
    opened = []
    monkeypatch.setattr(cli, "_open_path", lambda path: opened.append(path))
    assert cli.main(["open", str(tmp_path)]) == 0
    assert opened == [tmp_path]


def test_pdf_to_png_dispatches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli,
        "_pdf_to_png",
        lambda input_path, output, dpi: calls.append((input_path, output, dpi)),
    )
    input_path = tmp_path / "input.pdf"
    output = tmp_path / "out"
    assert (
        cli.main(["pdf-to-png", str(input_path), "-o", str(output), "--dpi", "200"])
        == 0
    )
    assert calls == [(input_path, output, 200)]


def test_tiff_to_pdf_dispatches(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        cli,
        "_tiff_to_pdf",
        lambda input_path, output: calls.append((input_path, output)),
    )
    input_path = tmp_path / "input.tiff"
    assert cli.main(["tiff-to-pdf", str(input_path)]) == 0
    assert calls == [(input_path, None)]


def test_ai_subcommand_history(tmp_path, capsys):
    history_path = tmp_path / "history.jsonl"
    assert cli.main(["ai", "--history", "--history-path", str(history_path)]) == 0
    assert "履歴はありません" in capsys.readouterr().out


def test_ai_subcommand_profile_passthrough(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_PROFILE_ENV, raising=False)
    assert (
        cli.main(
            [
                "ai",
                "--history",
                "--profile",
                "5060ti",
                "--history-path",
                str(tmp_path / "history.jsonl"),
            ]
        )
        == 0
    )
    assert os.environ[MODEL_PROFILE_ENV] == "5060ti"


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


def test_python_m_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "base_pyfile", "version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert base_pyfile.__version__ in result.stdout
