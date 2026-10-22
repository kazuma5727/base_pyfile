"""AIクリップボード安全機能（プレビュー・履歴・アンドゥ・ホットキー）のテスト。

Ollama・pyperclip・pynput・pyautogui の実物は使わない。
"""

import sys
import types

import pytest

from base_pyfile import dev
from base_pyfile.dev import ai_clipboard


def test_safety_helpers_are_public_dev_names():
    expected = {"format_diff", "load_history", "undo_last", "watch_hotkey"}
    assert expected <= set(dev.__all__)
    assert expected <= set(dev._MODULE_OF)


def test_process_clipboard_once_saves_history_and_undo(monkeypatch, tmp_path):
    history_path = tmp_path / "history.jsonl"
    copied = []
    monkeypatch.setattr(ai_clipboard, "_safe_paste", lambda: "before")
    monkeypatch.setattr(
        ai_clipboard, "_safe_copy", lambda text: copied.append(text) or True
    )
    monkeypatch.setattr(
        ai_clipboard,
        "process_text",
        lambda text, **kwargs: {"corrected": "after", "model": "test-model"},
    )

    result = ai_clipboard.process_clipboard_once(
        mode="correct", history_path=history_path
    )

    assert result == {"corrected": "after", "model": "test-model"}
    assert copied == ["after"]
    entries = ai_clipboard.load_history(history_path=history_path)
    assert len(entries) == 1
    assert entries[0]["original"] == "before"
    assert entries[0]["result"] == "after"
    assert entries[0]["mode"] == "correct"
    assert entries[0]["model"] == "test-model"

    assert ai_clipboard.undo_last(history_path=history_path) == "before"
    assert copied[-1] == "before"
    assert ai_clipboard.load_history(history_path=history_path) == []


def test_preview_does_not_copy_or_save_history(monkeypatch, tmp_path):
    history_path = tmp_path / "history.jsonl"
    copied = []
    monkeypatch.setattr(ai_clipboard, "_safe_paste", lambda: "before")
    monkeypatch.setattr(
        ai_clipboard, "_safe_copy", lambda text: copied.append(text) or True
    )
    monkeypatch.setattr(
        ai_clipboard,
        "process_text",
        lambda text, **kwargs: {"corrected": "after", "model": "test-model"},
    )

    result = ai_clipboard.process_clipboard_once(
        mode="correct", copy_result=False, history_path=history_path
    )

    assert result["corrected"] == "after"
    assert copied == []
    assert ai_clipboard.load_history(history_path=history_path) == []


def test_history_limit_keeps_newest_entries(monkeypatch, tmp_path):
    history_path = tmp_path / "history.jsonl"
    monkeypatch.setattr(ai_clipboard, "HISTORY_LIMIT", 2)

    for text in ("a", "b", "c"):
        ai_clipboard._record_history(
            text, text.upper(), "correct", "m", history_path=history_path
        )

    entries = ai_clipboard.load_history(limit=10, history_path=history_path)
    assert [entry["original"] for entry in entries] == ["b", "c"]


def test_format_diff():
    diff = ai_clipboard.format_diff("hello\nworld", "hello\nthere")
    assert "-world" in diff
    assert "+there" in diff


def test_watch_hotkey_processes_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ai_clipboard,
        "process_clipboard_once",
        lambda **kwargs: calls.append(kwargs) or {"corrected": "x"},
    )

    class FakeHotKeys:
        def __init__(self, mapping):
            self.mapping = mapping
            self.running = False

        def start(self):
            self.running = True
            next(iter(self.mapping.values()))()

        def stop(self):
            self.running = False

    fake_keyboard = types.SimpleNamespace(GlobalHotKeys=FakeHotKeys)
    fake_pynput = types.ModuleType("pynput")
    fake_pynput.keyboard = fake_keyboard
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)

    processed = ai_clipboard.watch_hotkey(stop_after=1)

    assert processed == 1
    assert len(calls) == 1
    assert calls[0]["mode"] == "answer"


def test_watch_hotkey_preview_callback(monkeypatch):
    monkeypatch.setattr(ai_clipboard, "_safe_paste", lambda: "before")
    monkeypatch.setattr(
        ai_clipboard,
        "process_clipboard_once",
        lambda **kwargs: {"corrected": "after"},
    )
    results = []

    class FakeHotKeys:
        def __init__(self, mapping):
            self.mapping = mapping
            self.running = False

        def start(self):
            self.running = True
            next(iter(self.mapping.values()))()

        def stop(self):
            self.running = False

    fake_keyboard = types.SimpleNamespace(GlobalHotKeys=FakeHotKeys)
    fake_pynput = types.ModuleType("pynput")
    fake_pynput.keyboard = fake_keyboard
    monkeypatch.setitem(sys.modules, "pynput", fake_pynput)

    processed = ai_clipboard.watch_hotkey(
        stop_after=1,
        copy_result=False,
        on_result=lambda text, result: results.append((text, result)),
    )

    assert processed == 1
    assert results == [("before", {"corrected": "after"})]


def test_watch_hotkey_requires_pynput(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynput", None)
    with pytest.raises(RuntimeError, match="pynput"):
        ai_clipboard.watch_hotkey(stop_after=1)


def test_main_preview_prints_diff_without_copy(monkeypatch, tmp_path, capsys):
    history_path = tmp_path / "history.jsonl"
    copied = []
    monkeypatch.setattr(ai_clipboard, "_safe_paste", lambda: "before")
    monkeypatch.setattr(
        ai_clipboard, "_safe_copy", lambda text: copied.append(text) or True
    )
    monkeypatch.setattr(
        ai_clipboard,
        "process_text",
        lambda text, **kwargs: {"corrected": "after"},
    )

    exit_code = ai_clipboard.main(
        ["correct", "--preview", "--history-path", str(history_path)]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert copied == []
    assert "変更前" in output
    assert "変更後" in output
    assert ai_clipboard.load_history(history_path=history_path) == []


def test_main_undo(monkeypatch, tmp_path, capsys):
    history_path = tmp_path / "history.jsonl"
    ai_clipboard._record_history(
        "before", "after", "correct", "m", history_path=history_path
    )
    copied = []
    monkeypatch.setattr(
        ai_clipboard, "_safe_copy", lambda text: copied.append(text) or True
    )

    exit_code = ai_clipboard.main(
        ["--undo", "--history-path", str(history_path)]
    )

    assert exit_code == 0
    assert copied == ["before"]
    assert "戻しました" in capsys.readouterr().out
    assert ai_clipboard.load_history(history_path=history_path) == []
