"""Tests for the TTY-free parts of the UI: Choice coercion and the non-TTY guard."""
from __future__ import annotations

import sys

from kavier_ui import app, prompts
from kavier_ui.prompts import Choice


def test_choice_coerce_mixes_strings_and_choices() -> None:
    out = prompts._coerce(["a", Choice("b", "B", "hint")])
    assert [c.value for c in out] == ["a", "b"]
    assert out[0].label == "a" and out[1].label == "B" and out[1].hint == "hint"


def test_main_exits_cleanly_without_tty(monkeypatch, capsys) -> None:
    # stdin is not a TTY under pytest; main() must warn and return (not crash on raw-tty).
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    app.main()
    assert "interactive terminal" in capsys.readouterr().out.lower()
