"""Tests for the TTY-free logic of ``kavier.ui.prompts`` and the ``app.main`` non-TTY guard.

The interactive widgets (``menu``/``fuzzy_select``/``text_prompt``/``number_prompt``/``confirm``)
are all driven by a single seam — ``prompts.read_key`` — so we script a keypress sequence and assert
the widget's *contract* (which value it returns, when it aborts), never a snapshot of Rich output.
``read_key`` itself is exercised against the ANSI/ASCII terminal standard (ESC[A = cursor-up,
0x03 = Ctrl-C/ETX, 0x0d = CR, 0x7f = DEL) with the OS read seam faked.
"""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Iterator

import pytest
from rich.console import Console

from kavier.ui import app, prompts
from kavier.ui.prompts import Abort, Choice


@pytest.fixture(autouse=True)
def _quiet_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route widget rendering to a throwaway buffer so scripted runs stay silent and deterministic."""
    monkeypatch.setattr(prompts, "console", Console(file=io.StringIO(), width=80))


def _script(monkeypatch: pytest.MonkeyPatch, keys: list[str]) -> None:
    """Make ``read_key`` yield ``keys`` in order (StopIteration if a test under-feeds it)."""
    it: Iterator[str] = iter(keys)
    monkeypatch.setattr(prompts, "read_key", lambda: next(it))


# --------------------------------------------------------------------------- _coerce


def test_coerce_wraps_bare_string_as_value_and_label() -> None:
    # Contract: a str "a" becomes Choice(value="a", label="a", hint="").
    (out,) = prompts._coerce(["a"])
    assert isinstance(out, Choice)
    assert (out.value, out.label, out.hint) == ("a", "a", "")


def test_coerce_passes_choice_through_by_identity() -> None:
    # A Choice must NOT be re-wrapped: same object, fields intact (would break if we rebuilt it).
    original = Choice("b", "B", "hint")
    (out,) = prompts._coerce([original])
    assert out is original


# --------------------------------------------------------------------------- menu


def test_menu_enter_returns_value_at_default_index(monkeypatch: pytest.MonkeyPatch) -> None:
    # Immediate enter with default=1 must return the SECOND item's value, not the first.
    _script(monkeypatch, ["enter"])
    assert prompts.menu("t", [Choice("va", "a"), Choice("vb", "b")], default=1) == "vb"


def test_menu_down_advances_one_row(monkeypatch: pytest.MonkeyPatch) -> None:
    # default=0, one "down" -> index 1 -> value of second item.
    _script(monkeypatch, ["down", "enter"])
    assert prompts.menu("t", ["a", "b", "c"]) == "b"


def test_menu_up_from_top_wraps_to_last(monkeypatch: pytest.MonkeyPatch) -> None:
    # (0 - 1) % 3 == 2 -> the modulo wrap lands on the last item, not a clamp at 0.
    _script(monkeypatch, ["up", "enter"])
    assert prompts.menu("t", ["a", "b", "c"]) == "c"


def test_menu_default_clamped_into_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # default=99 with 3 items -> min(99, len-1) == 2; enter returns the last item (no IndexError).
    _script(monkeypatch, ["enter"])
    assert prompts.menu("t", ["a", "b", "c"], default=99) == "c"


@pytest.mark.parametrize("key", ["q", "esc"])
def test_menu_quit_keys_abort(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    _script(monkeypatch, [key])
    with pytest.raises(Abort):
        prompts.menu("t", ["a", "b"])


# --------------------------------------------------------------------------- fuzzy_select


def test_fuzzy_preselects_row_matching_default_value(monkeypatch: pytest.MonkeyPatch) -> None:
    # default="vy" -> cursor starts on that row, so a bare enter returns it (not the first item).
    _script(monkeypatch, ["enter"])
    got = prompts.fuzzy_select("t", [Choice("vx", "x"), Choice("vy", "y")], default="vy")
    assert got == "vy"


def test_fuzzy_substring_filter_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    # Typing uppercase "B" filters to the single label containing "b" -> enter returns banana.
    # A case-SENSITIVE match would leave zero rows and enter would fall through to StopIteration.
    _script(monkeypatch, ["B", "enter"])
    got = prompts.fuzzy_select("t", [Choice("va", "apple"), Choice("vb", "banana")])
    assert got == "vb"


def test_fuzzy_backspace_restores_full_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # "z" filters to nothing; backspace clears the query back to the full list; enter -> first item.
    # If backspace were a no-op the list stays empty, enter is ignored, and read_key runs dry.
    _script(monkeypatch, ["z", "backspace", "enter"])
    got = prompts.fuzzy_select("t", [Choice("va", "apple"), Choice("vb", "banana")])
    assert got == "va"


def test_fuzzy_esc_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, ["esc"])
    with pytest.raises(Abort):
        prompts.fuzzy_select("t", ["a", "b"])


# --------------------------------------------------------------------------- text_prompt


def test_text_prompt_enter_accepts_editable_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, ["enter"])
    assert prompts.text_prompt("m", default="foo") == "foo"


def test_text_prompt_typing_then_backspace(monkeypatch: pytest.MonkeyPatch) -> None:
    # append h,i,x -> "hix"; one backspace -> "hi".
    _script(monkeypatch, ["h", "i", "x", "backspace", "enter"])
    assert prompts.text_prompt("m", default="") == "hi"


def test_text_prompt_esc_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, ["esc"])
    with pytest.raises(Abort):
        prompts.text_prompt("m")


# --------------------------------------------------------------------------- number_prompt


def _feed_text(monkeypatch: pytest.MonkeyPatch, raws: list[str]) -> None:
    """Drive number_prompt by scripting its inner text_prompt returns (one per re-prompt)."""
    it: Iterator[str] = iter(raws)
    monkeypatch.setattr(prompts, "text_prompt", lambda *a, **k: next(it))


def test_number_prompt_empty_returns_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cleared input ("") short-circuits to the default value unchanged.
    _feed_text(monkeypatch, [""])
    assert prompts.number_prompt("m", default=128, minimum=1) == 128


def test_number_prompt_parses_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    # default is int + integer=True -> "256" parses via int(), yielding int 256 (not float 256.0).
    _feed_text(monkeypatch, ["256"])
    got = prompts.number_prompt("m", default=1, minimum=1)
    assert got == 256 and isinstance(got, int)


def test_number_prompt_parses_float_when_integer_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _feed_text(monkeypatch, ["2.5"])
    got = prompts.number_prompt("m", default=1.0, integer=False)
    assert got == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("bad", "kwargs"),
    [
        ("0", {"minimum": 1}),  # below minimum
        ("20", {"maximum": 10}),  # above maximum
        ("abc", {}),  # not a number
    ],
)
def test_number_prompt_rejects_then_reprompts(
    monkeypatch: pytest.MonkeyPatch, bad: str, kwargs: dict[str, int]
) -> None:
    # Each validation branch must REJECT the first entry and loop; the valid "7" is what's returned.
    _feed_text(monkeypatch, [bad, "7"])
    assert prompts.number_prompt("m", default=1, **kwargs) == 7


# --------------------------------------------------------------------------- confirm


@pytest.mark.parametrize("default", [True, False])
def test_confirm_enter_takes_default(monkeypatch: pytest.MonkeyPatch, default: bool) -> None:
    _script(monkeypatch, ["enter"])
    assert prompts.confirm("m", default=default) is default


def test_confirm_y_is_true_even_when_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, ["y"])
    assert prompts.confirm("m", default=False) is True


def test_confirm_non_y_is_false_even_when_default_true(monkeypatch: pytest.MonkeyPatch) -> None:
    # Any key other than y/Y is a "no", overriding a True default.
    _script(monkeypatch, ["n"])
    assert prompts.confirm("m", default=True) is False


def test_confirm_esc_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    _script(monkeypatch, ["esc"])
    with pytest.raises(Abort):
        prompts.confirm("m")


# --------------------------------------------------------------------------- read_key (byte decode)


def _fake_io(monkeypatch: pytest.MonkeyPatch, byte_chunks: list[bytes], *, pending: bool = False) -> None:
    """Replace read_key's OS seam so os.read yields the given chunks and select reports ``pending``."""
    reads: Iterator[bytes] = iter(byte_chunks)
    fake_os = types.SimpleNamespace(read=lambda fd, n: next(reads))
    fake_termios = types.SimpleNamespace(tcgetattr=lambda fd: [], tcsetattr=lambda *a: None, TCSADRAIN=1)
    fake_tty = types.SimpleNamespace(setraw=lambda fd: None)
    fake_select = types.SimpleNamespace(select=lambda *a: ([1] if pending else [], [], []))
    monkeypatch.setattr(prompts, "os", fake_os)
    monkeypatch.setattr(prompts, "termios", fake_termios)
    monkeypatch.setattr(prompts, "tty", fake_tty)
    monkeypatch.setattr(prompts, "select", fake_select)
    monkeypatch.setattr(sys.stdin, "fileno", lambda: 0)


@pytest.mark.parametrize(
    ("byte", "expected"),
    [
        (b"\r", "enter"),  # ASCII CR
        (b"\n", "enter"),  # ASCII LF
        (b"\x7f", "backspace"),  # ASCII DEL
        (b"\x08", "backspace"),  # ASCII BS
        (b"a", "a"),  # printable passthrough
    ],
)
def test_read_key_decodes_control_bytes(monkeypatch: pytest.MonkeyPatch, byte: bytes, expected: str) -> None:
    _fake_io(monkeypatch, [byte])
    assert prompts.read_key() == expected


def test_read_key_decodes_escape_arrow_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    # VT100/ANSI cursor-up is ESC [ A; with a pending suffix the two-byte tail "[A" maps to "up".
    _fake_io(monkeypatch, [b"\x1b", b"[A"], pending=True)
    assert prompts.read_key() == "up"


def test_read_key_bare_escape_is_esc(monkeypatch: pytest.MonkeyPatch) -> None:
    # ESC with nothing pending within the timeout is a plain cancel key, not an arrow prefix.
    _fake_io(monkeypatch, [b"\x1b"], pending=False)
    assert prompts.read_key() == "esc"


def test_read_key_eof_is_esc(monkeypatch: pytest.MonkeyPatch) -> None:
    # os.read returning b"" (EOF) is treated as cancel.
    _fake_io(monkeypatch, [b""])
    assert prompts.read_key() == "esc"


def test_read_key_ctrl_c_aborts(monkeypatch: pytest.MonkeyPatch) -> None:
    # ASCII ETX (0x03) is Ctrl-C -> Abort.
    _fake_io(monkeypatch, [b"\x03"])
    with pytest.raises(Abort):
        prompts.read_key()


# --------------------------------------------------------------------------- app.main guard / loop


def test_main_warns_and_returns_without_tty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # No TTY under pytest: main() must warn and return, never touching raw-tty read_key.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    app.main()
    assert "interactive terminal" in capsys.readouterr().out.lower()


def test_main_quit_selection_breaks_loop(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    # With a TTY and the "quit" pick, the loop exits with the sign-off, never dispatching a flow.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(app.prompts, "menu", lambda *a, **k: "quit")
    app.main()
    assert "thanks for using kavier" in capsys.readouterr().out.lower()
