"""Tests for the ``kavier`` CLI dispatcher ``kavier.cli.main``.

Covers six behaviours of ``main``:
  1. bare invocation              -> help to stderr, exit 2 (as git does)
  2. ``-V`` / ``--version`` first -> ``kavier <version>\\n`` to stdout, exit 0
  3. ``-h`` / ``--help`` first    -> help listing the subcommands to stdout, exit 0
  4. unknown command              -> argparse error to stderr listing the valid choices, exit 2
  5. valid command                -> routes to that command's engine with the trailing argv forwarded
  6. a top-level flag after a command goes to the engine

test_cli_contract.py runs the engines end-to-end. Here the engines are stubbed, and only the routing
table and the four engine-less paths are tested.
"""

from __future__ import annotations

import pytest

import kavier
from kavier.cli.main import _COMMANDS, main

# Subcommands in table order, listed independently of _COMMANDS so that dropping, renaming, reordering
# or adding a command fails a test. `calibrate` needs the [calibration] extra.
_SUBCOMMANDS = ("inference", "training", "cluster", "energy", "carbon", "calibrate")


def test_command_table_is_exactly_the_documented_subcommands() -> None:
    assert tuple(_COMMANDS) == _SUBCOMMANDS


def test_bare_invocation_prints_help_to_stderr_and_exits_2(capsys) -> None:
    # Like git, a bare invocation prints usage to stderr and exits 2 (usage error).
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "usage" in captured.err.lower()
    assert captured.out == ""  # help goes to stderr


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version_and_exits_0(flag, capsys) -> None:
    # Expected: "kavier <version>" and a newline on stdout, exit 0.
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"kavier {kavier.__version__}\n"


def test_help_exits_0_and_lists_every_subcommand(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in _SUBCOMMANDS:
        assert cmd in out


def test_unknown_command_exits_2_and_names_choice_and_valid_options(capsys) -> None:
    # argparse invalid choice: exit 2 and an error on stderr naming the bad token and the valid choices.
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err
    assert "bogus" in err  # the offending token is echoed back
    for cmd in _SUBCOMMANDS:
        assert cmd in err


@pytest.mark.parametrize("command", _SUBCOMMANDS)
def test_valid_command_routes_to_that_engine_with_trailing_argv_forwarded(command, monkeypatch) -> None:
    # `kavier <command> A B` calls that command's engine main with ["A", "B"], order preserved.
    # Every engine main is stubbed, so only the routed one records a call.
    calls: dict[str, list[str] | None] = {}
    for name in _SUBCOMMANDS:
        module_main = f"kavier.cli.{name}.main"
        monkeypatch.setattr(module_main, (lambda argv, _n=name: calls.__setitem__(_n, list(argv) if argv else argv)))

    main([command, "alpha", "beta"])

    assert calls == {command: ["alpha", "beta"]}  # one engine called, command name stripped


def test_help_after_a_command_is_forwarded_not_intercepted(monkeypatch) -> None:
    # The root checks only argv[0] for -h/-V, so `kavier inference --help` reaches the inference engine,
    # which prints its own help. A root that scanned all of argv for these flags would fail this test.
    forwarded: list[str] = []
    monkeypatch.setattr("kavier.cli.inference.main", lambda argv: forwarded.extend(argv))

    main(["inference", "--help"])

    assert forwarded == ["--help"]
