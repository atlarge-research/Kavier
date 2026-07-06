"""Contract for the unified ``kavier`` CLI dispatcher (``kavier.cli.main``).

Covers the six distinguishable behaviors of ``main``:
  1. bare invocation      -> help to stderr, exit 2 (git-style)
  2. ``-V`` / ``--version`` first -> ``kavier <version>\\n`` to stdout, exit 0
  3. ``-h`` / ``--help`` first    -> help listing the subcommands to stdout, exit 0
  4. unknown command      -> argparse error to stderr listing the valid choices, exit 2
  5. valid command        -> routes to THAT command's engine with the trailing argv forwarded
  6. a top-level flag AFTER a command is forwarded to the engine, not intercepted by the root

The subcommand engines themselves are exercised end-to-end by test_cli_contract.py; here we only
pin the routing table and the four engine-less surfaces, stubbing the engines so no real run occurs.
"""

from __future__ import annotations

import pytest

import kavier
from kavier.cli.main import _COMMANDS, main

# The subcommands the dispatcher must expose, in table order. Kept explicit (not read from _COMMANDS)
# so the test is an INDEPENDENT statement of the contract: dropping/renaming/reordering a command, or
# adding an undocumented one, must turn a test red. (`calibrate` needs the [calibration] extra.)
_SUBCOMMANDS = ("inference", "training", "energy", "carbon", "calibrate")


def test_command_table_is_exactly_the_documented_subcommands() -> None:
    # Oracle: the public contract promises these commands, in this order, and no others.
    # Falsification: adding an undocumented command, dropping one (e.g. "carbon"), or reordering.
    assert tuple(_COMMANDS) == _SUBCOMMANDS


def test_bare_invocation_prints_help_to_stderr_and_exits_2(capsys) -> None:
    # Oracle: git-style dispatchers exit 2 (usage error) on a bare invocation and print usage.
    # Falsification: `return` instead of `raise SystemExit(2)`, or exit 0, or help sent to stdout.
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "usage" in captured.err.lower()
    assert captured.out == ""  # help must go to stderr, never stdout


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag_prints_version_and_exits_0(flag, capsys) -> None:
    # Oracle: the advertised format is the literal "kavier <version>\n" on stdout, exit 0.
    # __version__ is imported independently here; the load-bearing part is the "kavier " prefix
    # and the trailing newline. Falsification: dropping the prefix, wrong stream, or non-zero exit.
    with pytest.raises(SystemExit) as exc:
        main([flag])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out == f"kavier {kavier.__version__}\n"


def test_help_exits_0_and_lists_every_subcommand(capsys) -> None:
    # Oracle: --help is a success (exit 0) and the top-level help must advertise every subcommand.
    # Falsification: exit != 0, or a command missing from the help body.
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in _SUBCOMMANDS:
        assert cmd in out


def test_unknown_command_exits_2_and_names_choice_and_valid_options(capsys) -> None:
    # Oracle: argparse's invalid-choice contract -> exit 2, error to stderr naming the bad token
    # and the valid choices. Falsification: silently dispatching a bogus command, or exit 0, or an
    # error that omits the offending token / the choice list.
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
    # THE dispatcher's job: `kavier <command> A B` calls THAT command's engine main with exactly
    # ["A", "B"] (command name stripped, order preserved). We stub every engine main so no real run
    # happens, then assert only the routed one fired and with the right tail.
    # Oracle: forwarded argv == input argv minus the leading command token.
    # Falsification: routing "energy" to the carbon engine, dropping/reordering the tail, or a
    # missing route (KeyError) all turn this red.
    calls: dict[str, list[str] | None] = {}
    for name in _SUBCOMMANDS:
        module_main = f"kavier.cli.{name}.main"
        monkeypatch.setattr(module_main, (lambda argv, _n=name: calls.__setitem__(_n, list(argv) if argv else argv)))

    main([command, "alpha", "beta"])

    assert calls == {command: ["alpha", "beta"]}  # exactly one engine fired, with the stripped tail


def test_help_after_a_command_is_forwarded_not_intercepted(monkeypatch) -> None:
    # The root only inspects argv[0] for -h/-V, so `kavier inference --help` must reach the inference
    # engine (which renders ITS own help) rather than the root printing top-level help and exiting.
    # Oracle: the engine main receives ["--help"]; no SystemExit escapes from the root dispatch.
    # Falsification: broadening the top-level flag check to scan all of argv would swallow "--help"
    # here and this would fail (engine never called / SystemExit raised).
    forwarded: list[str] = []
    monkeypatch.setattr("kavier.cli.inference.main", lambda argv: forwarded.extend(argv))

    main(["inference", "--help"])

    assert forwarded == ["--help"]
