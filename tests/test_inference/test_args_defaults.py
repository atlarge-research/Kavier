"""Defaults and argument validation for ``kavier inference``.

A bare invocation reads the packaged example trace and writes to ``kavier_output/`` from any working
directory. ``PerfArgs`` enforces the bounds documented on each ``Field``: positive export rate,
non-negative counters, ``max_cached_prompts >= 1``, and the spec-key shape.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import kavier.sdk.inference
from kavier.cli.inference import DEFAULT_OUTPUT_FOLDER, DEFAULT_TRACE, PerfArgs, parse_args


def _valid_kwargs(**overrides) -> dict:
    """Return valid PerfArgs kwargs with one or more fields overridden."""
    base = {
        "llm": "Llama-3-8B",
        "gpu": "A10",
        "trace": Path("t.csv"),
        "output_folder": Path("out"),
        "kv_cache": "on",
        "export_rate": 0.1,
        "flush_size": 0,
        "prefix_cache_min_tokens": 0,
        "max_cached_prompts": 1,
        "cache_scope": "session",
        "prefix_cache_policy": "prefill",
    }
    base.update(overrides)
    return base


def test_valid_kwargs_helper_actually_validates() -> None:
    # Without this, an invalid base dict would make every rejection test below pass.
    PerfArgs.model_validate(_valid_kwargs())


# --- default trace: a packaged file, independent of the CWD -----------------------------------------


def test_default_trace_is_packaged_example() -> None:
    assert DEFAULT_TRACE.is_file(), f"packaged example trace missing: {DEFAULT_TRACE}"
    # Must resolve inside the kavier.sdk.inference package. A CWD-relative path would still be a file
    # when run from the repo root, but would not sit under pkg_root.
    pkg_root = Path(kavier.sdk.inference.__file__).resolve().parent
    assert pkg_root in DEFAULT_TRACE.parents


# --- bare invocation: documented defaults after parsing and validation ------------------------------


def test_bare_invocation_populates_documented_defaults() -> None:
    # The ``default=`` values in _build_parser.
    args = parse_args([])
    assert args.llm == "Llama-3-8B"
    assert args.gpu == "A10"
    assert args.trace == DEFAULT_TRACE
    assert args.trace.is_file()
    assert args.output_folder == Path("kavier_output")
    assert args.kv_cache == "on"
    assert args.export_rate == 0.1
    assert args.flush_size == 1000
    assert args.prefix_cache_min_tokens == 1024
    assert args.max_cached_prompts == 10
    assert args.cache_scope == "session"
    assert args.prefix_cache_policy == "prefill"


def test_output_folder_default_matches_shared_constant() -> None:
    # argparse uses DEFAULT_OUTPUT_FOLDER; the pydantic field default is declared separately.
    assert PerfArgs.model_fields["output_folder"].default == DEFAULT_OUTPUT_FOLDER
    assert DEFAULT_OUTPUT_FOLDER == Path("kavier_output")


# --- numeric field bounds: floor accepted, floor - 1 rejected ---------------------------------------


@pytest.mark.parametrize(
    ("field", "floor", "below_floor"),
    [
        # ge=0: 0 means a single export at the end.
        ("flush_size", 0, -1),
        # ge=0
        ("prefix_cache_min_tokens", 0, -1),
        # ge=1: with maxsize=0 the first insert raises KeyError.
        ("max_cached_prompts", 1, 0),
    ],
)
def test_integer_field_floor(field: str, floor: int, below_floor: int) -> None:
    PerfArgs.model_validate(_valid_kwargs(**{field: floor}))  # floor is accepted
    with pytest.raises(ValidationError):
        PerfArgs.model_validate(_valid_kwargs(**{field: below_floor}))  # one below is rejected


def test_export_rate_must_be_strictly_positive() -> None:
    # Field(gt=0): the runner divides by export_rate.
    PerfArgs.model_validate(_valid_kwargs(export_rate=1e-9))
    with pytest.raises(ValidationError):
        PerfArgs.model_validate(_valid_kwargs(export_rate=0.0))


# --- spec-key pattern: single-space-separated tokens, no leading/trailing/doubled whitespace ---------


@pytest.mark.parametrize("field", ["llm", "gpu"])
@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("H200", True),  # a bare token
        ("H200 SXM", True),  # single interior space between tokens
        ("NVIDIA-A100-80GB.v2", True),  # dots, dashes, and digits allowed
        (" H200", False),  # leading space
        ("H200 ", False),  # trailing space
        ("H200  SXM", False),  # doubled interior space
        ("", False),  # empty
    ],
)
def test_spec_key_pattern(field: str, value: str, valid: bool) -> None:
    # Spec keys are words of [A-Za-z0-9._-] joined by single spaces.
    kwargs = _valid_kwargs(**{field: value})
    if valid:
        PerfArgs.model_validate(kwargs)
    else:
        with pytest.raises(ValidationError):
            PerfArgs.model_validate(kwargs)


# --- parse_args error paths: distinct exit codes for the two rejection layers -----------------------


def test_parse_args_exits_1_on_pydantic_rejection() -> None:
    # A leading space passes argparse but fails the PerfArgs pattern -> sys.exit(1).
    with pytest.raises(SystemExit) as exc:
        parse_args(["--llm", " bad "])
    assert exc.value.code == 1


def test_parse_args_rejects_invalid_choice_with_code_2() -> None:
    # argparse choices={on,off} rejects "maybe" with exit code 2. PerfArgs does not constrain kv_cache.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--kv_cache", "maybe"])
    assert exc.value.code == 2


def test_default_argv_reads_process_args(monkeypatch: pytest.MonkeyPatch) -> None:
    # parse_args(None) reads sys.argv[1:].
    monkeypatch.setattr(sys, "argv", ["kavier", "--gpu", " nope "])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 1
