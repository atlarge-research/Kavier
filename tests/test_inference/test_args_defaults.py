"""Defaults and argument validation for ``kavier inference``.

A bare invocation must find the packaged example trace and write into ``kavier_output/`` regardless of
the working directory; the validated ``PerfArgs`` model must enforce the field bounds documented next to
each ``Field`` (positive export rate, non-negative counters, ``max_cached_prompts >= 1``, spec-key shape).
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

import kavier.sdk.inference
from kavier.cli.inference import DEFAULT_OUTPUT_FOLDER, DEFAULT_TRACE, PerfArgs, parse_args


def _valid_kwargs(**overrides) -> dict:
    """A fully-valid PerfArgs kwarg dict; callers override one field to probe a single constraint."""
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
    # Smoke guard so the negative tests below can't pass vacuously (a broken base dict would make every
    # "REJ" trivially true). The whole falsification burden lives in the per-constraint tests.
    PerfArgs.model_validate(_valid_kwargs())


# --- default trace: a real file, resolved inside the package (not the CWD) --------------------------


def test_default_trace_is_packaged_example() -> None:
    assert DEFAULT_TRACE.is_file(), f"packaged example trace missing: {DEFAULT_TRACE}"
    # Independent property: importlib.resources must resolve it under the kavier.sdk.inference package
    # tree, never a CWD-relative path. Mutating DEFAULT_TRACE to Path("data/input/input_example.csv")
    # would still be a file when run from the repo root but would NOT sit under pkg_root -> red.
    pkg_root = Path(kavier.sdk.inference.__file__).resolve().parent
    assert pkg_root in DEFAULT_TRACE.parents


# --- bare invocation: the documented default contract flows through parse+validate ------------------


def test_bare_invocation_populates_documented_defaults() -> None:
    # The CLI's public default contract (the ``default=`` values in _build_parser). This guards against
    # any accidental default change; e.g. flipping kv_cache default to "off" or gpu to "A100" -> red.
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
    # Cross-check of two independently-declared defaults: argparse uses DEFAULT_OUTPUT_FOLDER, and the
    # pydantic field default is spelled out separately. They used to disagree (src/data/output vs
    # data/output_traces). Retyping the pydantic default as Path("data/output_traces") -> red.
    assert PerfArgs.model_fields["output_folder"].default == DEFAULT_OUTPUT_FOLDER
    assert DEFAULT_OUTPUT_FOLDER == Path("kavier_output")


# --- numeric field bounds: floor accepted, just-below rejected --------------------------------------


@pytest.mark.parametrize(
    ("field", "floor", "below_floor"),
    [
        # ge=0: 0 is the "single-shot export" sentinel and must be accepted; -1 rejected.
        ("flush_size", 0, -1),
        # ge=0: a prefix length of 0 tokens is legal; negatives are not.
        ("prefix_cache_min_tokens", 0, -1),
        # ge=1: LRUCache(maxsize=0) raises KeyError on first insert, so 1 is the real floor.
        ("max_cached_prompts", 1, 0),
    ],
)
def test_integer_field_floor(field: str, floor: int, below_floor: int) -> None:
    PerfArgs.model_validate(_valid_kwargs(**{field: floor}))  # floor is accepted
    with pytest.raises(ValidationError):
        PerfArgs.model_validate(_valid_kwargs(**{field: below_floor}))  # one below is rejected


def test_export_rate_must_be_strictly_positive() -> None:
    # Field(gt=0): the runner divides by export_rate, so 0 (ZeroDivisionError) must be rejected while
    # any positive value passes. gt=0 vs ge=0 is the distinguishing behavior probed here.
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
        ("NVIDIA-A100-80GB.v2", True),  # dot/dash/digits are in the allowed char class
        (" H200", False),  # leading space
        ("H200 ", False),  # trailing space
        ("H200  SXM", False),  # doubled interior space
        ("", False),  # empty
    ],
)
def test_spec_key_pattern(field: str, value: str, valid: bool) -> None:
    # Oracle: the documented spec-key shape (words of [A-Za-z0-9._-] joined by single spaces). Derived
    # by hand from that rule, NOT by echoing the regex. Removing the pattern= would accept " H200" -> red.
    kwargs = _valid_kwargs(**{field: value})
    if valid:
        PerfArgs.model_validate(kwargs)
    else:
        with pytest.raises(ValidationError):
            PerfArgs.model_validate(kwargs)


# --- parse_args error paths: distinct exit codes for the two rejection layers -----------------------


def test_parse_args_exits_1_on_pydantic_rejection() -> None:
    # A leading space passes argparse (free-form str) but fails the PerfArgs pattern; the except
    # ValidationError branch must sys.exit(1). If it re-raised or exited 0 instead -> red.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--llm", " bad "])
    assert exc.value.code == 1


def test_parse_args_rejects_invalid_choice_with_code_2() -> None:
    # kv_cache is constrained by argparse choices={on,off}; an out-of-set value is argparse's own error,
    # which exits 2 (distinct from the pydantic-layer exit 1 above). Dropping choices= -> "maybe"
    # would reach pydantic (kv_cache is an unconstrained str there) and NOT exit -> red.
    with pytest.raises(SystemExit) as exc:
        parse_args(["--kv_cache", "maybe"])
    assert exc.value.code == 2


def test_default_argv_reads_process_args(monkeypatch: pytest.MonkeyPatch) -> None:
    # parse_args(None) must fall through argparse to sys.argv[1:] (not hardcode an empty list). With a
    # bad --gpu injected into argv, the default-argv path must surface the same exit(1) rejection.
    monkeypatch.setattr(sys, "argv", ["kavier", "--gpu", " nope "])
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 1
