"""``kavier carbon`` CLI plumbing: token-source resolution, ``--config`` folding, and the carbon integral.

These exercise ``kavier.cli.carbon.main`` (argument wiring + error paths) plus a cross-check of the CLI's
carbon integration against a closed-form energy/CO2 computation. The training-sim sizing (power_w, runtime_s)
is read back from ``fragments_from_training`` and used only as the *input* to an independent oracle — the
windowed integrator in ``compute_emissions`` is what is under test here.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from kavier.cli.carbon import main
from kavier.sdk.co2.fragments import fragments_from_training

# Constant carbon intensity of the synthetic trace (gCO2/kWh). With a flat trace the down-estimation
# min(own, next) collapses to this single value, so the CLI's per-window integral has an exact closed form.
_INTENSITY = 150.0

_BASE_ARGS = [
    "--model_name",
    "mistral-7b-v0.1",
    "--method",
    "lora",
    "--gpu_model",
    "NVIDIA-A100-SXM4-80GB",
    "--tokens_per_sample",
    "1024",
    "--batch_size",
    "4",
    "--number_gpus",
    "8",
    "--number_nodes",
    "1",
    "--start_time",
    "2025-06-01 00:00",
]


@pytest.fixture()
def small_trace(tmp_path):
    # 4000 half-hour windows from the run start => ~83 days of coverage, far longer than any runtime here,
    # so every fragment lands fully inside the trace and no coverage error is raised.
    ts = pd.date_range("2025-06-01 00:00", periods=4000, freq="30min")
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": [_INTENSITY] * len(ts)})
    p = tmp_path / "carbon.parquet"
    df.to_parquet(p)
    return str(p)


def _co2_grams(out: str) -> float:
    m = re.search(r"Total CO2:\s+([\d,.]+) g", out)
    assert m, f"no 'Total CO2' line in output:\n{out}"
    return float(m.group(1).replace(",", ""))


def _closed_form_co2_grams(total_tokens: int) -> float:
    """Independent CO2 oracle: single fragment fully inside a flat-intensity trace.

    Uses the training-sim fragment (power, runtime) as input, then applies the carbon integral in closed
    form — NOT the windowed accumulation ``compute_emissions`` uses. Physical reference: 1 kWh = 3.6e6 W*s.
    """
    frag = fragments_from_training(
        model_name="mistral-7b-v0.1",
        method="lora",
        gpu_model="NVIDIA-A100-SXM4-80GB",
        tokens_per_sample=1024,
        batch_size=4,
        number_gpus=8,
        number_nodes=1,
        total_tokens=total_tokens,
        start_time=pd.Timestamp("2025-06-01 00:00"),
    )[0]
    energy_kwh = frag.power_w * frag.duration_s / 3.6e6  # W * s / (W*s per kWh)
    return energy_kwh * _INTENSITY


def test_from_training_co2_matches_closed_form(small_trace, capsys):
    # Cross-check the CLI's windowed carbon integral against the flat-trace closed form. A /1000 (Wh<->kWh)
    # or an off-by-one in the window accumulation would move the printed grams away from this oracle.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    printed = _co2_grams(capsys.readouterr().out)

    expected = _closed_form_co2_grams(10_000_000)
    # Printed value is rounded to 2 decimals (".2f"); abs=0.01 covers that rounding, nothing looser.
    assert printed == pytest.approx(expected, abs=0.01)


def test_epochs_dataset_tokens_parity_with_total_tokens(small_trace, capsys):
    # epochs * dataset_tokens = 2 * 5_000_000 = 10_000_000, resolved to the same job size as --total_tokens.
    # Falsifier: dropping --epochs/--dataset_tokens wiring (or a rounding bug in _resolve_total_tokens)
    # makes the two runs diverge or the epochs run error out.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_total = _co2_grams(capsys.readouterr().out)

    main(
        ["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--epochs", "2", "--dataset_tokens", "5000000"]
    )
    by_epochs = _co2_grams(capsys.readouterr().out)

    assert by_epochs == pytest.approx(by_total)


def test_co2_scales_linearly_with_total_tokens(small_trace, capsys):
    # Runtime = total_tokens / tokens_per_second and energy = power * runtime, so CO2 is linear in tokens:
    # halving the job halves the emissions. Falsifier: a constant/clamped runtime, or a quadratic term.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    co2_10m = _co2_grams(capsys.readouterr().out)

    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "5000000"])
    co2_5m = _co2_grams(capsys.readouterr().out)

    assert co2_5m == pytest.approx(co2_10m / 2.0, rel=1e-3)


def test_missing_token_source_errors(small_trace, capsys):
    # No --total_tokens and no --epochs/--dataset_tokens => parser.error => SystemExit naming the token flag.
    # Falsifier: removing the required-token-source check lets the run proceed with total_tokens=None.
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS])
    err = capsys.readouterr().err
    assert "total_tokens" in err


def test_epochs_without_dataset_tokens_errors(small_trace, capsys):
    # --epochs alone is not a complete token source (needs --dataset_tokens too). Falsifier: turning the
    # "epochs AND dataset_tokens" guard into an OR would accept this and then raise an *uncaught* ValueError
    # instead of the clean SystemExit — so pytest.raises(SystemExit) would go red.
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--epochs", "2"])
    err = capsys.readouterr().err
    assert "total_tokens" in err and "dataset_tokens" in err


def test_config_yaml_matches_flags(small_trace, tmp_path, capsys):
    # A config file supplying the same values as the flags must produce the same emissions.
    # Falsifier: if --config were ignored, the config run would error on missing required args.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_flags = _co2_grams(capsys.readouterr().out)

    cfg = tmp_path / "co2.yaml"
    cfg.write_text(
        "model_name: mistral-7b-v0.1\n"
        "method: lora\n"
        "gpu_model: NVIDIA-A100-SXM4-80GB\n"
        "tokens_per_sample: 1024\n"
        "batch_size: 4\n"
        "number_gpus: 8\n"
        "number_nodes: 1\n"
        "total_tokens: 10000000\n"
        'start_time: "2025-06-01 00:00"\n'
    )
    main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg)])
    by_config = _co2_grams(capsys.readouterr().out)
    assert by_config == pytest.approx(by_flags)


def test_explicit_flag_overrides_config(small_trace, tmp_path, capsys):
    # Config is folded as *defaults*; an explicit flag must win. Config says 5M, flag says 10M => 10M result.
    # Falsifier: if config values overrode explicit flags, we'd get the 5M number (~half), not the 10M one.
    main(["--from-training", "--carbon_trace", small_trace, *_BASE_ARGS, "--total_tokens", "10000000"])
    by_flags_10m = _co2_grams(capsys.readouterr().out)

    cfg = tmp_path / "co2.yaml"
    cfg.write_text(
        "model_name: mistral-7b-v0.1\n"
        "method: lora\n"
        "gpu_model: NVIDIA-A100-SXM4-80GB\n"
        "tokens_per_sample: 1024\n"
        "batch_size: 4\n"
        "number_gpus: 8\n"
        "number_nodes: 1\n"
        "total_tokens: 5000000\n"
        'start_time: "2025-06-01 00:00"\n'
    )
    main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg), "--total_tokens", "10000000"])
    by_override = _co2_grams(capsys.readouterr().out)
    assert by_override == pytest.approx(by_flags_10m)


def test_config_unknown_key_errors(small_trace, tmp_path, capsys):
    # A key that is not a parser dest must be rejected (not silently ignored), naming the offending key.
    # Falsifier: dropping the unknown-key validation would let this parse and run.
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("model_name: mistral-7b-v0.1\nbogus_key: 1\n")
    with pytest.raises(SystemExit):
        main(["--from-training", "--carbon_trace", small_trace, "--config", str(cfg)])
    err = capsys.readouterr().err
    assert "unknown config key" in err.lower()
    assert "bogus_key" in err
