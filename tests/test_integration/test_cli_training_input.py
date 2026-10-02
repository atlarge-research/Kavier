"""Input errors in ``kavier training`` end in a parser error (exit 2, one message) before any simulation."""

from __future__ import annotations

import pytest

from kavier.cli.training import main

_HEADER = "model_name,method,gpu_model,tokens_per_sample,batch_size,number_gpus,number_nodes\n"
_ROW = "mistral-7b-v0.1,{method},NVIDIA-A100-SXM4-80GB,1024,4,8,1\n"

_JOB_FLAGS = [
    "--model_name", "mistral-7b-v0.1",
    "--method", "lora",
    "--gpu_model", "NVIDIA-A100-SXM4-80GB",
    "--tokens_per_sample", "1024",
    "--batch_size", "4",
    "--number_gpus", "8",
    "--number_nodes", "1",
]  # fmt: skip


def _exit_code(argv: list[str]) -> int | str | None:
    with pytest.raises(SystemExit) as ei:
        main(argv)
    return ei.value.code


def test_csv_row_with_unknown_method_is_rejected_with_its_line(tmp_path, capsys):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(_HEADER + _ROW.format(method="lora") + _ROW.format(method="lorra"))
    assert _exit_code(["--input_csv", str(csv_path)]) == 2
    out, err = capsys.readouterr()
    assert "line 3" in err and "lorra" in err and "gptq-lora" in err
    assert "configurations simulated" not in out


@pytest.mark.parametrize("method", ["qlora", "alora"])
def test_csv_accepts_every_method_the_engine_accepts(tmp_path, capsys, method):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(_HEADER + _ROW.format(method=method))
    main(["--input_csv", str(csv_path)])
    assert "1 configurations simulated." in capsys.readouterr().out


@pytest.mark.parametrize("method", ["qlora", "alora"])
def test_method_flag_accepts_every_method_the_engine_accepts(capsys, method):
    argv = [*_JOB_FLAGS]
    argv[argv.index("lora")] = method
    main(argv)
    assert f"Method: {method}" in capsys.readouterr().out


def test_csv_saved_with_utf8_bom_is_read(tmp_path, capsys):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_bytes(b"\xef\xbb\xbf" + (_HEADER + _ROW.format(method="lora")).encode())
    main(["--input_csv", str(csv_path)])
    assert "1 configurations simulated." in capsys.readouterr().out


def test_unknown_model_is_a_parser_error(capsys):
    argv = [*_JOB_FLAGS]
    argv[argv.index("mistral-7b-v0.1")] = "not-a-real-model"
    assert _exit_code(argv) == 2
    err = capsys.readouterr().err
    assert "kavier training: error:" in err and "not-a-real-model" in err


def test_epochs_without_dataset_tokens_errors_before_the_banner(capsys):
    assert _exit_code([*_JOB_FLAGS, "--epochs", "2"]) == 2
    out, err = capsys.readouterr()
    assert "dataset_tokens" in err
    assert "Kavier Training Simulator" not in out


def test_config_method_outside_choices_is_rejected(tmp_path, capsys):
    cfg = tmp_path / "job.yaml"
    cfg.write_text("method: bogus\n")
    flags = [*_JOB_FLAGS]
    del flags[2:4]  # drop --method lora so the config value is used
    assert _exit_code(["--config", str(cfg), *flags]) == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err and "bogus" in err
