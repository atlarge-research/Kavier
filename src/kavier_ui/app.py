"""Kavier interactive CLI — guided REPL over the inference & training simulators.

Launch via the ``kavier`` console script (or ``python -m kavier_ui``): pick a simulator, answer
guided prompts, then chain into energy/carbon, export OpenDC, or tweak and re-run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

from kavier_library import GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY, UnknownSpecError
from kavier_ui import prompts, render, sims
from kavier_ui.prompts import Abort, Choice
from kavier_ui.theme import DOMAINS, banner, console

# --------------------------------------------------------------------------- #
# Shared prompt helpers
# --------------------------------------------------------------------------- #
_DEFAULT_MODEL = "Llama-3-8B"
_DEFAULT_GPU = "A10"


def _param_hint(spec: Any) -> str:
    """Clean size hint, e.g. '7B' or (for MoE) '47B (13B active)'."""
    total = spec.m_params / 1e9
    if spec.active_params < spec.m_params:
        return f"{total:,.0f}B ({spec.active_params / 1e9:,.0f}B active)"
    return f"{total:,.0f}B"


def _pick_model(accent: str, default: str = _DEFAULT_MODEL) -> str:
    # Sorted small → large (not raw alphabetical) so the list reads by size.
    names = sorted(sims.model_names(), key=lambda n: LLM_SPEC_LIBRARY[n].m_params)
    choices = [Choice(n, n, _param_hint(LLM_SPEC_LIBRARY[n])) for n in names]
    return str(prompts.fuzzy_select("Model", choices, accent=accent, default=default if default in names else names[0]))


def _pick_gpu(accent: str, default: str = _DEFAULT_GPU) -> str:
    names = sims.gpu_names()
    choices = [Choice(n, n, f"{GPU_SPEC_LIBRARY[n].memory_gb:,.0f}GB") for n in names]
    return str(prompts.fuzzy_select("GPU", choices, accent=accent, default=default if default in names else names[0]))


def _show_specs(model: str, gpu: str, accent: str) -> None:
    console.print(render.specs_panel(model, gpu, accent))


# --------------------------------------------------------------------------- #
# Inference flow
# --------------------------------------------------------------------------- #
def _inference_inputs(seed: dict[str, Any] | None = None) -> dict[str, Any]:
    s = seed or {}
    accent = "cyan"
    model = _pick_model(accent, s.get("model", _DEFAULT_MODEL))
    gpu = _pick_gpu(accent, s.get("gpu", _DEFAULT_GPU))
    _show_specs(model, gpu, accent)
    num_requests = prompts.number_prompt(
        "Number of requests:", default=s.get("num_requests", 128), minimum=1, accent=accent
    )
    input_tokens = prompts.number_prompt(
        "Input tokens / request:", default=s.get("input_tokens", 512), minimum=1, accent=accent
    )
    output_tokens = prompts.number_prompt(
        "Output tokens / request:", default=s.get("output_tokens", 128), minimum=1, accent=accent
    )
    kv_cache = prompts.confirm("Enable KV cache?", default=s.get("kv_cache", True), accent=accent)
    policy = prompts.menu(
        "Prefix-cache policy",
        [
            Choice("prefill", "prefill", "skip prefill on a cache hit"),
            Choice("full", "full", "skip prefill + decode"),
            Choice("none", "none", "disable prefix cache"),
        ],
        accent=accent,
        default={"prefill": 0, "full": 1, "none": 2}.get(s.get("prefix_policy", "prefill"), 0),
    )
    return {
        "model": model,
        "gpu": gpu,
        "num_requests": num_requests,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "kv_cache": kv_cache,
        "prefix_policy": policy,
        "prefix_min_tokens": s.get("prefix_min_tokens", 1024),
    }


def _flow_inference(seed: dict[str, Any] | None = None) -> None:
    accent = "cyan"
    inputs = _inference_inputs(seed)
    with render.spinner(f"Simulating {inputs['num_requests']:,} requests…", accent):
        result = sims.run_inference(inputs)
    console.print(render.inference_result(result))
    _inference_followups(result, inputs)


def _inference_followups(result: dict[str, Any], inputs: dict[str, Any]) -> None:
    while True:
        try:
            action = prompts.menu(
                "Next",
                [
                    Choice("energy", "Chain → Energy efficiency", "Wh / gCO2 / $ per Mtoken"),
                    Choice("co2", "Chain → Carbon", "gCO2 for this run"),
                    Choice("opendc", "Export OpenDC input", "tasks + fragments parquet"),
                    Choice("rerun", "Tweak & re-run", "edit inputs"),
                    Choice("back", "Back to main menu", ""),
                ],
                accent="cyan",
            )
        except Abort:
            return
        if action == "back":
            return
        if action == "rerun":
            _flow_inference(seed=inputs)
            return
        if action == "energy":
            price = prompts.number_prompt(
                "GPU $/hour (0 to skip cost):", default=2.5, minimum=0.0, accent="green", integer=False
            )
            with render.spinner("Computing efficiency…", "green"):
                e = sims.energy_from_inference(result, price if price > 0 else None)
            console.print(render.energy_result(e))
        elif action == "co2":
            intensity = prompts.number_prompt("Carbon intensity (gCO2/kWh):", default=400, minimum=1, accent="yellow")
            with render.spinner("Billing carbon…", "yellow"):
                c = sims.run_carbon_from_inference(result, float(intensity))
            console.print(render.carbon_result(c))
        elif action == "opendc":
            dst = prompts.text_prompt("Output dir:", default="kavier_opendc_out", accent="cyan")
            try:
                with render.spinner("Writing parquet…", "cyan"):
                    out = sims.export_opendc(result, Path(dst).expanduser())
                console.print(
                    f"[green]  ✓ wrote[/] [cyan]{out}/tasks.parquet[/] [green]+[/] [cyan]{out}/fragments.parquet[/]"
                )
            except Exception as exc:  # noqa: BLE001 — surface any IO/adapter error gracefully
                console.print(f"[red]  ✗ export failed: {exc}[/]")


# --------------------------------------------------------------------------- #
# Training flow
# --------------------------------------------------------------------------- #
def _training_inputs(seed: dict[str, Any] | None = None) -> dict[str, Any]:
    s = seed or {}
    accent = "magenta"
    model = _pick_model(accent, s.get("model", "mistral-7b-v0.1"))
    gpu = _pick_gpu(accent, s.get("gpu", "NVIDIA-A100-SXM4-80GB"))
    _show_specs(model, gpu, accent)
    method = prompts.menu(
        "Fine-tuning method",
        [
            Choice("lora", "lora", "low-rank adapters"),
            Choice("full", "full", "full fine-tune"),
            Choice("gptq-lora", "gptq-lora", "quantised LoRA"),
        ],
        accent=accent,
        default={"lora": 0, "full": 1, "gptq-lora": 2}.get(s.get("method", "lora"), 0),
    )
    batch_size = prompts.number_prompt("Batch size:", default=s.get("batch_size", 4), minimum=1, accent=accent)
    seq_len = prompts.number_prompt(
        "Tokens / sample (seq len):", default=s.get("seq_len", 1024), minimum=1, accent=accent
    )
    num_gpus = prompts.number_prompt("GPUs per node:", default=s.get("num_gpus", 8), minimum=1, accent=accent)
    num_nodes = prompts.number_prompt("Number of nodes:", default=s.get("num_nodes", 1), minimum=1, accent=accent)
    total_tokens = prompts.number_prompt(
        "Total tokens to train (0 to skip runtime):",
        default=s.get("total_tokens", 10_000_000),
        minimum=0,
        accent=accent,
    )
    return {
        "model": model,
        "gpu": gpu,
        "method": method,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "num_gpus": num_gpus,
        "num_nodes": num_nodes,
        "total_tokens": total_tokens,
    }


def _flow_training(seed: dict[str, Any] | None = None) -> None:
    inputs = _training_inputs(seed)
    with render.spinner("Simulating training step…", "magenta"):
        result = sims.run_training(inputs)
    console.print(render.training_result(result))
    while True:
        try:
            action = prompts.menu(
                "Next",
                [
                    Choice("co2", "Chain → Carbon", "gCO2 for this training run"),
                    Choice("rerun", "Tweak & re-run", "edit inputs"),
                    Choice("back", "Back to main menu", ""),
                ],
                accent="magenta",
            )
        except Abort:
            return
        if action == "back":
            return
        if action == "rerun":
            _flow_training(seed=inputs)
            return
        if action == "co2":
            if not inputs["total_tokens"]:
                console.print("[yellow]  set 'total tokens' > 0 to bill carbon — re-run first.[/]")
                continue
            intensity = prompts.number_prompt("Carbon intensity (gCO2/kWh):", default=400, minimum=1, accent="yellow")
            with render.spinner("Billing carbon…", "yellow"):
                c = sims.run_carbon_from_training({**inputs, "intensity": float(intensity)})
            console.print(render.carbon_result(c))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
# Top-level menu = the two run-producing simulators only. Energy and carbon are
# derived analyses, reached as follow-ups after an inference/training run.
_MAIN_MENU = ("inference", "training")
_FLOWS: dict[str, Callable[[], None]] = {
    "inference": _flow_inference,
    "training": _flow_training,
}


def main() -> None:
    if not sys.stdin.isatty():
        console.print(
            "[yellow]The Kavier UI needs an interactive terminal. "
            "Use the one-shot CLIs (kavier-perf / kavier-train / …) for scripted runs.[/]"
        )
        return
    console.print(banner())
    while True:
        choices = [Choice(k, label, blurb) for k, label, _accent, blurb in DOMAINS if k in _MAIN_MENU]
        choices.append(Choice("quit", "Quit", ""))
        try:
            pick = prompts.menu("Choose a simulator", choices, footer="↑↓ move · enter select · q quit")
        except Abort:
            break
        if pick == "quit":
            break
        try:
            _FLOWS[str(pick)]()
        except Abort:
            console.print("[dim]  cancelled — back to menu[/]")
        except UnknownSpecError as exc:
            console.print(f"[red]  ✗ {exc}[/]")
        except Exception as exc:  # noqa: BLE001 — never let one bad run kill the REPL
            console.print(f"[red]  ✗ simulation error: {exc}[/]")
    console.print("\n[cyan]  thanks for using Kavier 👋[/]\n")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Abort):
        console.print("\n[cyan]  bye 👋[/]\n")
        sys.exit(0)
