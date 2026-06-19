"""Rich rendering for spec details, run summaries and per-domain result tables."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from rich.columns import Columns
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from kavier_library import get_gpu, get_llm
from kavier_ui.theme import console


@contextmanager
def spinner(message: str, accent: str = "cyan") -> Iterator[None]:
    with Progress(SpinnerColumn(style=accent), TextColumn("[progress.description]{task.description}"),
                  TimeElapsedColumn(), console=console, transient=True) as prog:
        prog.add_task(f"[{accent}]{message}", total=None)
        yield


def _kv_table(rows: list[tuple[str, str]], accent: str) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(justify="right", style="dim")
    t.add_column(style="white")
    for k, v in rows:
        t.add_row(k, v)
    return t


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:,.1f}s"
    if s < 5400:
        return f"{s / 60:,.1f}m"
    return f"{s / 3600:,.2f}h"


def specs_panel(model: str, gpu: str, accent: str = "cyan") -> Panel:
    """Side-by-side model + GPU spec cards so the user sees what they picked."""
    llm = get_llm(model)
    g = get_gpu(gpu)
    m = _kv_table([
        ("params", f"{llm.m_params / 1e9:,.1f} B"),
        ("active", f"{llm.active_params / 1e9:,.1f} B"),
        ("layers", str(llm.n_layers)),
        ("d_model", f"{llm.d_model:,}"),
        ("heads", str(llm.n_heads)),
        ("bytes/param", str(llm.p_bytes)),
    ], accent)
    gp = _kv_table([
        ("memory", f"{g.memory_gb:,.0f} GB"),
        ("FP16 TC", f"{g.fp_16_tensor_core_tflops:,} TFLOPs"),
        ("bandwidth", f"{g.bandwidth_bps / 1e9:,.0f} GB/s"),
        ("max power", f"{g.max_power_w:,.0f} W"),
        ("MFU factor", f"{g.mfu_factor:.2f}"),
        ("calib", f"{g.calibration_factor:.2f}"),
    ], accent)
    return Panel(Columns([
        Panel(m, title=f"[bold]{llm.name}[/]", border_style=accent, padding=(0, 2)),
        Panel(gp, title=f"[bold]{g.name}[/]", border_style=accent, padding=(0, 2)),
    ], equal=True, expand=True), title="[bold]selected specs[/]", border_style="dim", padding=(0, 1))


def _result_table(title: str, rows: list[tuple[str, str, str]], accent: str) -> Table:
    """rows = (metric, value, unit/hint)."""
    t = Table(title=f"[bold {accent}]{title}[/]", title_justify="left", header_style=f"bold {accent}",
              border_style=accent, expand=True)
    t.add_column("Metric", style="white")
    t.add_column("Value", justify="right", style=f"bold {accent}")
    t.add_column("", style="dim")
    for metric, value, unit in rows:
        t.add_row(metric, value, unit)
    return t


def inference_result(r: dict[str, Any]) -> Panel:
    config = _kv_table([
        ("requests", f"{r['num_requests']:,}"),
        ("in/out tokens", f"{r['input_tokens']:,} / {r['output_tokens']:,}"),
        ("KV cache", "on" if r["kv_cache"] else "off"),
        ("prefix policy", f"{r['prefix_policy']} (≥{r['prefix_min_tokens']}t)"),
    ], "cyan")
    table = _result_table("Inference results", [
        ("Prefill time", _fmt_secs(r["prefill_s"]), "total"),
        ("Decode time", _fmt_secs(r["decode_s"]), "total"),
        ("Total time", _fmt_secs(r["total_s"]), "wall"),
        ("Throughput", f"{r['throughput_tok_s']:,.0f}", "tok/s"),
        ("Throughput", f"{r['throughput_req_s']:,.2f}", "req/s"),
        ("Mean TTFT", f"{r['mean_ttft_ms']:,.0f}", "ms"),
        ("Latency p50 / p95 / p99", f"{r['p50_ms']:,.0f} / {r['p95_ms']:,.0f} / {r['p99_ms']:,.0f}", "ms"),
        ("Cache hit ratio", f"{r['cache_hit_ratio']:.1%}", f"{r['cache_hits']:,} hits"),
        ("LRU evictions", f"{r['evictions']:,}", ""),
    ], "cyan")
    return Panel(Columns([config, table], expand=True), title="[bold cyan]Inference[/]",
                 border_style="cyan", padding=(1, 2))


def training_result(r: dict[str, Any]) -> Panel:
    config = _kv_table([
        ("method", r["method"]),
        ("batch × seq", f"{r['batch_size']} × {r['tokens_per_sample']:,}"),
        ("GPUs", f"{r['total_gpus']}"),
        ("total tokens", f"{r['total_tokens']:,}" if r["total_tokens"] else "—"),
    ], "magenta")
    rows = [
        ("Throughput", f"{r['train_tokens_per_second']:,.0f}", "tok/s"),
        ("Per-GPU throughput", f"{r['train_tokens_per_gpu_per_second']:,.0f}", "tok/s/GPU"),
        ("Samples/s", f"{r['train_samples_per_second']:,.2f}", "samp/s"),
        ("Steps/s", f"{r['train_steps_per_second']:,.3f}", "step/s"),
        ("Step time", f"{r['step_time_ms']:,.0f}", "ms"),
        ("MFU", f"{r['gpu_compute_utilization']:,.1f}", "%"),
        ("Memory util", f"{r['gpu_memory_utilization']:,.1f}", "%"),
        ("Power / GPU", f"{r['gpu_power_watts']:,.0f}", "W"),
        ("Aggregate power", f"{r['aggregate_power_w']:,.0f}", "W"),
    ]
    if r["total_tokens"]:
        rows.append(("Training runtime", _fmt_secs(r["train_runtime"]), "wall"))
    return Panel(Columns([config, _result_table("Training results", rows, "magenta")], expand=True),
                 title="[bold magenta]Training[/]", border_style="magenta", padding=(1, 2))


def energy_result(r: dict[str, Any]) -> Panel:
    fin = f"{r['financial_per_mtoken']:,.4f}" if r["financial_per_mtoken"] is not None else "N/A"
    rows = [
        ("Total energy", f"{r['energy_wh']:,.1f}", "Wh"),
        ("Total energy", f"{r['energy_kwh']:,.4f}", "kWh"),
        ("Energy efficiency", f"{r['energy_per_mtoken_wh']:,.1f}", "Wh / Mtoken"),
        ("Carbon efficiency", f"{r['carbon_per_mtoken_g']:,.1f}", "gCO2 / Mtoken"),
        ("Cost efficiency", fin, "$ / Mtoken"),
        ("Tokens / Wh", f"{r['tokens_per_wh']:,.1f}", "tok/Wh"),
        ("GPU-hours", f"{r['gpu_hours']:,.3f}", "h"),
    ]
    return Panel(_result_table("Energy & efficiency", rows, "green"),
                 title="[bold green]Energy[/] [dim](lower per-Mtoken = better)[/]",
                 border_style="green", padding=(1, 2))


def carbon_result(r: dict[str, Any]) -> Panel:
    per_m = (r["total_co2_g"] / r["total_tokens"] * 1e6) if r["total_tokens"] else 0.0
    rows = [
        ("Source", r["source"], ""),
        ("Carbon intensity", f"{r['intensity']:,.0f}", "gCO2/kWh"),
        ("Run power", f"{r['power_w']:,.0f}", "W"),
        ("Runtime", _fmt_secs(r["runtime_s"]), "wall"),
        ("Energy", f"{r['total_energy_kwh']:,.4f}", "kWh"),
        ("Emissions", f"{r['total_co2_g']:,.1f}", "gCO2"),
        ("Emissions", f"{r['total_co2_kg']:,.4f}", "kgCO2"),
    ]
    if r["total_tokens"]:
        rows.append(("Carbon / Mtoken", f"{per_m:,.1f}", "gCO2"))
    return Panel(_result_table("Carbon emissions", rows, "yellow"),
                 title="[bold yellow]Carbon[/]", border_style="yellow", padding=(1, 2))
