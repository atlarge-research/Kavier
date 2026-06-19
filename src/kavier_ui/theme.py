"""Shared console, palette and banner for the Kavier interactive UI."""
from __future__ import annotations

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# Per-domain accent colours, reused across menus, prompts, spinners and result panels
# so each simulator reads as a consistent "channel".
ACCENTS: dict[str, str] = {
    "inference": "cyan",
    "training": "magenta",
    "energy": "green",
    "co2": "yellow",
    "neutral": "cyan",
}

# (key, label, accent, blurb) — the four simulators, surfaced on the main menu.
DOMAINS: list[tuple[str, str, str, str]] = [
    ("inference", "Inference", "cyan",
     "Prefill / decode latency, throughput, KV + prefix cache."),
    ("training", "Training", "magenta",
     "Step throughput, runtime, MFU and power for a fine-tune."),
    ("energy", "Energy", "green",
     "Energy, carbon and $ efficiency per million tokens."),
    ("co2", "Carbon", "yellow",
     "Grams of CO2 for a run against a carbon intensity."),
]

_LOGO = r"""
 _  __          _
| |/ /__ ___   _(_) ___ _ __
| ' // _` \ \ / / |/ _ \ '__|
| . \ (_| |\ V /| |  __/ |
|_|\_\__,_| \_/ |_|\___|_|
""".strip("\n")


def banner() -> Panel:
    logo = Text(_LOGO, style="bold cyan")
    sub = Text("LLM ecosystem simulator", style="dim")
    tag = Text("inference · training · energy · carbon", style="cyan")
    body = Align.center(Text("\n").join([logo, Text(), sub, tag]))
    return Panel(body, border_style="cyan", padding=(1, 4), title="[bold]interactive[/]", title_align="right")


def rule(text: str, accent: str = "cyan") -> Text:
    return Text.assemble(("  ", ""), (text, f"bold {accent}"))
