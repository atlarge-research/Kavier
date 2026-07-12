"""Render the operational cluster-timeline figure for a :class:`ClusterSimResult`.

Draws GPUs-in-use (filled area, primary axis) + jobs-in-queue (step line, twin axis) over time, taken
straight from the result's timeline and cluster metrics. This is a duplicate of coastline's
``plot_trace_timeline`` rendering, adapted to consume Kavier's result object directly.

matplotlib is an optional dependency (the ``[plot]`` extra), imported lazily inside
:func:`plot_timeline`, so importing this module — and the rest of ``kavier.sdk.cluster`` — stays light
and does not require matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kavier.sdk.cluster.facade import ClusterSimResult

_GPU_FILL = "#0072B2"  # colourblind blue for the GPUs-in-use area


def _is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def _savefig(fig: Any, path: str) -> None:
    """A ``.pdf`` path -> reproducible vector PDF (timestamp stripped), thesis-ready; else a 130-dpi raster."""
    if _is_pdf(path):
        fig.savefig(path, metadata={"CreationDate": None})
    else:
        fig.savefig(path, dpi=130)


def plot_timeline(result: ClusterSimResult, output_path: str, *, title: str | None = None) -> dict[str, Any]:
    """Draw ``result``'s operational timeline (GPUs in use + jobs queued over time) to ``output_path``.

    ``output_path`` ending in ``.pdf`` writes a reproducible vector PDF (no title, thesis-ready);
    any other extension writes a raster and, unless ``title`` is given, an auto-generated title.
    Pass ``title=""`` to suppress the title on a raster. Returns a small stats dict
    (jobs, cluster_gpus, makespan_h, peak_gpus, peak_queue). Requires the ``[plot]`` extra (matplotlib).
    """
    try:
        import matplotlib
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "plotting needs matplotlib — install the 'plot' extra: `uv sync --extra plot` "
            "(or `pip install 'kavier[plot]'`)"
        ) from exc

    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    cluster_gpus = result.cluster.capacity_gpus
    makespan_h = result.cluster.makespan_h
    peak_gpus = result.cluster.peak_gpus
    peak_queue = result.cluster.peak_queue
    n_jobs = result.cluster.n_jobs
    gpu_t = q_t = result.timeline.times_h
    gpu_v = result.timeline.gpus_in_use
    q_v = result.timeline.queue_depth
    t_end = gpu_t[-1] if gpu_t else 0.0

    if title is None and not _is_pdf(output_path):
        title = (
            f"Cluster timeline · {n_jobs} jobs · {cluster_gpus} GPUs · "
            f"makespan {makespan_h:.1f} h · peak queue {peak_queue}"
        )

    fs_label, fs_tick, fs_legend, fs_title = 14, 12, 13, 13
    gpu_dark = "#005a8d"  # darker blue: GPU area edge + left-axis label/ticks (matches the fill)
    queue_col = "black"  # the jobs-in-queue series + its (right) axis
    cap_col = "#8a8a8a"  # the dashed cluster-capacity line

    fig, ax = plt.subplots(figsize=(9, 3.9))

    # Light horizontal grid keyed to the GPU axis, kept behind the data.
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d3d3d3", lw=0.6, alpha=0.7, zorder=0)

    # Left axis — GPUs in use: filled blue area with a crisp darker-blue top edge.
    ax.fill_between(gpu_t, gpu_v, color=_GPU_FILL, alpha=0.45, lw=0, zorder=2)
    ax.plot(gpu_t, gpu_v, color=gpu_dark, lw=1.1, alpha=0.9, zorder=2.5)
    ax.axhline(cluster_gpus, ls=(0, (6, 4)), color=cap_col, lw=1.3, zorder=1)  # cluster cap
    ax.set_ylim(0, cluster_gpus * 1.08)
    ax.set_xlim(0, t_end if t_end > 0 else 1)
    ax.set_yticks(sorted({0, cluster_gpus // 2, cluster_gpus}))
    ax.set_xlabel("Time [h]", fontsize=fs_label)
    ax.set_ylabel("GPUs in use", fontsize=fs_label, color=gpu_dark)
    ax.tick_params(labelsize=fs_tick)
    ax.tick_params(axis="y", colors=gpu_dark)
    ax.spines["left"].set_color(gpu_dark)

    # Right axis — jobs in queue: a crisp black step line.
    queue_ax = ax.twinx()
    queue_ax.plot(
        q_t, q_v, color=queue_col, lw=1.7, alpha=0.9, solid_joinstyle="round", solid_capstyle="round", zorder=3
    )
    queue_ax.set_ylim(0, max(peak_queue * 1.15, 1))
    queue_ax.set_ylabel("Jobs in queue", fontsize=fs_label, color=queue_col)
    queue_ax.tick_params(labelsize=fs_tick)
    queue_ax.tick_params(axis="y", colors=queue_col)
    # Colour the visible spines to match each series; keep the shared left spine blue.
    queue_ax.spines["left"].set_color(gpu_dark)
    queue_ax.spines["right"].set_color(queue_col)
    # jobs are integers — keep the queue axis on whole numbers (no 2.5, 5.5, ...)
    queue_ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    handles = [
        Patch(facecolor=_GPU_FILL, alpha=0.45, edgecolor=gpu_dark, lw=1.1, label="GPUs in use"),
        Line2D([0], [0], color=queue_col, lw=1.7, label="Jobs in queue"),
        Line2D([0], [0], color=cap_col, lw=1.3, ls=(0, (6, 4)), label="Cluster capacity"),
    ]
    if title:
        ax.set_title(title, fontsize=fs_title, pad=10)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        fontsize=fs_legend,
        handlelength=1.8,
        columnspacing=2.0,
        handletextpad=0.7,
    )
    # Reserve top room for the legend (and a title when present) so nothing collides.
    fig.tight_layout(rect=(0, 0, 1, 0.84 if title else 0.87))
    _savefig(fig, output_path)
    plt.close(fig)
    return {
        "jobs": n_jobs,
        "cluster_gpus": cluster_gpus,
        "makespan_h": round(makespan_h, 2),
        "peak_gpus": int(peak_gpus),
        "peak_queue": int(peak_queue),
    }
