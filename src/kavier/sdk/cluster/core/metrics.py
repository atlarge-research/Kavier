"""Timeline step-series and per-node activity helpers for the cluster simulator (stdlib-only).

``cumulative_steps`` builds a single staircase from events; ``build_timeline`` builds the aligned
GPUs-in-use and queue-depth staircases the timeline figure draws (on one shared time axis).
``node_activity`` reduces one node's busy intervals to peak concurrent GPUs and idle wall-time.
"""

from __future__ import annotations


def cumulative_steps(events: list[tuple[float, float]], t_end: float) -> tuple[list[float], list[float]]:
    """Turn ``(time, change)`` events into a step line ``(times, values)``.

    Same-instant events are netted so a zero-net instant (e.g. a job that starts the moment it is
    submitted) never makes the line dip. Anchored at ``(0, 0)`` and closed at ``(t_end, final)``.
    """
    net: dict[float, float] = {}
    for time, change in events:
        net[time] = net.get(time, 0.0) + change
    times: list[float] = [0.0]
    values: list[float] = [0.0]
    running = 0.0
    for time in sorted(net):
        change = net[time]
        if change == 0:
            continue
        times.append(time)
        values.append(running)
        running += change
        times.append(time)
        values.append(running)
    times.append(t_end)
    values.append(running)
    return times, values


def build_timeline(
    gpu_events: list[tuple[float, float]],
    queue_events: list[tuple[float, float]],
    t_end: float,
) -> tuple[list[float], list[float], list[float]]:
    """Aligned ``(times, gpus_in_use, queue_depth)`` staircases on one shared time axis.

    Both series are netted per instant and stepped together, so the three returned lists have equal
    length. Anchored at ``t=0`` and closed at ``t_end``; instants where neither series changes are
    skipped.
    """
    net_gpu: dict[float, float] = {}
    net_queue: dict[float, float] = {}
    for time, change in gpu_events:
        net_gpu[time] = net_gpu.get(time, 0.0) + change
    for time, change in queue_events:
        net_queue[time] = net_queue.get(time, 0.0) + change

    times: list[float] = [0.0]
    gpus: list[float] = [0.0]
    queue: list[float] = [0.0]
    run_gpu = 0.0
    run_queue = 0.0
    for time in sorted(set(net_gpu) | set(net_queue)):
        d_gpu = net_gpu.get(time, 0.0)
        d_queue = net_queue.get(time, 0.0)
        if d_gpu == 0 and d_queue == 0:
            continue
        times.append(time)
        gpus.append(run_gpu)
        queue.append(run_queue)
        run_gpu += d_gpu
        run_queue += d_queue
        times.append(time)
        gpus.append(run_gpu)
        queue.append(run_queue)
    times.append(t_end)
    gpus.append(run_gpu)
    queue.append(run_queue)
    return times, gpus, queue


def node_activity(
    intervals: list[tuple[float, float, int]], t0: float, t_end: float
) -> tuple[int, float]:
    """Peak concurrent GPUs and idle wall-seconds for one node over ``[t0, t_end]``.

    ``intervals`` are ``(start_s, end_s, gpus_on_node)`` for the jobs that placed GPUs on this node.
    ``idle_s`` is the wall-clock time in the window during which the node had zero GPUs in use. A
    node with no intervals is idle for the whole window.
    """
    if not intervals:
        return 0, max(0.0, t_end - t0)
    events: list[tuple[float, int]] = []
    for start, end, gpus in intervals:
        events.append((start, gpus))
        events.append((end, -gpus))
    events.sort()
    peak = 0
    current = 0
    busy_wall = 0.0
    prev_t = t0
    for time, delta in events:
        if current > 0:
            busy_wall += time - prev_t
        current += delta
        if current > peak:
            peak = current
        prev_t = time
    idle = (t_end - t0) - busy_wall
    return peak, idle if idle > 0.0 else 0.0
