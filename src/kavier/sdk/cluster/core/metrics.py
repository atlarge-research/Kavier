"""Timeline step series and per-node activity for the cluster simulator (stdlib-only)."""

from __future__ import annotations


def cumulative_steps(events: list[tuple[float, float]], t_end: float) -> tuple[list[float], list[float]]:
    """Return the step line ``(times, values)`` of ``(time, change)`` events.

    Events at the same instant are netted, so a zero-net instant such as a job starting at its submit
    time adds no dip. The line starts at ``(0, 0)`` and ends at ``(t_end, final)``.
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
    """Return aligned ``(times, gpus_in_use, queue_depth)`` step series on one time axis.

    Both series are netted per instant and stepped together, so the three lists have equal length.
    Instants where neither series changes are skipped. The series start at ``t=0`` and end at ``t_end``.
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


def node_activity(intervals: list[tuple[float, float, int]], t0: float, t_end: float) -> tuple[int, float]:
    """Return peak concurrent GPUs and idle wall-seconds for one node over ``[t0, t_end]``.

    ``intervals`` are ``(start_s, end_s, gpus_on_node)`` of the jobs placed on this node. Idle time is
    the wall time with zero GPUs in use; a node with no intervals is idle for the whole window.
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
