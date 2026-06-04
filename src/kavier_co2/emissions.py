"""Carbon-emission accounting for power timelines.

A *carbon trace* is a step function over time. Each row covers a window
``[timestamp, timestamp + step)`` and supplies a constant carbon intensity in
gCO2/kWh for that window — exactly the shape OpenDC consumes when it converts a
power source's energy to carbon. A *power fragment* ``(start_time, duration_s,
power_w)`` is the unit produced by Kavier's training simulation and by OpenDC's
per-step output.

Join semantics (window split + conservative down-estimation)
------------------------------------------------------------
Energy is the integral of power over time, so a fragment that straddles a
window boundary cannot use a single intensity. We split each fragment at the
30-min window boundaries and bill every sub-interval's energy
(``power_w * sub_duration_s``).

Billing rule (DOWN-ESTIMATION): each split piece is billed at
``min(intensity of its own window, intensity of the NEXT window)``. A moment
that sits *between* two trace points (e.g. 15:34 between points at 15:00 and
16:00) is deliberately under-estimated to ``min(intensity@15:00,
intensity@16:00)``. The final trace window has no successor, so it bills at its
own value. The per-window breakdown records the down-estimated intensity per
window; the total is their sum. The reported "average intensity" is
energy-weighted (total gCO2 / total kWh).

Deviation from OpenDC
---------------------
OpenDC's ``CarbonModel`` (``opendc-simulator-compute``) treats the carbon trace
as a pure *left-continuous step function*: the loader's ``fixReportTimes`` sets
each fragment's window to ``[t_i, t_{i+1})`` and the model holds the EARLIER
point's intensity until the next point — no interpolation, no min. So OpenDC
bills the 15:34 moment at ``intensity@15:00`` (the left bound), whereas Kavier
bills it at ``min(intensity@15:00, intensity@16:00)``. Kavier's total is
therefore always <= OpenDC's left-step total. This is an intentional
conservative (lower-bound) carbon estimate, not OpenDC's exact accounting.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

WS_PER_KWH = 3.6e6  # watt-seconds in one kilowatt-hour


@dataclass(frozen=True)
class Fragment:
    """A constant-power interval. ``power_w`` is aggregate (all GPUs)."""

    start_time: pd.Timestamp
    duration_s: float
    power_w: float


@dataclass(frozen=True)
class CarbonTrace:
    """A step-function carbon-intensity trace, sorted by timestamp.

    ``timestamps[i]`` opens a window of width ``step`` carrying
    ``intensities[i]`` gCO2/kWh. Coverage is ``[timestamps[0], last + step)``.
    """

    timestamps: pd.DatetimeIndex
    intensities: "pd.Series"
    step: dt.timedelta

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        step: dt.timedelta | None = None,
    ) -> "CarbonTrace":
        if "timestamp" not in df.columns or "carbon_intensity" not in df.columns:
            raise ValueError(
                "carbon trace must have columns ['timestamp', 'carbon_intensity']; "
                f"got {list(df.columns)}"
            )
        df = df[["timestamp", "carbon_intensity"]].sort_values("timestamp").reset_index(drop=True)
        ts = pd.DatetimeIndex(df["timestamp"])
        if ts.tz is not None:
            raise ValueError("carbon trace timestamps must be timezone-naive")
        if len(ts) < 1:
            raise ValueError("carbon trace is empty")
        if step is None:
            if len(ts) < 2:
                raise ValueError("cannot infer step from a single-row trace; pass step explicitly")
            step = (ts[1] - ts[0]).to_pytimedelta()
        return cls(timestamps=ts, intensities=df["carbon_intensity"].reset_index(drop=True), step=step)

    @property
    def coverage_start(self) -> pd.Timestamp:
        return self.timestamps[0]

    @property
    def coverage_end(self) -> pd.Timestamp:
        """Exclusive upper bound: the end of the last window."""
        return self.timestamps[-1] + self.step

    def _coverage_msg(self) -> str:
        return f"trace covers [{self.coverage_start} .. {self.coverage_end}) (step {self.step})"


@dataclass(frozen=True)
class EmissionResult:
    total_energy_kwh: float
    total_co2_g: float
    breakdown: List[dict]

    @property
    def total_co2_kg(self) -> float:
        return self.total_co2_g / 1000.0

    @property
    def average_intensity(self) -> float:
        """Energy-weighted mean intensity (gCO2/kWh); 0 if no energy."""
        if self.total_energy_kwh == 0:
            return 0.0
        return self.total_co2_g / self.total_energy_kwh


def load_carbon_trace(path: str, step_minutes: int | None = None) -> CarbonTrace:
    """Load a carbon-intensity parquet into a :class:`CarbonTrace`."""
    df = pd.read_parquet(path)
    step = dt.timedelta(minutes=step_minutes) if step_minutes else None
    return CarbonTrace.from_dataframe(df, step=step)


def _window_index_for(ts: pd.Timestamp, trace: CarbonTrace) -> int:
    """Return the index of the window covering ``ts``.

    ``searchsorted(..., side='right') - 1`` finds the last window whose start is
    ``<= ts``. Callers guarantee ``ts`` is within coverage.
    """
    return int(trace.timestamps.searchsorted(ts, side="right")) - 1


def compute_emissions(fragments: Iterable[Fragment], trace: CarbonTrace) -> EmissionResult:
    """Join power fragments against the carbon trace and total the emissions.

    Each fragment is split at window boundaries; every sub-interval's energy is
    billed at its window's intensity. Fragments touching time outside the trace
    coverage raise ``ValueError`` naming the coverage.
    """
    # window_start (Timestamp) -> {"carbon_intensity", "energy_kwh", "co2_g"}
    acc: dict[pd.Timestamp, dict] = {}
    total_energy_kwh = 0.0
    total_co2_g = 0.0

    cov_start = trace.coverage_start
    cov_end = trace.coverage_end

    for frag in fragments:
        start = frag.start_time
        if start.tz is not None:
            raise ValueError(
                f"fragment timestamp {start} is timezone-aware; the carbon trace is "
                "timezone-naive. Use naive timestamps for consistency."
            )
        if frag.duration_s < 0:
            raise ValueError(f"fragment duration must be >= 0, got {frag.duration_s}")
        end = start + pd.Timedelta(seconds=frag.duration_s)

        if start < cov_start or end > cov_end:
            raise ValueError(
                f"fragment time [{start} .. {end}] is outside the carbon trace coverage: "
                f"{trace._coverage_msg()}"
            )

        if frag.duration_s == 0:
            continue

        # Walk the fragment window by window, slicing at each boundary.
        cursor = start
        last_wi = len(trace.timestamps) - 1
        while cursor < end:
            wi = _window_index_for(cursor, trace)
            window_start = trace.timestamps[wi]
            window_end = window_start + trace.step
            seg_end = min(end, window_end)
            seg_seconds = (seg_end - cursor).total_seconds()
            energy_kwh = frag.power_w * seg_seconds / WS_PER_KWH
            # DOWN-ESTIMATION: bill this piece at the lower of its own window's
            # intensity and the next window's intensity. The final trace window
            # has no successor, so it uses its own value.
            own = float(trace.intensities.iloc[wi])
            if wi < last_wi:
                intensity = min(own, float(trace.intensities.iloc[wi + 1]))
            else:
                intensity = own
            co2_g = energy_kwh * intensity

            bucket = acc.setdefault(
                window_start,
                {"carbon_intensity": intensity, "energy_kwh": 0.0, "co2_g": 0.0},
            )
            bucket["energy_kwh"] += energy_kwh
            bucket["co2_g"] += co2_g
            total_energy_kwh += energy_kwh
            total_co2_g += co2_g

            cursor = seg_end

    breakdown = [
        {"window_start": ws, **vals} for ws, vals in sorted(acc.items(), key=lambda kv: kv[0])
    ]
    return EmissionResult(
        total_energy_kwh=total_energy_kwh,
        total_co2_g=total_co2_g,
        breakdown=breakdown,
    )
