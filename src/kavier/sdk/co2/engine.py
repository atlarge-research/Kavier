"""Carbon model: power ``Fragment``s integrated over a ``CarbonTrace`` (gCO2/kWh) -> ``EmissionResult`` (kWh, gCO2)."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Iterable, List

import pandas as pd

from kavier.sdk.units import WS_PER_KWH


@dataclass(frozen=True)
class Fragment:
    """A constant-power interval (naive ``start_time``, ``duration_s`` seconds at ``power_w`` watts)."""

    start_time: pd.Timestamp
    duration_s: float
    power_w: float


@dataclass(frozen=True)
class CarbonTrace:
    """Piecewise-constant timeline: ``intensities[i]`` (gCO2/kWh) applies over half-open ``[timestamps[i], +step)``."""

    timestamps: pd.DatetimeIndex
    intensities: "pd.Series"
    step: dt.timedelta

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        step: dt.timedelta | None = None,
    ) -> "CarbonTrace":
        """Build from a ``[timestamp, carbon_intensity]`` frame; ``step`` inferred from the first interval if unset."""
        if "timestamp" not in df.columns or "carbon_intensity" not in df.columns:
            raise ValueError(
                f"carbon trace must have columns ['timestamp', 'carbon_intensity']; got {list(df.columns)}"
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
        """Timestamp of the first intensity window."""
        return self.timestamps[0]

    @property
    def coverage_end(self) -> pd.Timestamp:
        """Exclusive end of the last intensity window (``last_ts + step``)."""
        return self.timestamps[-1] + self.step

    def _coverage_msg(self) -> str:
        return f"trace covers [{self.coverage_start} .. {self.coverage_end}) (step {self.step})"


@dataclass(frozen=True)
class EmissionResult:
    """Totals (energy kWh, CO2 g) plus per-window ``breakdown`` (window_start, carbon_intensity, energy_kwh, co2_g)."""

    total_energy_kwh: float
    total_co2_g: float
    breakdown: List[dict[str, Any]]

    @property
    def total_co2_kg(self) -> float:
        """Total emissions in kg (``total_co2_g / 1000``)."""
        return self.total_co2_g / 1000.0

    @property
    def average_intensity(self) -> float:
        """Energy-weighted mean intensity (gCO2/kWh); 0 when no energy was used."""
        if self.total_energy_kwh == 0:
            return 0.0
        return self.total_co2_g / self.total_energy_kwh


def load_carbon_trace(path: str, step_minutes: int | None = None) -> CarbonTrace:
    """Load a carbon-intensity parquet into a CarbonTrace; ``step_minutes`` overrides the inferred step."""
    df = pd.read_parquet(path)
    step = dt.timedelta(minutes=step_minutes) if step_minutes else None
    return CarbonTrace.from_dataframe(df, step=step)


def _window_index_for(ts: pd.Timestamp, trace: CarbonTrace) -> int:
    return int(trace.timestamps.searchsorted(ts, side="right")) - 1


def compute_emissions(fragments: Iterable[Fragment], trace: CarbonTrace) -> EmissionResult:
    """Integrate each fragment's energy over the trace (down-estimated intensity); raises if outside coverage."""
    acc: dict[pd.Timestamp, dict[str, float]] = {}
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
                f"fragment time [{start} .. {end}] is outside the carbon trace coverage: {trace._coverage_msg()}"
            )

        if frag.duration_s == 0:
            continue

        cursor = start
        last_wi = len(trace.timestamps) - 1
        while cursor < end:
            wi = _window_index_for(cursor, trace)
            window_start = trace.timestamps[wi]
            # Window boundaries are the trace's own timestamps: a non-final window runs to the NEXT
            # timestamp, absorbing any gap wider than the step inferred from the first interval. Using
            # window_start + step here would place window_end at/behind the cursor inside such a gap
            # (seg_end <= cursor), stalling the loop forever. Only the final window has no successor,
            # so it spans exactly one step (which also defines coverage_end).
            window_end = trace.timestamps[wi + 1] if wi < last_wi else window_start + trace.step
            seg_end = min(end, window_end)
            seg_seconds = (seg_end - cursor).total_seconds()
            energy_kwh = frag.power_w * seg_seconds / WS_PER_KWH
            # DOWN-ESTIMATION: bill at min(own window, next window) intensity; the final window has no successor.
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

    breakdown = [{"window_start": ws, **vals} for ws, vals in sorted(acc.items(), key=lambda kv: kv[0])]
    return EmissionResult(
        total_energy_kwh=total_energy_kwh,
        total_co2_g=total_co2_g,
        breakdown=breakdown,
    )
