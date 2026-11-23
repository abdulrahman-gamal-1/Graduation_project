"""CGM data cleaning pipeline.

Pure pandas/numpy - no web-framework imports, so it is unit-testable and reusable by the
training/batch jobs later.

Pipeline order matters:
  1. parse + validate timestamps, coerce values
  2. invalidate impossible values (NaN/inf/<=0)
  3. de-duplicate identical timestamps (mean)
  4. clip to the physical sensor range (38-400 mg/dL)
  5. resample onto a regular grid (default 5 min, bin mean)
  6. impute ONLY short gaps (time interpolation); long gaps stay missing
  7. build a data-quality report (coverage, gaps, what we changed)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.constants import PHYSICAL_MAX_MG_DL, PHYSICAL_MIN_MG_DL
from app.core.errors import InsufficientDataError


@dataclass(frozen=True)
class PreprocessConfig:
    interval_minutes: int = 5
    clip_low: float = PHYSICAL_MIN_MG_DL
    clip_high: float = PHYSICAL_MAX_MG_DL
    max_interpolation_gap_minutes: int = 30
    timezone: str = "UTC"


@dataclass
class QualityReport:
    n_raw: int
    n_invalid_timestamp: int
    n_invalid_value: int
    n_duplicate_timestamps: int
    n_clipped_low: int
    n_clipped_high: int
    n_grid_points: int
    n_observed: int
    n_imputed: int
    n_missing_remaining: int
    coverage_pct: float
    n_gaps_remaining: int
    longest_gap_minutes: int
    start: datetime
    end: datetime
    span_days: float


@dataclass
class CleanedSeries:
    """`frame` has a UTC DatetimeIndex on a regular grid and columns:
    glucose (float, NaN where missing) and is_imputed (bool)."""

    frame: pd.DataFrame
    report: QualityReport
    config: PreprocessConfig


def readings_to_frame(
    readings: list[tuple[datetime | str, float | None]],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [r[0] for r in readings],
            "glucose": [np.nan if r[1] is None else r[1] for r in readings],
        }
    )


def _gap_runs(is_missing: pd.Series) -> pd.Series:
    """Length of the missing-run each point belongs to (0 for non-missing points)."""
    run_id = (is_missing != is_missing.shift(fill_value=False)).cumsum()
    return is_missing.groupby(run_id).transform("sum").astype(int)


def clean_cgm_series(raw: pd.DataFrame, cfg: PreprocessConfig | None = None) -> CleanedSeries:
    cfg = cfg or PreprocessConfig()
    if cfg.clip_low >= cfg.clip_high:
        raise ValueError("clip_low must be smaller than clip_high")
    n_raw = len(raw)

    df = raw[["timestamp", "glucose"]].copy()
    # Naive timestamps are interpreted as UTC (the Node backend should always send UTC/ISO-8601).
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["glucose"] = pd.to_numeric(df["glucose"], errors="coerce")

    n_bad_ts = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    # Values that cannot be glucose readings: NaN (missing), +-inf, zero/negative.
    invalid = ~np.isfinite(df["glucose"]) | (df["glucose"] <= 0)
    n_bad_val = int(invalid.sum())
    df.loc[invalid, "glucose"] = np.nan

    n_dups = int(df["timestamp"].duplicated().sum())
    df = df.groupby("timestamp", sort=True)["glucose"].mean().reset_index()

    n_clip_low = int((df["glucose"] < cfg.clip_low).sum())
    n_clip_high = int((df["glucose"] > cfg.clip_high).sum())
    df["glucose"] = df["glucose"].clip(cfg.clip_low, cfg.clip_high)

    series = df.set_index("timestamp")["glucose"].dropna()
    if series.empty:
        raise InsufficientDataError("No valid glucose readings after cleaning.")

    freq = f"{cfg.interval_minutes}min"
    grid = series.resample(freq).mean()  # empty bins -> NaN, covers the full time span

    # Impute only gaps no longer than the configured limit; leave long gaps untouched.
    max_gap_points = cfg.max_interpolation_gap_minutes // cfg.interval_minutes
    missing = grid.isna()
    fillable = missing & (_gap_runs(missing) <= max_gap_points)
    interpolated = grid.interpolate(method="time", limit_area="inside")
    filled = grid.where(~fillable, interpolated)
    is_imputed = fillable & filled.notna()

    still_missing = filled.isna()
    runs = _gap_runs(still_missing)
    n_gaps = int(((still_missing) & (still_missing != still_missing.shift(fill_value=False))).sum())
    longest_gap = int(runs.max()) * cfg.interval_minutes if still_missing.any() else 0

    n_obs = int(grid.notna().sum())
    report = QualityReport(
        n_raw=n_raw,
        n_invalid_timestamp=n_bad_ts,
        n_invalid_value=n_bad_val,
        n_duplicate_timestamps=n_dups,
        n_clipped_low=n_clip_low,
        n_clipped_high=n_clip_high,
        n_grid_points=len(grid),
        n_observed=n_obs,
        n_imputed=int(is_imputed.sum()),
        n_missing_remaining=int(still_missing.sum()),
        coverage_pct=round(100.0 * n_obs / len(grid), 2),
        n_gaps_remaining=n_gaps,
        longest_gap_minutes=longest_gap,
        start=grid.index[0].to_pydatetime(),
        end=grid.index[-1].to_pydatetime(),
        span_days=round((len(grid) * cfg.interval_minutes) / 1440.0, 3),
    )
    frame = pd.DataFrame({"glucose": filled, "is_imputed": is_imputed})
    frame.index.name = "timestamp"
    return CleanedSeries(frame=frame, report=report, config=cfg)


def observed_only(frame: pd.DataFrame) -> pd.Series:
    """Real sensor readings only (imputed points removed, NaNs dropped)."""
    return frame["glucose"].where(~frame["is_imputed"]).dropna()
