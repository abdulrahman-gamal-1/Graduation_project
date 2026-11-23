"""User-level feature extraction.

Produces a FLAT, stable-named feature dict. The same names feed the early-risk classifier
(next module), so treat renames as breaking changes and bump FEATURE_SCHEMA_VERSION.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.constants import (
    EVENT_MIN_DURATION_MINUTES,
    TARGET_HIGH,
    TARGET_LOW,
    TIME_BLOCKS,
    VERY_HIGH_ABOVE,
    VERY_LOW_BELOW,
)
from app.services.analytics import compute_glucose_stats
from app.services.preprocessing import CleanedSeries, observed_only

MIN_READINGS_PER_BLOCK = 12  # ~1 hour of 5-min data


def count_excursion_events(
    glucose: pd.Series, threshold: float, below: bool, interval_minutes: int,
    min_minutes: int = EVENT_MIN_DURATION_MINUTES,
) -> int:
    """Count runs of consecutive grid points beyond `threshold` lasting >= min_minutes.

    Simplification of the consensus definition: a missing value ends a run (we never assume
    what happened inside a gap) and we do not require a 15-min recovery to split two events.
    """
    mask = (glucose < threshold) if below else (glucose > threshold)  # NaN -> False
    run_id = (mask != mask.shift(fill_value=False)).cumsum()
    lengths = mask.groupby(run_id).sum()  # True-runs: their length; False-runs: 0
    return int(((lengths * interval_minutes) >= min_minutes).sum())


def extract_user_features(clean: CleanedSeries) -> dict[str, float | int | None]:
    cfg, frame = clean.config, clean.frame
    interval = cfg.interval_minutes
    obs = observed_only(frame)
    st = compute_glucose_stats(obs.to_numpy())
    tir = st["time_in_range"]

    f: dict[str, float | int | None] = {
        "n_days": round(clean.report.span_days, 2),
        "coverage_pct": clean.report.coverage_pct,
        "mean_mg_dl": st["mean_mg_dl"],
        "median_mg_dl": st["median_mg_dl"],
        "sd_mg_dl": st["sd_mg_dl"],
        "cv_pct": st["cv_pct"],
        "min_mg_dl": st["min_mg_dl"],
        "max_mg_dl": st["max_mg_dl"],
        "gmi_pct": st["gmi_pct"],
        "tir_pct": tir["in_range_pct"],
        "tbr_pct": tir["tbr_pct"],
        "tbr_level2_pct": tir["very_low_pct"],
        "tar_pct": tir["tar_pct"],
        "tar_level2_pct": tir["very_high_pct"],
    }

    # Excursion events use the gap-filled series (short gaps bridged, long gaps break runs).
    g = frame["glucose"]
    f["hypo_events_lt70"] = count_excursion_events(g, TARGET_LOW, True, interval)
    f["hypo_events_lt54"] = count_excursion_events(g, VERY_LOW_BELOW, True, interval)
    f["hyper_events_gt180"] = count_excursion_events(g, TARGET_HIGH, False, interval)
    f["hyper_events_gt250"] = count_excursion_events(g, VERY_HIGH_ABOVE, False, interval)
    days = max(clean.report.span_days, 1e-9)
    f["hypo_events_per_week"] = round(f["hypo_events_lt70"] * 7.0 / days, 2)

    # Time-of-day blocks in the user's local time.
    local = obs.tz_convert(cfg.timezone)
    hours = local.index.hour
    for name, (a, b) in TIME_BLOCKS.items():
        v = local[(hours >= a) & (hours < b)].to_numpy()
        if v.size >= MIN_READINGS_PER_BLOCK:
            bs = compute_glucose_stats(v)
            f[f"block_{name}_mean"] = bs["mean_mg_dl"]
            f[f"block_{name}_cv_pct"] = bs["cv_pct"]
            f[f"block_{name}_tbr_pct"] = bs["time_in_range"]["tbr_pct"]
            f[f"block_{name}_tar_pct"] = bs["time_in_range"]["tar_pct"]
        else:
            for k in ("mean", "cv_pct", "tbr_pct", "tar_pct"):
                f[f"block_{name}_{k}"] = None

    # Day-to-day variability and trend (days with >= 50% coverage only).
    expected = 1440 / interval
    by_day = local.groupby(local.index.date)
    day_means = by_day.mean()[by_day.count() >= 0.5 * expected]
    f["n_valid_days"] = int(len(day_means))
    if len(day_means) >= 2:
        f["daily_mean_sd_mg_dl"] = round(float(day_means.std(ddof=1)), 2)
    else:
        f["daily_mean_sd_mg_dl"] = None
    if len(day_means) >= 3:
        slope = np.polyfit(np.arange(len(day_means)), day_means.to_numpy(), 1)[0]
        f["trend_mg_dl_per_day"] = round(float(slope), 3)
    else:
        f["trend_mg_dl_per_day"] = None
    return f


def build_daily_profiles(clean: CleanedSeries, min_hours_present: int = 18) -> pd.DataFrame:
    """Days x 24 matrix of hourly mean glucose (local time) - input for K-Means (next module).

    Days with fewer than `min_hours_present` populated hours are dropped; remaining holes are
    linearly interpolated across the day.
    """
    local = clean.frame["glucose"].tz_convert(clean.config.timezone).dropna()
    hourly = local.groupby([local.index.date, local.index.hour]).mean().unstack()
    hourly = hourly.reindex(columns=range(24))
    hourly = hourly[hourly.notna().sum(axis=1) >= min_hours_present]
    hourly = hourly.interpolate(axis=1, limit_direction="both")
    hourly.index.name = "date"
    return hourly
