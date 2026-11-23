"""Descriptive glucose analytics: TIR, SD, CV, GMI, rolling stats, daily summaries, AGP.

Convention: summary statistics are computed on OBSERVED readings only (imputed points are
excluded) so interpolation can never inflate coverage or bias TIR/GMI. Rolling statistics use the
gap-filled series so curves stay continuous.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.core.constants import (
    CV_STABLE_THRESHOLD_PCT,
    MIN_COVERAGE_PCT_FOR_CONSENSUS,
    MIN_DAYS_FOR_CONSENSUS,
    TARGET_HIGH,
    TARGET_LOW,
    VERY_HIGH_ABOVE,
    VERY_LOW_BELOW,
)
from app.core.errors import InsufficientDataError
from app.services.preprocessing import CleanedSeries, QualityReport, observed_only


def compute_gmi(mean_glucose_mg_dl: float) -> float:
    """Glucose Management Indicator (%) = 3.31 + 0.02392 x mean glucose (mg/dL)."""
    return 3.31 + 0.02392 * mean_glucose_mg_dl


def compute_time_in_range(values: np.ndarray) -> dict[str, float]:
    """Percent of readings per consensus band. Bands partition the line, so they sum to 100."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise InsufficientDataError("No readings to compute time-in-range.")
    n = v.size
    pct = lambda mask: round(100.0 * float(mask.sum()) / n, 2)  # noqa: E731
    very_low = pct(v < VERY_LOW_BELOW)
    low = pct((v >= VERY_LOW_BELOW) & (v < TARGET_LOW))
    in_range = pct((v >= TARGET_LOW) & (v <= TARGET_HIGH))
    high = pct((v > TARGET_HIGH) & (v <= VERY_HIGH_ABOVE))
    very_high = pct(v > VERY_HIGH_ABOVE)
    return {
        "very_low_pct": very_low,
        "low_pct": low,
        "in_range_pct": in_range,
        "high_pct": high,
        "very_high_pct": very_high,
        "tbr_pct": round(very_low + low, 2),
        "tar_pct": round(high + very_high, 2),
    }


def compute_glucose_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise InsufficientDataError("No readings to summarise.")
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    cv = 100.0 * sd / mean if mean > 0 else 0.0
    return {
        "n_readings": int(v.size),
        "mean_mg_dl": round(mean, 2),
        "median_mg_dl": round(float(np.median(v)), 2),
        "sd_mg_dl": round(sd, 2),
        "cv_pct": round(cv, 2),
        "cv_is_stable": cv <= CV_STABLE_THRESHOLD_PCT,
        "min_mg_dl": round(float(v.min()), 2),
        "max_mg_dl": round(float(v.max()), 2),
        "gmi_pct": round(compute_gmi(mean), 2),
        "time_in_range": compute_time_in_range(v),
    }


def data_sufficiency(report: QualityReport) -> tuple[bool, list[str]]:
    """Check against the consensus criteria; return (meets_criteria, caveats).

    Clipping is reported as a caveat but does not by itself fail the criteria.
    """
    warnings: list[str] = []
    meets = True
    if report.span_days < MIN_DAYS_FOR_CONSENSUS:
        meets = False
        warnings.append(
            f"Only {report.span_days:.1f} days of data (consensus recommends "
            f">= {MIN_DAYS_FOR_CONSENSUS}); TIR/GMI may not represent longer-term control."
        )
    if report.coverage_pct < MIN_COVERAGE_PCT_FOR_CONSENSUS:
        meets = False
        warnings.append(
            f"Sensor coverage is {report.coverage_pct:.1f}% (consensus recommends "
            f">= {MIN_COVERAGE_PCT_FOR_CONSENSUS:.0f}%)."
        )
    if report.n_clipped_low or report.n_clipped_high:
        warnings.append(
            f"{report.n_clipped_low + report.n_clipped_high} reading(s) were outside the "
            "physical sensor range and were clipped."
        )
    return meets, warnings


def summarize(clean: CleanedSeries) -> dict:
    obs = observed_only(clean.frame)
    stats = compute_glucose_stats(obs.to_numpy())
    meets, warnings = data_sufficiency(clean.report)
    stats["meets_consensus_criteria"] = meets
    stats["warnings"] = warnings
    return stats


def rolling_statistics(
    clean: CleanedSeries, windows_minutes: list[int], output_step_minutes: int
) -> list[dict]:
    """Trailing rolling mean/SD over the gap-filled series, down-sampled for payload size."""
    interval = clean.config.interval_minutes
    step = max(1, output_step_minutes // interval)
    s = clean.frame["glucose"]
    out = []
    for w in sorted(set(windows_minutes)):
        n = max(2, w // interval)
        roll = s.rolling(window=n, min_periods=max(2, math.ceil(0.5 * n)))
        df = pd.DataFrame({"mean": roll.mean(), "sd": roll.std(ddof=1)}).iloc[::step]
        points = [
            {
                "timestamp": ts.to_pydatetime(),
                "mean_mg_dl": None if np.isnan(r["mean"]) else round(float(r["mean"]), 2),
                "sd_mg_dl": None if np.isnan(r["sd"]) else round(float(r["sd"]), 2),
            }
            for ts, r in df.iterrows()
        ]
        out.append({"window_minutes": w, "points": points})
    return out


def daily_summaries(clean: CleanedSeries) -> list[dict]:
    """Per local-calendar-day statistics (observed readings only)."""
    tz = clean.config.timezone
    expected = 1440 / clean.config.interval_minutes
    local = observed_only(clean.frame).tz_convert(tz)
    rows = []
    for day, grp in local.groupby(local.index.date):
        st = compute_glucose_stats(grp.to_numpy())
        rows.append(
            {
                "date": day,
                "coverage_pct": round(min(100.0, 100.0 * len(grp) / expected), 2),
                "n_readings": st["n_readings"],
                "mean_mg_dl": st["mean_mg_dl"],
                "sd_mg_dl": st["sd_mg_dl"],
                "cv_pct": st["cv_pct"],
                "gmi_pct": st["gmi_pct"],
                "min_mg_dl": st["min_mg_dl"],
                "max_mg_dl": st["max_mg_dl"],
                "time_in_range": st["time_in_range"],
            }
        )
    return rows


AGP_PERCENTILES = (5, 25, 50, 75, 95)


def ambulatory_glucose_profile(clean: CleanedSeries) -> list[dict]:
    """Hour-of-day percentile curve overlaying all days (the standard 'AGP' view)."""
    local = observed_only(clean.frame).tz_convert(clean.config.timezone)
    q = local.groupby(local.index.hour).quantile([p / 100 for p in AGP_PERCENTILES]).unstack()
    out = []
    for hour, row in q.iterrows():
        out.append(
            {
                "hour": int(hour),
                **{f"p{p}": round(float(row[p / 100]), 2) for p in AGP_PERCENTILES},
            }
        )
    return out
