"""Rule-based recurring-pattern detection + facade combining rules and K-Means.

Each rule evaluates every local calendar day independently, then asks "does this happen on a
meaningful share of the days we could evaluate?". A pattern is reported only when
  (a) enough days were evaluable (window coverage >= 70%), and
  (b) the behaviour recurs (thresholds in PatternConfig).
Every pattern carries its evidence (days affected / evaluated, example dates, metrics) so the
UI and the later risk model can show *why* it fired.

Language is descriptive ("glucose rose ... on 9 of 14 days"), never diagnostic and never
suggests treatment changes. Meal windows are a time-of-day PROXY (no meal logs here).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from app.core.constants import (
    CV_STABLE_THRESHOLD_PCT, TARGET_HIGH, TARGET_LOW, TIME_BLOCKS, VERY_HIGH_ABOVE,
    VERY_LOW_BELOW,
)
from app.services.clustering import ClusteringResult, cluster_daily_profiles
from app.services.features import build_daily_profiles, count_excursion_events
from app.services.preprocessing import CleanedSeries

SEVERITY_ORDER = {"priority": 0, "attention": 1, "info": 2}
_NOTE = " This is an observed pattern in the data, not a diagnosis; consider reviewing it with your care team."


@dataclass(frozen=True)
class PatternConfig:
    min_days_evaluated: int = 5
    min_window_coverage: float = 0.7
    hypo_min_days: int = 2
    hypo_min_fraction: float = 0.15
    recurrence_fraction: float = 0.5
    dawn_rise_mg_dl: float = 20.0
    postmeal_rise_mg_dl: float = 50.0
    variability_day_fraction: float = 0.4
    meal_windows: dict = field(default_factory=lambda: {
        "breakfast": (6, 10), "lunch": (11, 15), "dinner": (17, 21)})


@dataclass
class Pattern:
    id: str
    category: str            # hypoglycemia | hyperglycemia | dawn_rise | postmeal_spike | variability
    label: str
    severity: str            # priority | attention | info
    window: str              # local time window, e.g. "00:00-06:00"
    description: str
    n_days_affected: int
    n_days_evaluated: int
    fraction: float
    example_dates: list[date]
    metrics: dict[str, float]


def _win(a: float, b: float) -> str:
    f = lambda h: f"{int(h) % 24:02d}:{int(round((h % 1) * 60)):02d}"  # noqa: E731
    return f"{f(a)}-{f(b)}"


def _ev(n_aff: int, n_eval: int) -> str:
    return f"{n_aff} of {n_eval} evaluated days ({100 * n_aff / n_eval:.0f}%)"


class _DayWindows:
    """Per-local-day slices of the gap-filled series for a given hour-of-day window."""

    def __init__(self, clean: CleanedSeries, min_cov: float):
        self.interval = clean.config.interval_minutes
        self.s = clean.frame["glucose"].tz_convert(clean.config.timezone)
        self.h = (self.s.index.hour + self.s.index.minute / 60).to_numpy()
        self.min_cov = min_cov

    def window(self, a: float, b: float) -> dict[date, pd.Series]:
        w = self.s[(self.h >= a) & (self.h < b)]
        need = self.min_cov * (b - a) * 60 / self.interval
        return {d: g for d, g in w.groupby(w.index.date) if g.notna().sum() >= need}


def _make(id_, cat, label, sev, window, desc, affected, n_eval, metrics) -> Pattern:
    return Pattern(
        id=id_, category=cat, label=label, severity=sev, window=window,
        description=desc + (_NOTE if sev != "info" else ""),
        n_days_affected=len(affected), n_days_evaluated=n_eval,
        fraction=round(len(affected) / n_eval, 3),
        example_dates=sorted(affected)[-5:],
        metrics={k: round(float(v), 1) for k, v in metrics.items()},
    )


def detect_rule_patterns(clean: CleanedSeries, cfg: PatternConfig | None = None) -> list[Pattern]:
    cfg = cfg or PatternConfig()
    dw = _DayWindows(clean, cfg.min_window_coverage)
    iv = dw.interval
    out: list[Pattern] = []

    # 1) Recurrent hypoglycaemia per time-of-day block ------------------------------------
    for name, (a, b) in TIME_BLOCKS.items():
        days = dw.window(a, b)
        if len(days) < cfg.min_days_evaluated:
            continue
        hypo = {d for d, g in days.items() if count_excursion_events(g, TARGET_LOW, True, iv) > 0}
        l2 = {d for d in hypo if count_excursion_events(days[d], VERY_LOW_BELOW, True, iv) > 0}
        if len(hypo) >= cfg.hypo_min_days and len(hypo) / len(days) >= cfg.hypo_min_fraction:
            lowest = min(float(days[d].min()) for d in hypo)
            mins = np.mean([(days[d] < TARGET_LOW).sum() * iv for d in hypo])
            desc = (f"On {_ev(len(hypo), len(days))}, glucose stayed below 70 mg/dL for at least "
                    f"15 minutes during the {name} window ({_win(a, b)}).")
            if l2:
                desc += f" On {len(l2)} of those days it dropped below 54 mg/dL."
            out.append(_make(
                f"hypo_{name}", "hypoglycemia", f"Recurrent low glucose ({name})",
                "priority" if l2 else "attention", _win(a, b), desc, hypo, len(days),
                {"lowest_mg_dl": lowest, "mean_minutes_below_70_on_affected_days": mins,
                 "days_below_54": len(l2)}))

    # 2) Early-morning rise (pre-dawn nadir -> fasting window) ----------------------------
    nadir, fast = dw.window(2, 4), dw.window(6, 8)
    common = sorted(nadir.keys() & fast.keys())
    if len(common) >= cfg.min_days_evaluated:
        rises = {d: float(fast[d].mean() - nadir[d].mean()) for d in common}
        aff = {d for d, r in rises.items() if r >= cfg.dawn_rise_mg_dl}
        if len(aff) / len(common) >= cfg.recurrence_fraction:
            fasting = float(np.mean([fast[d].mean() for d in common]))
            out.append(_make(
                "dawn_rise", "dawn_rise", "Recurring early-morning glucose rise",
                "attention" if fasting > TARGET_HIGH else "info", "02:00-08:00",
                f"On {_ev(len(aff), len(common))}, average glucose at 06:00-08:00 was at least "
                f"{cfg.dawn_rise_mg_dl:.0f} mg/dL higher than at 02:00-04:00.",
                aff, len(common),
                {"median_rise_mg_dl": np.median(list(rises.values())),
                 "mean_fasting_window_mg_dl": fasting}))

    # 3) Post-meal-window spikes (time-of-day proxy for meals) ----------------------------
    for meal, (a, b) in cfg.meal_windows.items():
        wins, pre = dw.window(a, b), dw.window(a - 0.5, a)
        common = sorted(wins.keys() & pre.keys())
        if len(common) < cfg.min_days_evaluated:
            continue
        info = {}
        for d in common:
            peak = float(wins[d].max())
            info[d] = (peak, peak - float(pre[d].mean()), wins[d].idxmax().hour
                       + wins[d].idxmax().minute / 60)
        aff = {d for d, (pk, rise, _) in info.items()
               if pk > TARGET_HIGH and rise >= cfg.postmeal_rise_mg_dl}
        if len(aff) / len(common) >= cfg.recurrence_fraction:
            pk = np.median([info[d][0] for d in aff])
            out.append(_make(
                f"postmeal_{meal}", "postmeal_spike", f"Recurring glucose spike ({meal} window)",
                "attention" if pk > VERY_HIGH_ABOVE else "info", _win(a, b),
                f"On {_ev(len(aff), len(common))}, glucose peaked above 180 mg/dL and rose at "
                f"least {cfg.postmeal_rise_mg_dl:.0f} mg/dL from the preceding 30 minutes during "
                f"the {meal} window ({_win(a, b)}). Windows are fixed clock times, not logged meals.",
                aff, len(common),
                {"median_peak_mg_dl": pk,
                 "median_rise_mg_dl": np.median([info[d][1] for d in aff]),
                 "median_peak_hour": np.median([info[d][2] for d in aff])}))

    # 4) Sustained hyperglycaemia per block (>= half of the window above 180) ------------
    for name, (a, b) in TIME_BLOCKS.items():
        days = dw.window(a, b)
        if len(days) < cfg.min_days_evaluated:
            continue
        aff = {d for d, g in days.items() if (g.dropna() > TARGET_HIGH).mean() >= 0.5}
        if len(aff) / len(days) >= cfg.recurrence_fraction:
            mean_aff = float(np.mean([days[d].mean() for d in aff]))
            out.append(_make(
                f"hyper_{name}", "hyperglycemia", f"Recurrent sustained high glucose ({name})",
                "priority" if mean_aff > VERY_HIGH_ABOVE else "attention", _win(a, b),
                f"On {_ev(len(aff), len(days))}, glucose was above 180 mg/dL for at least half "
                f"of the {name} window ({_win(a, b)}).",
                aff, len(days), {"mean_glucose_on_affected_days_mg_dl": mean_aff}))

    # 5) Highly variable days (daily CV above 36%) ---------------------------------------
    days = dw.window(0, 24)
    if len(days) >= cfg.min_days_evaluated:
        cvs = {d: 100 * float(g.std(ddof=1) / g.mean()) for d, g in days.items()}
        aff = {d for d, c in cvs.items() if c > CV_STABLE_THRESHOLD_PCT}
        if len(aff) / len(days) >= cfg.variability_day_fraction:
            out.append(_make(
                "high_variability_days", "variability", "Frequent highly variable days",
                "attention", "00:00-24:00",
                f"On {_ev(len(aff), len(days))}, day-level glucose variability (CV) exceeded "
                f"{CV_STABLE_THRESHOLD_PCT:.0f}%.",
                aff, len(days), {"median_daily_cv_pct": np.median(list(cvs.values()))}))

    out.sort(key=lambda p: (SEVERITY_ORDER[p.severity], -p.fraction, p.id))
    return out


@dataclass
class PatternReport:
    n_days_profiled: int
    patterns: list[Pattern] | None
    clustering: ClusteringResult | None


def detect_all(
    clean: CleanedSeries, include_rules: bool = True, include_clusters: bool = True,
    max_clusters: int = 5, cfg: PatternConfig | None = None,
) -> PatternReport:
    profiles = build_daily_profiles(clean) if include_clusters else None
    return PatternReport(
        n_days_profiled=0 if profiles is None else len(profiles),
        patterns=detect_rule_patterns(clean, cfg) if include_rules else None,
        clustering=cluster_daily_profiles(profiles, max_k=max_clusters) if include_clusters else None,
    )
