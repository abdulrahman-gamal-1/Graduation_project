"""Anomaly detection: deterministic safety rules + Isolation Forest / robust z-score.

Layer 1 - RULES (always on, never delegated to a model): sustained urgent low (<54), low (<70),
          urgent high (>250) for >= 15 min, and rapid falls/rises (rate over 15 min).
Layer 2 - STATISTICAL ("unusual for THIS person at THIS time of day"):
          features are deviations from the person's own median at the same half-hour of day,
          so recurring meal spikes are NOT anomalies. Two modes:
            * isolation_forest: IF proposes the most unusual points (contamination), and a
              robust z-score on the deviation must CONFIRM them (|z| >= 3 and a minimum
              practical size). Without confirmation IF would flag ~1% of points even on a
              perfectly clean trace.
            * robust_z (fallback when < 3 days of data, sklearn failure, or requested):
              modified z-score (median/MAD, Iglewicz-Hoaglin) >= 3.5 plus the same size floors.
Artifact guard: detections need >= 2 consecutive grid points (rate / statistical) or >= 15 min
          (level rules); isolated single points are dropped and counted, because one-point
          spikes are typically sensor noise.

Retrospective analytics only - this is NOT a real-time alarm system.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.constants import (
    EVENT_MIN_DURATION_MINUTES, TARGET_LOW, VERY_HIGH_ABOVE, VERY_LOW_BELOW, TARGET_HIGH,
)
from app.services.preprocessing import CleanedSeries

SEVERITY_ORDER = {"priority": 0, "attention": 1, "info": 2}
# Headline priority: sustained STATES outrank RATES, rules outrank statistical flags.
TYPE_ORDER = ["urgent_low", "low", "urgent_high", "rapid_fall", "rapid_rise", "statistical_outlier"]
RULE_TYPES = TYPE_ORDER[:-1]
TYPE_LABEL = {
    "urgent_low": "urgent low", "rapid_fall": "rapid fall", "low": "low",
    "urgent_high": "urgent high", "rapid_rise": "rapid rise",
    "statistical_outlier": "statistical outlier",
}
DRIVER_LABEL = {
    "resid_level": "glucose level unusual for this time of day",
    "resid_roc": "rate of change unusual for this time of day",
}
_NOTE = " Consider reviewing it with your care team; this is not a diagnosis."


@dataclass(frozen=True)
class AnomalyConfig:
    method: str = "auto"                  # auto | isolation_forest | robust_z
    include_statistical: bool = True
    urgent_low: float = VERY_LOW_BELOW
    low: float = TARGET_LOW
    urgent_high: float = VERY_HIGH_ABOVE
    min_event_minutes: int = EVENT_MIN_DURATION_MINUTES
    rapid_fall_mg_dl_per_min: float = 2.0     # consensus trend-arrow convention
    rapid_rise_mg_dl_per_min: float = 3.0
    roc_window_minutes: int = 15
    min_rate_points: int = 2
    min_stat_points: int = 2
    contamination: float = 0.01
    confirm_z: float = 3.0                # IF mode confirmation
    z_threshold: float = 3.5              # robust_z mode
    floor_level_mg_dl: float = 40.0       # min |deviation from usual| to count
    floor_roc_mg_dl_per_min: float = 1.0
    min_days_for_model: int = 3
    merge_gap_minutes: int = 15
    n_estimators: int = 200
    seed: int = 42


@dataclass
class MethodInfo:
    mode: str                             # isolation_forest | robust_z | rules_only
    reason: str | None
    n_points_scored: int
    n_candidates: int
    n_confirmed: int


@dataclass
class AnomalyEvent:
    id: str
    primary_type: str
    types: list[str]
    severity: str
    methods: list[str]
    start: datetime
    end: datetime
    local_start: datetime
    duration_minutes: float
    glucose_min: float
    glucose_max: float
    peak_rate_mg_dl_per_min: float | None
    max_abs_robust_z: float | None
    isolation_score: float | None
    driver: str | None
    description: str


@dataclass
class AnomalyReport:
    method: MethodInfo
    n_events_total: int
    counts_by_type: dict[str, int]
    events_by_local_hour: list[int]
    n_isolated_points_ignored: int
    events: list[AnomalyEvent] = field(default_factory=list)


# ----------------------------------------------------------------------------- helpers
def robust_z(x: np.ndarray) -> np.ndarray:
    """Modified z-score 0.6745*(x-median)/MAD; mean-abs-deviation fallback if MAD == 0."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad > 0:
        return 0.6745 * (x - med) / mad
    meanad = np.nanmean(np.abs(x - med))
    return (x - med) / (1.253314 * meanad) if meanad > 0 else np.zeros_like(x)


def _sustained(mask: np.ndarray, min_pts: int) -> np.ndarray:
    """Keep only runs of True at least `min_pts` long."""
    if min_pts <= 1:
        return mask.copy()
    out = np.zeros_like(mask)
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_pts:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def build_features(clean: CleanedSeries, cfg: AnomalyConfig) -> pd.DataFrame:
    """Per-grid-point features, indexed like clean.frame (UTC)."""
    f, interval = clean.frame, clean.config.interval_minutes
    g = f["glucose"]
    imputed = f["is_imputed"]
    local_idx = g.index.tz_convert(clean.config.timezone)
    bins = (local_idx.hour * 2 + local_idx.minute // 30).to_numpy()   # 48 half-hour bins

    lag = max(1, cfg.roc_window_minutes // interval)
    roc = (g - g.shift(lag)) / (lag * interval)                        # mg/dL per minute
    usual_level = g.where(~imputed).groupby(bins).transform("median")
    usual_roc = roc.where(~imputed).groupby(bins).transform("median")
    sd_1h = g.rolling(max(3, 60 // interval), min_periods=max(2, int(0.66 * 60 // interval))).std()
    return pd.DataFrame({
        "glucose": g, "roc": roc, "resid_level": g - usual_level,
        "resid_roc": roc - usual_roc, "sd_1h": sd_1h,
    })


def _rule_masks(feats: pd.DataFrame, cfg: AnomalyConfig, interval: int) -> dict[str, np.ndarray]:
    g, roc = feats["glucose"].to_numpy(), feats["roc"].to_numpy()
    lvl = math.ceil(cfg.min_event_minutes / interval)
    urgent = _sustained(g < cfg.urgent_low, lvl)
    return {
        "urgent_low": urgent,
        "low": _sustained(g < cfg.low, lvl) & ~urgent,
        "urgent_high": _sustained(g > cfg.urgent_high, lvl),
        "rapid_fall": _sustained(roc <= -cfg.rapid_fall_mg_dl_per_min, cfg.min_rate_points),
        "rapid_rise": _sustained(roc >= cfg.rapid_rise_mg_dl_per_min, cfg.min_rate_points),
    }


def _statistical(
    clean: CleanedSeries, feats: pd.DataFrame, cfg: AnomalyConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, MethodInfo]:
    """Returns (flag, iso_score, max_z, per-feature hit z, method info), all full-length."""
    n = len(feats)
    flag, iso, maxz = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)
    zhit = pd.DataFrame(0.0, index=feats.index, columns=list(DRIVER_LABEL))

    usable_mask = (~clean.frame["is_imputed"]).to_numpy() & feats.notna().all(axis=1).to_numpy()
    X = feats.loc[usable_mask]
    interval = clean.config.interval_minutes
    min_pts = cfg.min_days_for_model * 1440 // interval

    # Decide mode
    reason = None
    if cfg.method == "robust_z":
        mode, reason = "robust_z", "robust_z requested"
    elif len(X) < min_pts:
        mode = "robust_z"
        reason = (f"Only {len(X)} usable points (< {cfg.min_days_for_model} days); "
                  "using robust z-score instead of Isolation Forest.")
    else:
        mode = "isolation_forest"

    if len(X) < 10:
        return flag, iso, maxz, zhit, MethodInfo(mode, reason or "Too few points", len(X), 0, 0)

    floors = {"resid_level": cfg.floor_level_mg_dl, "resid_roc": cfg.floor_roc_mg_dl_per_min}
    abs_z = {k: np.abs(robust_z(X[k].to_numpy())) for k in floors}
    zmax = np.maximum.reduce(list(abs_z.values()))

    def hits(thr: float) -> np.ndarray:
        """Points whose deviation is both statistically large (robust z) and practically large."""
        any_hit = np.zeros(len(X), bool)
        for name, floor in floors.items():
            h = (abs_z[name] >= thr) & (np.abs(X[name].to_numpy()) >= floor)
            zhit.loc[X.index[h], name] = abs_z[name][h]
            any_hit |= h
        return any_hit

    if mode == "isolation_forest":
        cols = ["glucose", "roc", "resid_level", "resid_roc", "sd_1h"]
        try:
            from sklearn.ensemble import IsolationForest
            model = IsolationForest(n_estimators=cfg.n_estimators, contamination=cfg.contamination,
                                    random_state=cfg.seed, n_jobs=1)
            cand = model.fit_predict(X[cols]) == -1
            iso[usable_mask] = -model.score_samples(X[cols])
            n_cand = int(cand.sum())
            candidates = cand & hits(cfg.confirm_z)        # IF proposes, robust z confirms
        except Exception as exc:                           # sklearn missing/failed -> fallback
            mode, reason = "robust_z", f"Isolation Forest unavailable ({type(exc).__name__}); fell back."
            zhit.loc[:, :] = 0.0
            candidates = hits(cfg.z_threshold)
            n_cand = int(candidates.sum())
    else:
        candidates = hits(cfg.z_threshold)
        n_cand = int(candidates.sum())

    flag[usable_mask] = candidates
    maxz[usable_mask] = zmax
    return flag, iso, maxz, zhit, MethodInfo(mode, reason, len(X), n_cand, int(candidates.sum()))


def _describe(primary: str, types: list[str], dur: float, gmin: float, gmax: float,
              rate: float | None, driver: str | None, severity: str) -> str:
    r = abs(rate) if rate is not None else 0.0
    text = {
        "urgent_low": f"Glucose was below 54 mg/dL for about {dur:.0f} minutes (lowest {gmin:.0f} mg/dL).",
        "low": f"Glucose was below 70 mg/dL for about {dur:.0f} minutes (lowest {gmin:.0f} mg/dL).",
        "rapid_fall": f"Glucose fell quickly (up to {r:.1f} mg/dL per minute; lowest {gmin:.0f} mg/dL). "
                      "Very rapid drops can also be sensor artifacts (e.g. pressure on the sensor).",
        "urgent_high": f"Glucose stayed above 250 mg/dL for about {dur:.0f} minutes (peak {gmax:.0f} mg/dL).",
        "rapid_rise": f"Glucose rose quickly (up to {r:.1f} mg/dL per minute; peak {gmax:.0f} mg/dL).",
        "statistical_outlier": (
            f"Glucose behaviour was unusual for this person at this time of day"
            f"{' (' + DRIVER_LABEL[driver] + ')' if driver else ''}; range {gmin:.0f}-{gmax:.0f} mg/dL. "
            "Statistical flag only - it may be a sensor artifact or an unrecorded event."),
    }[primary]
    others = [TYPE_LABEL[t] for t in types if t != primary]
    if others:
        text += f" Also flagged: {', '.join(others)}."
    return text + (_NOTE if severity != "info" else "")


def _severity(types: list[str], gmax: float) -> str:
    if "urgent_low" in types or ("rapid_fall" in types and "low" in types):
        return "priority"
    if any(t in types for t in ("low", "rapid_fall", "urgent_high")):
        return "attention"
    if "rapid_rise" in types and gmax > TARGET_HIGH:
        return "attention"
    return "info"


def detect_anomalies(clean: CleanedSeries, cfg: AnomalyConfig | None = None, max_events: int = 100) -> AnomalyReport:
    cfg = cfg or AnomalyConfig()
    interval, tz = clean.config.interval_minutes, clean.config.timezone
    frame = clean.frame
    feats = build_features(clean, cfg)
    n = len(frame)

    rules = _rule_masks(feats, cfg, interval)
    raw_rate = ((feats["roc"] <= -cfg.rapid_fall_mg_dl_per_min)
                | (feats["roc"] >= cfg.rapid_rise_mg_dl_per_min)).to_numpy()

    if cfg.include_statistical:
        s_flag, iso, maxz, zhit, info = _statistical(clean, feats, cfg)
    else:
        s_flag, iso, maxz = np.zeros(n, bool), np.full(n, np.nan), np.full(n, np.nan)
        zhit = pd.DataFrame(0.0, index=feats.index, columns=list(DRIVER_LABEL))
        info = MethodInfo("rules_only", "statistical detection disabled", 0, 0, 0)
    s_kept = _sustained(s_flag, cfg.min_stat_points)
    stat_method = info.mode if info.mode != "rules_only" else None

    flags = {**rules, "statistical_outlier": s_kept}
    any_flag = np.any(np.vstack(list(flags.values())), axis=0)
    raw_any = any_flag | raw_rate | s_flag
    n_ignored = int((raw_any & ~any_flag).sum())

    events: list[AnomalyEvent] = []
    pos = np.flatnonzero(any_flag)
    if pos.size:
        gap_pts = cfg.merge_gap_minutes // interval
        groups = np.split(pos, np.flatnonzero(np.diff(pos) > gap_pts + 1) + 1)
        g_arr, roc_arr = feats["glucose"].to_numpy(), feats["roc"].to_numpy()
        lag = max(1, cfg.roc_window_minutes // interval)
        for idx in groups:
            a, b = int(idx[0]), int(idx[-1]) + 1
            types = [t for t in TYPE_ORDER if flags[t][a:b].any()]
            primary = types[0]
            # Extremes cover the excursion +/- one rate window: rate flags fire at the END of a
            # 15-min interval, so the true nadir/peak can sit just outside the flagged points.
            gseg = g_arr[max(0, a - lag):min(n, b + lag)]
            gmin, gmax = float(np.nanmin(gseg)), float(np.nanmax(gseg))
            rseg = roc_arr[a:b]
            rate = float(rseg[np.nanargmax(np.abs(rseg))]) if np.isfinite(rseg).any() else None
            zseg, iseg = maxz[a:b], iso[a:b]
            start_ts, end_ts = frame.index[a], frame.index[b - 1]
            dur = (end_ts - start_ts).total_seconds() / 60 + interval
            driver = None
            if "statistical_outlier" in types:
                col_scores = zhit.iloc[a:b].max()
                driver = str(col_scores.idxmax()) if col_scores.max() > 0 else None
            methods = (["rule"] if any(t in RULE_TYPES for t in types) else []) + \
                      ([stat_method] if "statistical_outlier" in types and stat_method else [])
            sev = _severity(types, gmax)
            events.append(AnomalyEvent(
                id=f"ev_{start_ts:%Y%m%dT%H%M}", primary_type=primary, types=types, severity=sev,
                methods=methods, start=start_ts.to_pydatetime(), end=end_ts.to_pydatetime(),
                local_start=start_ts.tz_convert(tz).to_pydatetime(), duration_minutes=round(dur, 1),
                glucose_min=round(gmin, 1), glucose_max=round(gmax, 1),
                peak_rate_mg_dl_per_min=None if rate is None else round(rate, 2),
                max_abs_robust_z=None if np.isnan(zseg).all() else round(float(np.nanmax(zseg)), 2),
                isolation_score=None if np.isnan(iseg).all() else round(float(np.nanmax(iseg)), 3),
                driver=driver,
                description=_describe(primary, types, dur, gmin, gmax, rate, driver, sev),
            ))

    events.sort(key=lambda e: (SEVERITY_ORDER[e.severity], -e.start.timestamp()))
    counts: dict[str, int] = {}
    by_hour = [0] * 24
    for e in events:
        counts[e.primary_type] = counts.get(e.primary_type, 0) + 1
        by_hour[e.local_start.hour] += 1
    return AnomalyReport(
        method=info, n_events_total=len(events), counts_by_type=counts,
        events_by_local_hour=by_hour, n_isolated_points_ignored=n_ignored,
        events=events[:max_events],
    )
