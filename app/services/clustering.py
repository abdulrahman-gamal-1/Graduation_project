"""K-Means clustering of daily glucose profiles ("what kinds of days does this person have?").

Input: days x 24 matrix of hourly mean glucose in local time (features.build_daily_profiles).

Design choices
- No feature scaling: all 24 columns share one unit (mg/dL) and absolute level is clinically
  meaningful (a 'high day' must stay distinguishable from a 'low day').
- k is chosen by silhouette score over 2..max_k. If even the best silhouette is weak
  (< min_silhouette) we DO NOT force clusters: we report 'no_distinct_structure' and a single
  all-days profile. Reporting fake structure would be worse than reporting none.
- Deterministic (fixed random_state, n_init=10) and cluster ids are ordered by size
  (0 = most common day type), so results are stable across calls.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from app.core.constants import TIME_BLOCKS

LEVEL_DELTA_MG_DL = 10.0  # cluster mean vs overall mean to call a day type higher/lower


@dataclass
class ClusterProfile:
    cluster_id: int
    label: str
    n_days: int
    fraction: float
    centroid_mg_dl: list[float]          # 24 hourly values, local time
    mean_mg_dl: float
    peak_hour: int
    peak_mg_dl: float
    nadir_hour: int
    nadir_mg_dl: float
    weekend_fraction: float              # share of this cluster's days that are Sat/Sun
    dates: list = field(default_factory=list)


@dataclass
class ClusteringResult:
    status: str                          # ok | insufficient_days | no_distinct_structure
    message: str
    n_days: int
    k: int | None = None
    silhouette: float | None = None
    candidate_silhouettes: dict[int, float] = field(default_factory=dict)
    clusters: list[ClusterProfile] = field(default_factory=list)
    assignments: list[dict] = field(default_factory=list)   # [{"day": date, "cluster_id": int}]


def _block_of(hour: int) -> str:
    return next(n for n, (a, b) in TIME_BLOCKS.items() if a <= hour < b)


def _profile(cid: int, rows: pd.DataFrame, overall_mean: float, n_total: int, single: bool) -> ClusterProfile:
    centroid = rows.mean(axis=0).to_numpy()
    mean = float(centroid.mean())
    peak_h, nadir_h = int(np.argmax(centroid)), int(np.argmin(centroid))
    if single:
        label = "All days"
    else:
        delta = mean - overall_mean
        level = ("Higher-glucose days" if delta >= LEVEL_DELTA_MG_DL
                 else "Lower-glucose days" if delta <= -LEVEL_DELTA_MG_DL
                 else "Typical-level days")
        label = f"{level}, {_block_of(peak_h)} peak"
        if centroid.min() < 70:
            label += ", with low periods"
    weekend = float(np.mean([pd.Timestamp(d).weekday() >= 5 for d in rows.index]))
    return ClusterProfile(
        cluster_id=cid, label=label, n_days=len(rows), fraction=round(len(rows) / n_total, 3),
        centroid_mg_dl=[round(float(x), 1) for x in centroid], mean_mg_dl=round(mean, 1),
        peak_hour=peak_h, peak_mg_dl=round(float(centroid[peak_h]), 1),
        nadir_hour=nadir_h, nadir_mg_dl=round(float(centroid[nadir_h]), 1),
        weekend_fraction=round(weekend, 2), dates=[d for d in rows.index],
    )


def cluster_daily_profiles(
    profiles: pd.DataFrame, max_k: int = 5, min_days: int = 7,
    min_silhouette: float = 0.25, seed: int = 42,
) -> ClusteringResult:
    n = len(profiles)
    if n < min_days:
        return ClusteringResult(
            status="insufficient_days", n_days=n,
            message=f"Need at least {min_days} complete days to look for recurring day types "
                    f"(have {n}).",
        )

    X = profiles.to_numpy(dtype=float)
    overall_mean = float(X.mean())
    k_max = min(max_k, n // 3)           # keep ~3+ days per cluster on average
    scores: dict[int, float] = {}
    fits: dict[int, np.ndarray] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # duplicate-point ConvergenceWarnings on flat data
        for k in range(2, k_max + 1):
            labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
            if len(np.unique(labels)) < 2:
                continue
            scores[k] = round(float(silhouette_score(X, labels)), 4)
            fits[k] = labels

    if not scores or max(scores.values()) < min_silhouette:
        labels = np.zeros(n, dtype=int)
        best_k, best_sil = 1, (max(scores.values()) if scores else None)
        status = "no_distinct_structure"
        msg = ("Daily profiles do not separate into clearly different day types; showing one "
               "overall profile.")
    else:
        best_k = max(scores, key=scores.get)   # ties -> smaller k (dict order is ascending)
        labels, best_sil, status = fits[best_k], scores[best_k], "ok"
        msg = f"Found {best_k} recurring day types."

    # Re-number clusters by size (largest = 0) for stable, readable ids.
    order = np.argsort(-np.bincount(labels))
    remap = {int(old): new for new, old in enumerate(order) if (labels == old).any()}
    new_labels = np.array([remap[int(l)] for l in labels])

    clusters = [
        _profile(cid, profiles[new_labels == cid], overall_mean, n, single=(best_k == 1))
        for cid in range(len(remap))
    ]
    assignments = [{"day": d, "cluster_id": int(c)} for d, c in zip(profiles.index, new_labels)]
    return ClusteringResult(
        status=status, message=msg, n_days=n, k=best_k, silhouette=best_sil,
        candidate_silhouettes=scores, clusters=clusters, assignments=assignments,
    )
