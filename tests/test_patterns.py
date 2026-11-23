import numpy as np

from app.services.clustering import cluster_daily_profiles
from app.services.features import build_daily_profiles
from app.services.patterns import detect_all, detect_rule_patterns
from app.services.preprocessing import clean_cgm_series
from tests.conftest import bump, make_profile_cgm


def _ids(patterns):
    return {p.id: p for p in patterns}


def _clean(fn, **kw):
    return clean_cgm_series(make_profile_cgm(fn, **kw))


def test_flat_trace_has_no_patterns():
    assert detect_rule_patterns(_clean(lambda d, h: np.full_like(h, 110.0))) == []


def test_recurrent_overnight_hypo_detected_with_evidence():
    def fn(d, h):
        v = np.full_like(h, 110.0)
        if d % 3 == 0:                       # 7 of 21 nights
            v[(h >= 2) & (h < 3)] = 60.0     # 60 min < 70, but not < 54
        return v
    p = _ids(detect_rule_patterns(_clean(fn)))["hypo_overnight"]
    assert (p.n_days_affected, p.n_days_evaluated) == (7, 21)
    assert p.severity == "attention" and p.metrics["days_below_54"] == 0
    assert "not a diagnosis" in p.description


def test_level2_hypo_is_priority():
    def fn(d, h):
        v = np.full_like(h, 110.0)
        if d % 3 == 0:
            v[(h >= 2) & (h < 3)] = 45.0
        return v
    out = detect_rule_patterns(_clean(fn))
    assert out[0].id == "hypo_overnight" and out[0].severity == "priority"  # sorted first


def test_single_hypo_night_is_not_a_recurring_pattern():
    def fn(d, h):
        v = np.full_like(h, 110.0)
        if d == 4:
            v[(h >= 2) & (h < 3)] = 55.0
        return v
    assert "hypo_overnight" not in _ids(detect_rule_patterns(_clean(fn)))


def test_dawn_rise():
    fn = lambda d, h: 100 + np.clip((h - 3) / 3, 0, 1) * 45  # ramps +45 from 03:00 to 06:00
    p = _ids(detect_rule_patterns(_clean(fn)))["dawn_rise"]
    assert p.fraction == 1.0 and p.metrics["median_rise_mg_dl"] > 30


def test_breakfast_spike_only():
    fn = lambda d, h: 110 + bump(h, 8.0, 100, 0.7)
    ids = _ids(detect_rule_patterns(_clean(fn)))
    assert "postmeal_breakfast" in ids
    assert "postmeal_lunch" not in ids and "postmeal_dinner" not in ids
    assert 7 <= ids["postmeal_breakfast"].metrics["median_peak_hour"] <= 9


def test_sustained_afternoon_hyper():
    def fn(d, h):
        v = np.full_like(h, 120.0)
        v[(h >= 12) & (h < 18)] = 220.0
        return v
    p = _ids(detect_rule_patterns(_clean(fn)))["hyper_afternoon"]
    assert p.fraction == 1.0 and p.severity == "attention"


def test_high_variability_days():
    fn = lambda d, h: 150 + 90 * np.sin(h / 4 * 2 * np.pi)  # CV ~ 42%
    assert "high_variability_days" in _ids(detect_rule_patterns(_clean(fn)))


def test_too_few_days_reports_nothing():
    def fn(d, h):
        v = np.full_like(h, 110.0)
        v[(h >= 2) & (h < 3)] = 45.0
        return v
    assert detect_rule_patterns(_clean(fn, days=3)) == []


def _weekday_weekend(d, h):
    weekend = (d + 3) % 7 >= 5          # 2026-01-01 is a Thursday -> d=2,3 are Sat/Sun
    base = 110 + bump(h, 8, 40, 0.8)
    return base + (bump(h, 13, 110, 1.0) if weekend else bump(h, 19, 70, 1.0))


def test_kmeans_recovers_weekday_weekend_day_types():
    clean = _clean(_weekday_weekend, days=28, noise=2.0)
    res = cluster_daily_profiles(build_daily_profiles(clean))
    assert res.status == "ok" and res.k == 2 and res.silhouette > 0.5
    big, small = res.clusters
    assert (big.n_days, small.n_days) == (20, 8)             # ids ordered by size
    assert big.weekend_fraction == 0.0 and small.weekend_fraction == 1.0
    assert small.peak_hour in (12, 13, 14) and big.peak_hour in (18, 19, 20)
    assert len(res.assignments) == 28


def test_kmeans_does_not_invent_structure():
    clean = _clean(lambda d, h: 120 + bump(h, 8, 40, 1.0), days=21, noise=3.0)
    res = cluster_daily_profiles(build_daily_profiles(clean))
    assert res.status == "no_distinct_structure" and res.k == 1
    assert res.clusters[0].n_days == 21 and res.clusters[0].label == "All days"


def test_kmeans_needs_enough_days():
    clean = _clean(_weekday_weekend, days=5)
    assert cluster_daily_profiles(build_daily_profiles(clean)).status == "insufficient_days"


def test_kmeans_is_deterministic_and_facade_flags():
    clean = _clean(_weekday_weekend, days=28, noise=2.0)
    a, b = detect_all(clean), detect_all(clean)
    assert a.clustering.assignments == b.clustering.assignments
    only_rules = detect_all(clean, include_clusters=False)
    assert only_rules.clustering is None and only_rules.patterns is not None
    only_cl = detect_all(clean, include_rules=False)
    assert only_cl.patterns is None and only_cl.n_days_profiled >= 27
