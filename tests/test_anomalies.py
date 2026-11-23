import numpy as np

from app.services.anomalies import AnomalyConfig, detect_anomalies, robust_z
from app.services.preprocessing import clean_cgm_series
from tests.conftest import bump, make_profile_cgm


def _base(h):
    return 110 + bump(h, 8, 60, 0.8) + bump(h, 13, 50, 0.8) + bump(h, 19, 60, 0.8)


def _run(fn, days=14, cfg=None, noise=3.0, seed=0):
    clean = clean_cgm_series(make_profile_cgm(fn, days=days, noise=noise, seed=seed))
    return detect_anomalies(clean, cfg)


def test_robust_z_basics():
    x = np.array([10, 11, 9, 10, 10, 11, 9, 100.0])
    z = robust_z(x)
    assert abs(z[-1]) > 10 and np.all(np.abs(z[:-1]) < 3)
    assert np.all(robust_z(np.full(5, 7.0)) == 0)        # MAD=0 and meanAD=0 -> zeros


def test_clean_periodic_trace_has_no_events_and_uses_isolation_forest():
    r = _run(lambda d, h: _base(h))
    assert r.n_events_total == 0 and r.events == []
    assert r.method.mode == "isolation_forest" and r.method.reason is None


def test_sustained_urgent_low_is_priority_and_rule_based():
    def fn(d, h):
        v = _base(h)
        if d == 5:
            v[(h >= 3) & (h < 3.5)] = 45.0              # 30 min below 54
        return v
    r = _run(fn)
    assert r.n_events_total == 1
    e = r.events[0]
    assert e.primary_type == "urgent_low" and e.severity == "priority"
    assert "rule" in e.methods and 25 <= e.duration_minutes <= 45 and e.glucose_min < 54
    assert "not a diagnosis" in e.description
    assert e.local_start.hour == 3 and r.events_by_local_hour[3] == 1


def test_ten_minute_dip_is_not_an_urgent_low_event():
    def fn(d, h):
        v = _base(h)
        if d == 5:
            v[(h >= 3) & (h < 3 + 10 / 60)] = 45.0      # 2 grid points only
        return v
    r = _run(fn)
    types = {t for e in r.events for t in e.types}
    assert "urgent_low" not in types and "low" not in types
    assert r.events and r.events[0].primary_type == "rapid_fall"


def test_isolated_single_point_spike_is_ignored_as_artifact():
    clean_df = make_profile_cgm(lambda d, h: _base(h), days=14, noise=3.0, seed=0)
    clean_df.loc[1000, "glucose"] += 120                 # one-point sensor spike
    r = detect_anomalies(clean_cgm_series(clean_df))
    assert r.n_events_total == 0
    assert r.n_isolated_points_ignored >= 1


def test_smooth_nonrecurring_plateau_found_only_by_statistical_layer():
    def fn(d, h):
        v = _base(h)
        if d == 6:
            v = v + bump(h, 3.5, 100, 0.6)               # gentle (<2 mg/dL/min) 03:30 bulge to ~210
        return v
    r = _run(fn)
    assert r.method.mode == "isolation_forest"
    stat = [e for e in r.events if e.primary_type == "statistical_outlier"]
    assert len(stat) == 1
    e = stat[0]
    assert e.types == ["statistical_outlier"] and e.severity == "info"
    assert "isolation_forest" in e.methods and e.isolation_score is not None
    assert e.driver == "resid_level" and 2 <= e.local_start.hour <= 4


def test_robust_z_mode_forced_finds_same_plateau():
    def fn(d, h):
        v = _base(h)
        if d == 6:
            v = v + bump(h, 3.5, 100, 0.6)
        return v
    r = _run(fn, cfg=AnomalyConfig(method="robust_z"))
    assert r.method.mode == "robust_z" and r.method.reason == "robust_z requested"
    assert any("robust_z" in e.methods and e.primary_type == "statistical_outlier" for e in r.events)
    assert all(e.isolation_score is None for e in r.events)


def test_rapid_fall_above_70_is_attention_not_priority():
    def fn(d, h):
        if d == 4:
            return np.interp(h, [0, 15, 15 + 25 / 60, 16, 17, 24], [150, 150, 80, 80, 150, 150])
        return np.full_like(h, 150.0)
    r = _run(fn, cfg=AnomalyConfig(include_statistical=False))
    fall = [e for e in r.events if e.primary_type == "rapid_fall"]
    assert len(fall) == 1 and fall[0].severity == "attention"
    assert 70 <= fall[0].glucose_min <= 90  # target 80, sensor noise sigma=3 and fall[0].peak_rate_mg_dl_per_min < -2
    assert not any(e.severity == "priority" for e in r.events)


def test_events_sorted_priority_first():
    def fn(d, h):
        v = _base(h)
        if d == 2:
            v[(h >= 3) & (h < 3.5)] = 45.0
        if d == 9:
            v = v + bump(h, 3.5, 100, 0.6)
        return v
    r = _run(fn)
    sev = [e.severity for e in r.events]
    assert sev[0] == "priority" and sev == sorted(sev, key=["priority", "attention", "info"].index)


def test_short_history_falls_back_but_rules_still_work():
    def fn(d, h):
        v = _base(h)
        if d == 1:
            v[(h >= 3) & (h < 3.5)] = 45.0
        return v
    r = _run(fn, days=2)
    assert r.method.mode == "robust_z" and "days" in r.method.reason
    assert any(e.primary_type == "urgent_low" for e in r.events)


def test_rules_only_mode_and_max_events_cap_and_determinism():
    def fn(d, h):
        v = _base(h)
        if d in (2, 5, 8):
            v[(h >= 3) & (h < 3.5)] = 45.0
        return v
    clean = clean_cgm_series(make_profile_cgm(fn, days=14, noise=3.0))
    ro = detect_anomalies(clean, AnomalyConfig(include_statistical=False))
    assert ro.method.mode == "rules_only" and ro.n_events_total == 3
    assert all(t != "statistical_outlier" for e in ro.events for t in e.types)
    capped = detect_anomalies(clean, max_events=2)
    assert len(capped.events) == 2 and capped.n_events_total >= 3
    a, b = detect_anomalies(clean), detect_anomalies(clean)
    assert [e.id for e in a.events] == [e.id for e in b.events]
