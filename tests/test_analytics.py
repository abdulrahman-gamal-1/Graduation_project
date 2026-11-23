import numpy as np
import pytest

from app.services.analytics import (
    ambulatory_glucose_profile, compute_gmi, compute_glucose_stats, compute_time_in_range,
    daily_summaries, rolling_statistics, summarize,
)
from app.services.features import build_daily_profiles, count_excursion_events, extract_user_features
from app.services.preprocessing import PreprocessConfig, clean_cgm_series
from tests.conftest import make_cgm


def test_gmi_formula():
    assert compute_gmi(154) == pytest.approx(3.31 + 0.02392 * 154)
    assert compute_gmi(154) == pytest.approx(6.99, abs=0.01)  # mean 154 mg/dL ~ GMI 7.0%


def test_tir_bands_sum_to_100_and_boundaries():
    v = np.array([53, 54, 69, 70, 180, 181, 250, 251], dtype=float)
    t = compute_time_in_range(v)
    assert t["very_low_pct"] == 12.5    # 53
    assert t["low_pct"] == 25.0         # 54, 69
    assert t["in_range_pct"] == 25.0    # 70, 180
    assert t["high_pct"] == 25.0        # 181, 250
    assert t["very_high_pct"] == 12.5   # 251
    total = t["very_low_pct"] + t["low_pct"] + t["in_range_pct"] + t["high_pct"] + t["very_high_pct"]
    assert total == pytest.approx(100)


def test_cv_and_sd():
    v = np.array([100.0, 120.0, 140.0])
    s = compute_glucose_stats(v)
    assert s["sd_mg_dl"] == pytest.approx(20.0)
    assert s["cv_pct"] == pytest.approx(100 * 20 / 120, abs=0.01)


def test_summary_excludes_imputed_and_warns_on_short_data():
    df = make_cgm(days=3).drop(index=range(50, 53))
    clean = clean_cgm_series(df)
    s = summarize(clean)
    assert s["n_readings"] == clean.report.n_observed
    assert not s["meets_consensus_criteria"]
    assert any("days of data" in w for w in s["warnings"])


def test_full_14_days_meets_criteria(cgm):
    s = summarize(clean_cgm_series(cgm))
    assert s["meets_consensus_criteria"]
    assert 0 < s["time_in_range"]["in_range_pct"] <= 100


def test_rolling_and_daily_and_agp(cgm):
    clean = clean_cgm_series(cgm)
    r = rolling_statistics(clean, [60, 1440], 60)
    assert [x["window_minutes"] for x in r] == [60, 1440]
    assert len(r[0]["points"]) == 14 * 24
    assert r[1]["points"][0]["mean_mg_dl"] is None  # 24h window warm-up: first point has <50% of window
    d = daily_summaries(clean)
    assert len(d) == 14 and all(x["coverage_pct"] > 99 for x in d)
    agp = ambulatory_glucose_profile(clean)
    assert len(agp) == 24 and agp[0]["p5"] <= agp[0]["p50"] <= agp[0]["p95"]


def test_event_counting():
    import pandas as pd
    s = pd.Series([100, 65, 60, 62, 100, 65, 60, 100, 50, 50, 50, np.nan, 50, 50],
                  index=pd.date_range("2026-01-01", periods=14, freq="5min"))
    # runs <70: [65,60,62]=15min OK, [65,60]=10min no, [50,50,50]=15min OK, [50,50] after NaN=10min no
    assert count_excursion_events(s, 70, True, 5) == 2


def test_features_and_daily_profiles_with_timezone(cgm):
    clean = clean_cgm_series(cgm, PreprocessConfig(timezone="Africa/Cairo"))
    f = extract_user_features(clean)
    for k in ("tir_pct", "cv_pct", "gmi_pct", "block_overnight_mean", "hypo_events_per_week"):
        assert k in f
    assert f["n_valid_days"] >= 13
    prof = build_daily_profiles(clean)
    assert prof.shape[1] == 24 and len(prof) >= 13 and not prof.isna().any().any()
