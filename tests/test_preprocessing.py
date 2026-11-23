import numpy as np
import pandas as pd
import pytest

from app.core.errors import InsufficientDataError
from app.services.preprocessing import PreprocessConfig, clean_cgm_series
from tests.conftest import make_cgm


def test_clipping_and_report(cgm):
    cgm.loc[10, "glucose"] = 20      # below physical min
    cgm.loc[20, "glucose"] = 550     # above physical max
    out = clean_cgm_series(cgm)
    f = out.frame["glucose"].dropna()
    assert f.min() >= 38 and f.max() <= 400
    assert out.report.n_clipped_low == 1 and out.report.n_clipped_high == 1


def test_invalid_values_become_missing_not_clipped(cgm):
    cgm.loc[5, "glucose"] = -10
    cgm.loc[6, "glucose"] = np.nan
    out = clean_cgm_series(cgm)
    assert out.report.n_invalid_value == 2
    assert out.report.n_clipped_low == 0


def test_short_gap_is_imputed_long_gap_is_not():
    df = make_cgm(days=2, noise=0)
    df = df.drop(index=range(100, 104))      # 20 min gap -> impute
    df = df.drop(index=range(300, 320))      # 100 min gap -> keep missing
    out = clean_cgm_series(df, PreprocessConfig(max_interpolation_gap_minutes=30))
    assert out.report.n_imputed == 4
    assert out.report.n_missing_remaining == 20
    assert out.report.n_gaps_remaining == 1
    assert out.report.longest_gap_minutes == 100
    assert out.frame["is_imputed"].sum() == 4


def test_resampling_aligns_jittered_timestamps_to_grid():
    ts = pd.to_datetime(["2026-01-01 00:00:40", "2026-01-01 00:05:10", "2026-01-01 00:09:59",
                         "2026-01-01 00:15:01"], utc=True)
    df = pd.DataFrame({"timestamp": ts, "glucose": [100, 110, 120, 130]})
    out = clean_cgm_series(df)
    assert (out.frame.index.minute % 5 == 0).all()
    assert len(out.frame) == 4


def test_duplicates_are_averaged():
    ts = pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:00", "2026-01-01 00:05"], utc=True)
    out = clean_cgm_series(pd.DataFrame({"timestamp": ts, "glucose": [100, 120, 130]}))
    assert out.report.n_duplicate_timestamps == 1
    assert out.frame["glucose"].iloc[0] == pytest.approx(110)


def test_naive_timestamps_treated_as_utc():
    df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00", "2026-01-01T00:05:00"],
                       "glucose": [100, 105]})
    out = clean_cgm_series(df)
    assert str(out.frame.index.tz) == "UTC"


def test_no_valid_data_raises():
    df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00"], "glucose": [np.nan]})
    with pytest.raises(InsufficientDataError):
        clean_cgm_series(df)
