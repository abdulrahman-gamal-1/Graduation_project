import os

os.environ.setdefault("GLUCOCARE_ENV", "dev")

import numpy as np
import pandas as pd
import pytest


def make_cgm(days=14, interval=5, seed=0, base=130, amp=35, noise=8, start="2026-01-01"):
    """Synthetic CGM trace: daily sinusoid + meal-ish bump + noise, UTC timestamps."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440 // interval, freq=f"{interval}min", tz="UTC")
    hours = idx.hour + idx.minute / 60
    vals = base + amp * np.sin((hours - 8) / 24 * 2 * np.pi) + rng.normal(0, noise, len(idx))
    return pd.DataFrame({"timestamp": idx, "glucose": vals})


@pytest.fixture
def cgm():
    return make_cgm()


def make_profile_cgm(fn, days=21, interval=5, noise=3.0, seed=0, start="2026-01-01"):
    """Trace built from fn(day_index, hour_of_day_array) -> mg/dL array (UTC == local)."""
    rng = np.random.default_rng(seed)
    per_day = 1440 // interval
    idx = pd.date_range(start, periods=days * per_day, freq=f"{interval}min", tz="UTC")
    hours = np.tile(np.arange(per_day) * interval / 60.0, days)
    day_i = np.repeat(np.arange(days), per_day)
    vals = np.empty(len(idx))
    for d in range(days):
        m = day_i == d
        vals[m] = fn(d, hours[m])
    vals = vals + rng.normal(0, noise, len(idx))
    return pd.DataFrame({"timestamp": idx, "glucose": vals})


def bump(h, center, amp, width):
    return amp * np.exp(-0.5 * ((h - center) / width) ** 2)
