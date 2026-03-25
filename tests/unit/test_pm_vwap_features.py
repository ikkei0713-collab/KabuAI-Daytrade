"""PM-VWAP session features and trend_follow PM filter."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tools.feature_engineering import FeatureEngineer, JST
from strategies.momentum.trend_follow import TrendFollowStrategy


def _daily(rows: int = 40) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    dates = pd.date_range(end="2025-01-15", periods=rows, freq="B")
    close = 2000 + np.cumsum(rng.randn(rows) * 5)
    o = close + rng.randn(rows) * 2
    h = np.maximum(o, close) + abs(rng.randn(rows) * 3)
    l = np.minimum(o, close) - abs(rng.randn(rows) * 3)
    vol = (500_000 + rng.randint(0, 200_000, rows)).astype(float)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": close, "volume": vol},
        index=dates,
    )


def _intraday_pm_reclaim() -> pd.DataFrame:
    """12:30 以降に PM-VWAP を下から上へ reclaim し、終端で上に定着。"""
    base = datetime(2025, 1, 15, 9, 0, tzinfo=JST)
    rows = []
    t = base
    price = 1000.0
    for _ in range(20):
        rows.append((t, price, price + 1, price - 1, price, 50_000))
        t += timedelta(minutes=5)
        price += 0.2
    pm_start = datetime(2025, 1, 15, 12, 30, tzinfo=JST)
    t = pm_start
    prices_pm = [998, 997, 996, 997.5, 999, 1001, 1002, 1003]
    for p in prices_pm:
        rows.append((t, p, p + 0.5, p - 0.5, p, 120_000))
        t += timedelta(minutes=5)
    return pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])


@pytest.fixture
def fe() -> FeatureEngineer:
    return FeatureEngineer()


def test_pm_features_proxy_clock_afternoon(fe: FeatureEngineer) -> None:
    df = _daily()
    clock = datetime(2025, 1, 15, 13, 15, tzinfo=JST)
    out = fe.calculate_all_features(df, clock=clock)
    assert out.get("pm_session_active") is True
    assert out.get("pm_minutes_from_open", 0) >= 30
    assert out.get("pm_intraday_is_proxy") is True
    assert out.get("pm_intraday_quality_score", 0) < 0.6


def test_pm_features_real_intraday(fe: FeatureEngineer) -> None:
    df = _daily()
    idf = _intraday_pm_reclaim()
    clock = datetime(2025, 1, 15, 14, 0, tzinfo=JST)
    out = fe.calculate_all_features(df, clock=clock, intraday_ohlcv=idf)
    assert out.get("pm_intraday_is_proxy") is False
    assert out.get("pm_intraday_quality_score", 0) >= 0.6
    assert out.get("pm_vwap_hold_count", 0) >= 1


def test_pm_trend_filter_requires_alignment() -> None:
    feats = {
        "ema_9": 102.0,
        "ema_21": 100.0,
        "vwap": 99.0,
        "current_price": 103.0,
        "trend_strength": 0.5,
        "volume_trend": 1.3,
        "pm_vwap": 101.0,
        "pm_vwap_slope": 0.0001,
    }
    ok, _ = TrendFollowStrategy.pm_trend_filter(feats)
    assert ok is True


def test_pm_trend_filter_rejects_weak_trend() -> None:
    feats = {
        "ema_9": 101.0,
        "ema_21": 100.0,
        "vwap": 99.0,
        "current_price": 103.0,
        "trend_strength": 0.2,
        "volume_trend": 1.3,
        "pm_vwap": 101.0,
        "pm_vwap_slope": 0.0001,
    }
    ok, reason = TrendFollowStrategy.pm_trend_filter(feats)
    assert ok is False
    assert "trend" in reason.lower()
