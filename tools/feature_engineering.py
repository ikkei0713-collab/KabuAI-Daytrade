"""Feature engineering for Japanese stock day-trading.

Calculates technical indicators, volume metrics, VWAP statistics,
price features, candlestick patterns, and momentum signals from
OHLCV DataFrames.  All computations use pandas / numpy directly
so there is no hard dependency on external TA libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class FeatureSet:
    """Container for all computed features on a single ticker."""

    technical: dict[str, Any] = field(default_factory=dict)
    volume: dict[str, Any] = field(default_factory=dict)
    vwap: dict[str, Any] = field(default_factory=dict)
    price: dict[str, Any] = field(default_factory=dict)
    pattern: dict[str, Any] = field(default_factory=dict)
    momentum: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        """Merge every category into a single flat dict."""
        out: dict[str, Any] = {}
        for section in (
            self.technical,
            self.volume,
            self.vwap,
            self.price,
            self.pattern,
            self.momentum,
        ):
            out.update(section)
        return out


class FeatureEngineer:
    """Stateless feature calculator.

    All public methods accept a pandas DataFrame with *at least* the
    columns ``open``, ``high``, ``low``, ``close``, ``volume`` (case-
    insensitive).  The caller is responsible for providing enough rows
    for the longest look-back window (typically 50 bars for SMA-50).
    """

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def calculate_all_features(self, ohlcv: pd.DataFrame) -> dict[str, Any]:
        """Compute every feature category and return a flat dict.

        Args:
            ohlcv: DataFrame with columns open/high/low/close/volume.

        Returns:
            A flat ``{feature_name: value}`` dictionary suitable for
            storage and downstream model consumption.
        """
        df = self._normalise_columns(ohlcv)
        if df.empty:
            logger.warning("Empty DataFrame passed to calculate_all_features")
            return {}

        fs = FeatureSet(
            technical=self._technical(df),
            volume=self._volume(df),
            vwap=self._vwap(df),
            price=self._price(df),
            pattern=self._pattern(df),
            momentum=self._momentum(df),
        )
        return fs.to_flat_dict()

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case column names and validate required columns."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            logger.error("Missing required columns: {}", missing)
            return pd.DataFrame()
        # Ensure numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=list(required), inplace=True)
        return df

    # ------------------------------------------------------------------
    # Technical indicators
    # ------------------------------------------------------------------

    def _technical(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["close"]
        high = df["high"]
        low = df["low"]

        features: dict[str, Any] = {}

        # Simple Moving Averages
        for period in (5, 20, 50):
            sma = close.rolling(window=period, min_periods=period).mean()
            features[f"sma_{period}"] = _last(sma)

        # Exponential Moving Averages
        for period in (9, 21):
            ema = close.ewm(span=period, adjust=False).mean()
            features[f"ema_{period}"] = _last(ema)

        # RSI
        for period in (5, 14):
            features[f"rsi_{period}"] = self._rsi(close, period)

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        features["macd"] = _last(macd_line)
        features["macd_signal"] = _last(signal_line)
        features["macd_histogram"] = _last(histogram)

        # Bollinger Bands (20, 2)
        sma20 = close.rolling(window=20, min_periods=20).mean()
        std20 = close.rolling(window=20, min_periods=20).std()
        features["bb_upper"] = _last(sma20 + 2 * std20)
        features["bb_middle"] = _last(sma20)
        features["bb_lower"] = _last(sma20 - 2 * std20)
        bb_upper_val = features["bb_upper"]
        bb_lower_val = features["bb_lower"]
        if bb_upper_val is not None and bb_lower_val is not None and bb_upper_val != bb_lower_val:
            features["bb_pct"] = (close.iloc[-1] - bb_lower_val) / (bb_upper_val - bb_lower_val)
        else:
            features["bb_pct"] = None

        # ATR (14)
        features["atr_14"] = self._atr(high, low, close, 14)

        return features

    # ------------------------------------------------------------------
    # Volume features
    # ------------------------------------------------------------------

    def _volume(self, df: pd.DataFrame) -> dict[str, Any]:
        vol = df["volume"]
        close = df["close"]
        features: dict[str, Any] = {}

        # Volume ratio vs 20-day average
        vol_ma20 = vol.rolling(window=20, min_periods=1).mean()
        last_vol = vol.iloc[-1]
        last_vol_ma20 = vol_ma20.iloc[-1]
        features["volume_ratio"] = (
            last_vol / last_vol_ma20 if last_vol_ma20 > 0 else None
        )

        # Volume MA (5)
        vol_ma5 = vol.rolling(window=5, min_periods=1).mean()
        features["volume_ma_5"] = _last(vol_ma5)

        # On-Balance Volume
        obv = self._obv(close, vol)
        features["obv"] = _last(obv)

        return features

    # ------------------------------------------------------------------
    # VWAP features
    # ------------------------------------------------------------------

    def _vwap(self, df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap_series = cum_tp_vol / cum_vol.replace(0, np.nan)

        features["vwap"] = _last(vwap_series)

        last_close = df["close"].iloc[-1]
        last_vwap = features["vwap"]
        if last_vwap is not None and last_vwap > 0:
            features["distance_from_vwap"] = (last_close - last_vwap) / last_vwap
        else:
            features["distance_from_vwap"] = None

        # Fraction of bars where close was above VWAP
        above = (df["close"] > vwap_series).sum()
        total = len(df)
        features["time_above_vwap"] = above / total if total > 0 else None

        return features

    # ------------------------------------------------------------------
    # Price features
    # ------------------------------------------------------------------

    def _price(self, df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}

        # Gap percentage (today open vs previous close)
        if len(df) >= 2:
            prev_close = df["close"].iloc[-2]
            today_open = df["open"].iloc[-1]
            features["gap_pct"] = (
                (today_open - prev_close) / prev_close if prev_close > 0 else 0.0
            )
        else:
            features["gap_pct"] = 0.0

        # Intraday range (current bar)
        last_high = df["high"].iloc[-1]
        last_low = df["low"].iloc[-1]
        last_close = df["close"].iloc[-1]
        features["intraday_range"] = (
            (last_high - last_low) / last_close if last_close > 0 else 0.0
        )

        # Distance from recent high / low (20-day)
        recent_high = df["high"].tail(20).max()
        recent_low = df["low"].tail(20).min()
        features["price_vs_high"] = (
            (last_close - recent_high) / recent_high if recent_high > 0 else 0.0
        )
        features["price_vs_low"] = (
            (last_close - recent_low) / recent_low if recent_low > 0 else 0.0
        )

        return features

    # ------------------------------------------------------------------
    # Candlestick patterns
    # ------------------------------------------------------------------

    def _pattern(self, df: pd.DataFrame) -> dict[str, Any]:
        features: dict[str, Any] = {}

        o = df["open"].iloc[-1]
        h = df["high"].iloc[-1]
        l = df["low"].iloc[-1]  # noqa: E741
        c = df["close"].iloc[-1]

        body = abs(c - o)
        full_range = h - l if h != l else 1e-9
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        # Doji: very small body relative to range
        features["is_doji"] = body / full_range < 0.1

        # Hammer: small body at top, long lower shadow
        features["is_hammer"] = (
            lower_shadow > 2 * body and upper_shadow < body * 0.5 and body > 0
        )

        # Inverted hammer
        features["is_inverted_hammer"] = (
            upper_shadow > 2 * body and lower_shadow < body * 0.5 and body > 0
        )

        # Bullish / bearish engulfing (need at least 2 bars)
        if len(df) >= 2:
            prev_o = df["open"].iloc[-2]
            prev_c = df["close"].iloc[-2]
            features["is_bullish_engulfing"] = (
                prev_c < prev_o  # previous bearish
                and c > o  # current bullish
                and o <= prev_c  # opens below prev close
                and c >= prev_o  # closes above prev open
            )
            features["is_bearish_engulfing"] = (
                prev_c > prev_o  # previous bullish
                and c < o  # current bearish
                and o >= prev_c  # opens above prev close
                and c <= prev_o  # closes below prev open
            )
        else:
            features["is_bullish_engulfing"] = False
            features["is_bearish_engulfing"] = False

        # Aggregate pattern label
        if features["is_bullish_engulfing"]:
            features["candle_pattern"] = "bullish_engulfing"
        elif features["is_bearish_engulfing"]:
            features["candle_pattern"] = "bearish_engulfing"
        elif features["is_hammer"]:
            features["candle_pattern"] = "hammer"
        elif features["is_inverted_hammer"]:
            features["candle_pattern"] = "inverted_hammer"
        elif features["is_doji"]:
            features["candle_pattern"] = "doji"
        else:
            features["candle_pattern"] = "none"

        return features

    # ------------------------------------------------------------------
    # Momentum features
    # ------------------------------------------------------------------

    def _momentum(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["close"]
        features: dict[str, Any] = {}

        # Rate of change
        for period in (5, 10):
            if len(close) > period:
                prev = close.iloc[-period - 1]
                features[f"roc_{period}"] = (
                    (close.iloc[-1] - prev) / prev if prev > 0 else 0.0
                )
            else:
                features[f"roc_{period}"] = None

        # Momentum (raw price difference over 10 bars)
        if len(close) > 10:
            features["momentum_10"] = close.iloc[-1] - close.iloc[-11]
        else:
            features["momentum_10"] = None

        return features

    # ------------------------------------------------------------------
    # Internal calculation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> float | None:
        """Compute RSI using the Wilder smoothing method."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return _last(rsi)

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float | None:
        """Average True Range."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean()
        return _last(atr)

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume."""
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _last(series: pd.Series) -> float | None:
    """Return the last non-NaN value of a series, or None."""
    if series is None or series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)
