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
        out = fs.to_flat_dict()

        # Aliases: strategies expect short names
        if "atr_14" in out:
            out["atr"] = out["atr_14"]
        if "distance_from_vwap" in out:
            out["vwap_distance"] = out["distance_from_vwap"]

        # Simulated features for paper trading (日足ベース推定)
        # NOTE: これらは intraday proxy (擬似特徴量) であり、
        #       日足データから推定した値で実際のイントラデーデータではない。
        #       proxy 依存が高い戦略の評価信頼度は限定的。
        close = df["close"].iloc[-1] if not df.empty else 0
        high = df["high"].iloc[-1] if not df.empty else 0
        low = df["low"].iloc[-1] if not df.empty else 0
        vol = df["volume"].iloc[-1] if not df.empty else 0
        vol_avg = df["volume"].tail(20).mean() if len(df) >= 20 else vol

        out.setdefault("pre_market_volume", vol * 0.1)
        out.setdefault("sector_momentum", 0.0)
        out.setdefault("opening_range_size", high - low)
        out.setdefault("opening_range_high", high)
        out.setdefault("opening_range_low", low)
        out.setdefault("volume_first_5min", vol * 0.05)
        out.setdefault("tick_direction", 1 if close > df["open"].iloc[-1] else -1)
        out.setdefault("time_below_vwap", 30)
        out.setdefault("volume_at_reclaim", vol * 1.2)
        out.setdefault("vwap_touches_today", 2)
        out.setdefault("trend_direction", 1 if out.get("ema_9", 0) and out.get("ema_21", 0) and out["ema_9"] > out["ema_21"] else -1)
        out.setdefault("trend_strength", abs(out.get("rsi_14", 50) - 50) / 50)
        out.setdefault("volume_trend", vol / vol_avg if vol_avg > 0 else 1.0)
        out.setdefault("atr_distance_from_vwap", abs(out.get("vwap_distance", 0)) / out.get("atr", 1) if out.get("atr") else 0)
        out.setdefault("volume_climax", 1 if vol > vol_avg * 2.5 else 0)
        out.setdefault("volume_spike", 1 if vol > vol_avg * 2 else 0)
        out.setdefault("price_vs_bollinger", (close - out.get("bb_lower", close)) / max(out.get("bb_upper", close) - out.get("bb_lower", close), 0.01))
        out.setdefault("intraday_drop_pct", (low - high) / high * 100 if high > 0 else 0)
        out.setdefault("volume_surge", vol / vol_avg if vol_avg > 0 else 1.0)
        out.setdefault("selling_exhaustion", 0)
        out.setdefault("support_level", low)
        out.setdefault("bid_ask_ratio", 1.0 + (close - df["open"].iloc[-1]) / max(out.get("atr", 1), 0.01))
        out.setdefault("spread", out.get("atr", 1) * 0.05)
        out.setdefault("depth_imbalance", out.get("bid_ask_ratio", 1.0) - 1.0)
        out.setdefault("volume_price_divergence", 0.0)
        out.setdefault("large_trade_detection", 1 if vol > vol_avg * 3 else 0)
        out.setdefault("spread_percentile", 0.5)
        out.setdefault("volume_building", vol / vol_avg if vol_avg > 0 else 1.0)
        out.setdefault("price_compression", out.get("bb_pct", 0.5))
        out.setdefault("event_type", "")
        out.setdefault("event_magnitude", 0.0)
        out.setdefault("market_cap", 0.0)
        out.setdefault("historical_event_response", 0.0)
        out.setdefault("earnings_surprise_pct", 0.0)
        out.setdefault("revenue_growth", 0.0)
        out.setdefault("guidance_change", 0.0)
        out.setdefault("news_sentiment", 0.0)
        out.setdefault("price_acceleration", 0.0)
        out.setdefault("historical_catalyst_response", 0.0)

        # Proxy feature metadata: 各特徴量が proxy (擬似) か real かを記録
        out["_proxy_features"] = self.get_proxy_features()
        out["_proxy_usage_rate"] = self._calc_proxy_usage_rate(out)

        return out

    # ------------------------------------------------------------------
    # Proxy feature tracking
    # ------------------------------------------------------------------

    # 擬似 intraday 特徴量リスト
    # これらは日足データから推定しており、実際の intraday データではない
    PROXY_FEATURES: set[str] = {
        # Orderbook 系 (完全ダミー)
        "spread_percentile",    # 固定値 0.5
        "bid_ask_ratio",        # close-open から推定
        "spread",               # ATR * 0.05 で推定
        "depth_imbalance",      # bid_ask_ratio - 1.0
        "large_trade_detection",  # volume から推定
        # Intraday timing 系 (日足では不可能)
        "time_below_vwap",      # 固定値 30
        "volume_at_reclaim",    # volume * 1.2 で推定
        "vwap_touches_today",   # 固定値 2
        "volume_first_5min",    # volume * 0.05 で推定
        "pre_market_volume",    # volume * 0.1 で推定
        "tick_direction",       # close vs open で推定
        # Opening range 系 (日足 high/low で代替)
        "opening_range_size",   # high - low (日足レンジ = intraday OR ではない)
        "opening_range_high",   # high (日足高値 ≠ 始値5分高値)
        "opening_range_low",    # low (日足安値 ≠ 始値5分安値)
        # Intraday dynamics 系
        "intraday_drop_pct",    # (low-high)/high で推定
        "selling_exhaustion",   # 固定値 0
        "volume_price_divergence",  # 固定値 0
        "price_acceleration",   # 固定値 0
        # Event 系 (外部データなしの場合)
        "earnings_surprise_pct",  # 固定値 0
        "revenue_growth",         # 固定値 0
        "guidance_change",        # 固定値 0
        "news_sentiment",         # 固定値 0
        "historical_catalyst_response",  # 固定値 0
        "historical_event_response",     # 固定値 0 (TDnet注入時は real)
        "market_cap",            # 固定値 0
    }

    # 戦略が依存する特徴量マッピング
    STRATEGY_PROXY_DEPS: dict[str, list[str]] = {
        "vwap_reclaim": ["time_below_vwap", "volume_at_reclaim"],
        "orb": ["opening_range_high", "opening_range_low"],
        "trend_follow": [],  # EMA/VWAP は real
        "spread_entry": ["spread_percentile", "volume_building", "price_compression"],
        "gap_go": [],
        "gap_fade": [],
        "open_drive": ["opening_range_high", "opening_range_low"],
        "orderbook_imbalance": ["bid_ask_ratio", "depth_imbalance", "spread"],
        "large_absorption": ["large_trade_detection", "bid_ask_ratio"],
        "overextension": ["intraday_drop_pct"],
        "rsi_reversal": ["selling_exhaustion"],
        "crash_rebound": ["intraday_drop_pct", "selling_exhaustion"],
        "tdnet_event": [],  # TDnet は real データ
        "earnings_momentum": ["earnings_surprise_pct", "revenue_growth"],
        "catalyst_initial": ["news_sentiment", "historical_catalyst_response"],
        "vwap_bounce": ["time_below_vwap", "volume_at_reclaim"],
    }

    @classmethod
    def get_proxy_features(cls) -> list[str]:
        """Return the list of proxy (simulated) feature names."""
        return sorted(cls.PROXY_FEATURES)

    @classmethod
    def get_proxy_usage_rate(cls, strategy_name: str) -> float:
        """Return the proxy dependency rate (0-1) for a strategy.

        Higher values indicate greater reliance on simulated features,
        meaning lower evaluation reliability.
        """
        deps = cls.STRATEGY_PROXY_DEPS.get(strategy_name, [])
        if not deps:
            return 0.0
        proxy_count = sum(1 for d in deps if d in cls.PROXY_FEATURES)
        return proxy_count / max(len(deps), 1)

    @classmethod
    def get_all_proxy_usage_rates(cls) -> dict[str, float]:
        """Return proxy_usage_rate for all known strategies."""
        return {
            name: cls.get_proxy_usage_rate(name)
            for name in cls.STRATEGY_PROXY_DEPS
        }

    @classmethod
    def get_proxy_penalty(cls, strategy_name: str) -> float:
        """Return a confidence penalty (0 to 0.15) based on proxy dependency.

        Strategies that rely heavily on proxy features get a larger penalty
        to prevent overconfidence in their signals.

        Active 戦略 (vwap_reclaim) は主力なのでペナルティ上限を 0.05 に抑える。
        他戦略は rate * 0.15 (最大 0.15)。
        """
        from strategies.registry import StrategyRegistry
        rate = cls.get_proxy_usage_rate(strategy_name)
        status = StrategyRegistry.STRATEGY_STATUS.get(strategy_name, "off")
        if status == "active":
            # 主戦略は proxy ペナルティを軽減 (最大 0.05)
            return round(rate * 0.05, 3)
        # 0.0 → no penalty, 1.0 → -0.15
        return round(rate * 0.15, 3)

    def _calc_proxy_usage_rate(self, features: dict) -> float:
        """Calculate what fraction of non-None features are proxy."""
        if not features:
            return 0.0
        total = 0
        proxy_count = 0
        for k, v in features.items():
            if k.startswith("_"):
                continue
            if v is None:
                continue
            total += 1
            if k in self.PROXY_FEATURES:
                proxy_count += 1
        return round(proxy_count / max(total, 1), 3)

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case column names and validate required columns."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        # Drop duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]
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
