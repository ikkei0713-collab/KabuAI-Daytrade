"""Market regime detection for KabuAI-Daytrade.

Classifies the current market environment as bull / bear / range / volatile
by analysing index trends, volatility, and breadth.  Also provides sector
momentum and a simple risk-on / risk-off flag.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger


# Nikkei 225 sector mapping (simplified - sector code to readable name)
_SECTOR_NAMES: dict[str, str] = {
    "0050": "水産・農林業",
    "1050": "鉱業",
    "2050": "建設業",
    "3050": "食料品",
    "3100": "繊維製品",
    "3150": "パルプ・紙",
    "3200": "化学",
    "3250": "医薬品",
    "3300": "石油・石炭製品",
    "3350": "ゴム製品",
    "3400": "ガラス・土石製品",
    "3450": "鉄鋼",
    "3500": "非鉄金属",
    "3550": "金属製品",
    "3600": "機械",
    "3650": "電気機器",
    "3700": "輸送用機器",
    "3750": "精密機器",
    "3800": "その他製品",
    "4050": "電気・ガス業",
    "5050": "陸運業",
    "5100": "海運業",
    "5150": "空運業",
    "5200": "倉庫・運輸関連業",
    "5250": "情報・通信業",
    "6050": "卸売業",
    "6100": "小売業",
    "7050": "銀行業",
    "7100": "証券、商品先物取引業",
    "7150": "保険業",
    "7200": "その他金融業",
    "8050": "不動産業",
    "9050": "サービス業",
}

MarketRegime = Literal["bull", "bear", "range", "volatile"]


class MarketRegimeDetector:
    """Detect the prevailing market regime.

    The detector analyses an index DataFrame (Nikkei 225 daily OHLCV)
    and computes trend, volatility, and breadth metrics to classify the
    market into one of four regimes.

    Usage::

        detector = MarketRegimeDetector()
        regime = detector.detect_regime(nikkei_df)
    """

    # Thresholds (configurable via constructor)
    def __init__(
        self,
        trend_sma_short: int = 20,
        trend_sma_long: int = 50,
        volatility_window: int = 20,
        high_volatility_pct: float = 0.02,
        range_band_pct: float = 0.03,
    ) -> None:
        self.trend_sma_short = trend_sma_short
        self.trend_sma_long = trend_sma_long
        self.volatility_window = volatility_window
        self.high_volatility_pct = high_volatility_pct
        self.range_band_pct = range_band_pct

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def detect_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """Classify the current market regime.

        Args:
            market_data: OHLCV DataFrame for the market index (e.g.
                Nikkei 225).  Must contain ``close``, ``high``, ``low``
                columns.

        Returns:
            One of ``"bull"``, ``"bear"``, ``"range"``, ``"volatile"``.
        """
        df = self._normalise(market_data)
        if df.empty or len(df) < self.trend_sma_long:
            logger.warning(
                "Not enough data for regime detection ({} rows, need {})",
                len(df),
                self.trend_sma_long,
            )
            return "range"

        close = df["close"]

        # --- Trend signals ---
        sma_short = close.rolling(self.trend_sma_short).mean()
        sma_long = close.rolling(self.trend_sma_long).mean()
        trend_bull = sma_short.iloc[-1] > sma_long.iloc[-1]
        trend_bear = sma_short.iloc[-1] < sma_long.iloc[-1]

        # Slope of the short SMA over last 5 bars
        slope = (sma_short.iloc[-1] - sma_short.iloc[-6]) / sma_short.iloc[-6] if len(sma_short) > 5 else 0.0

        # --- Volatility ---
        daily_returns = close.pct_change().dropna()
        volatility = daily_returns.tail(self.volatility_window).std()
        is_volatile = volatility > self.high_volatility_pct

        # --- Range detection ---
        recent_high = close.tail(self.volatility_window).max()
        recent_low = close.tail(self.volatility_window).min()
        midpoint = (recent_high + recent_low) / 2.0
        band = (recent_high - recent_low) / midpoint if midpoint > 0 else 0.0
        is_range = band < self.range_band_pct

        # --- Classification ---
        if is_volatile and abs(slope) > 0.02:
            regime: MarketRegime = "volatile"
        elif trend_bull and slope > 0.005 and not is_range:
            regime = "bull"
        elif trend_bear and slope < -0.005 and not is_range:
            regime = "bear"
        else:
            regime = "range"

        logger.info(
            "Market regime: {} (slope={:.4f}, vol={:.4f}, band={:.4f})",
            regime,
            slope,
            volatility,
            band,
        )
        return regime

    # ------------------------------------------------------------------
    # Sector momentum
    # ------------------------------------------------------------------

    def get_sector_momentum(
        self,
        sector_data: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, float]:
        """Compute sector momentum scores.

        Args:
            sector_data: Mapping of sector code to OHLCV DataFrame.
                If ``None``, returns an empty dict (data must be
                supplied by the caller from J-Quants or another source).

        Returns:
            ``{sector_name: momentum_score}`` where the score is the
            20-day rate of change as a percentage.
        """
        if not sector_data:
            return {}

        momentum: dict[str, float] = {}
        for code, df in sector_data.items():
            df = self._normalise(df)
            if df.empty or len(df) < 21:
                continue
            close = df["close"]
            roc20 = (close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]
            name = _SECTOR_NAMES.get(code, code)
            momentum[name] = round(float(roc20), 4)

        # Sort strongest first
        return dict(sorted(momentum.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Risk on / off
    # ------------------------------------------------------------------

    def is_risk_on(
        self,
        market_data: pd.DataFrame,
        usd_jpy: float | None = None,
    ) -> bool:
        """Quick risk-on / risk-off determination.

        Risk is considered *on* when:
        - Market regime is bull or range, AND
        - Volatility is not extreme, AND
        - (Optionally) USD/JPY is not weakening sharply.

        Args:
            market_data: Index OHLCV DataFrame.
            usd_jpy: Current USD/JPY rate (optional).

        Returns:
            ``True`` if risk-on conditions are met.
        """
        regime = self.detect_regime(market_data)

        if regime in ("bear", "volatile"):
            return False

        # Additional USD/JPY guard: if yen is strengthening below 140,
        # foreign sellers may push the market down
        if usd_jpy is not None and usd_jpy < 140.0:
            logger.info("Risk-off: USD/JPY={:.2f} below 140 threshold", usd_jpy)
            return False

        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(df: pd.DataFrame) -> pd.DataFrame:
        """Lower-case columns, validate presence of 'close'."""
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            logger.error("Market data missing 'close' column")
            return pd.DataFrame()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df.dropna(subset=["close"], inplace=True)
        return df
