"""
Unified market data interface.

Provides a single entry point for fetching prices, calculating technical
indicators, and simulating paper-trading prices for testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from data_sources.jquants import JQuantsClient


class MarketCondition(str, Enum):
    """Overall market regime classification."""

    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class MarketSnapshot:
    """Point-in-time market state."""

    condition: MarketCondition
    nikkei225_change_pct: float = 0.0
    topix_change_pct: float = 0.0
    advancing: int = 0
    declining: int = 0
    new_highs: int = 0
    new_lows: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class MarketDataProvider:
    """
    Unified market data provider.

    Wraps JQuantsClient and adds:
    - Technical indicator calculation
    - VWAP computation
    - Market condition assessment
    - Paper trading price simulation

    Usage:
        provider = MarketDataProvider(jquants_client=client)
        price = await provider.get_current_price("7203")
        features = await provider.calculate_features("7203")
    """

    def __init__(
        self,
        jquants_client: Optional[JQuantsClient] = None,
        paper_mode: bool = True,
    ) -> None:
        self._jquants = jquants_client
        self._paper_mode = paper_mode
        # In-memory cache for current session
        self._price_cache: dict[str, float] = {}
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}
        self._last_close_cache: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Price access
    # ------------------------------------------------------------------

    async def get_current_price(self, ticker: str) -> float:
        """
        Get the current (or most recent) price for a ticker.

        In paper mode, returns a simulated price based on the last known
        close with a small random walk applied.

        Args:
            ticker: Stock code (e.g. "7203").

        Returns:
            Current price as a float.
        """
        if self._paper_mode:
            return self._simulate_price(ticker)

        # Use cached price if very recent
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        if not self._jquants:
            raise RuntimeError("JQuantsClient not configured and not in paper mode")

        # Fetch latest daily quote
        today = date.today().isoformat()
        week_ago = (date.today() - timedelta(days=7)).isoformat()
        quotes = await self._jquants.get_prices_daily(ticker, week_ago, today)

        if not quotes:
            logger.warning("No price data available for {}", ticker)
            raise ValueError(f"No price data for {ticker}")

        latest = quotes[-1]
        price = float(latest.get("AdjustmentClose") or latest.get("Close", 0))
        self._price_cache[ticker] = price
        self._last_close_cache[ticker] = price
        return price

    async def get_ohlcv(
        self,
        ticker: str,
        interval: str = "daily",
        periods: int = 60,
    ) -> pd.DataFrame:
        """
        Get OHLCV data as a DataFrame.

        Args:
            ticker: Stock code.
            interval: "daily" or "intraday".
            periods: Number of periods to fetch.

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume.
        """
        cache_key = f"{ticker}_{interval}_{periods}"
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]

        if self._paper_mode and not self._jquants:
            return self._generate_paper_ohlcv(ticker, periods)

        if not self._jquants:
            raise RuntimeError("JQuantsClient not configured")

        if interval == "daily":
            to_date = date.today().isoformat()
            # Fetch extra days to account for weekends/holidays
            from_date = (date.today() - timedelta(days=int(periods * 1.6))).isoformat()
            quotes = await self._jquants.get_prices_daily(ticker, from_date, to_date)

            if not quotes:
                logger.warning("No OHLCV data for {}", ticker)
                return pd.DataFrame()

            df = pd.DataFrame(quotes)
            df = df.rename(columns={
                "Date": "Date",
                "AdjustmentOpen": "Open",
                "AdjustmentHigh": "High",
                "AdjustmentLow": "Low",
                "AdjustmentClose": "Close",
                "AdjustmentVolume": "Volume",
            })

            # Fallback to non-adjusted if adjusted columns are missing
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    df[col] = df.get(col, 0)

            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].tail(periods)
            df = df.reset_index(drop=True)

            for col in ["Open", "High", "Low", "Close", "Volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        elif interval == "intraday":
            today_str = date.today().isoformat()
            quotes = await self._jquants.get_prices_intraday(ticker, today_str)
            if not quotes:
                return pd.DataFrame()
            df = pd.DataFrame(quotes).tail(periods)
        else:
            raise ValueError(f"Unknown interval: {interval}")

        self._ohlcv_cache[cache_key] = df
        return df

    async def get_vwap(self, ticker: str) -> float:
        """
        Calculate VWAP (Volume Weighted Average Price) for today.

        Uses intraday data if available, otherwise estimates from daily data.

        Args:
            ticker: Stock code.

        Returns:
            VWAP as a float.
        """
        # Try intraday first
        if self._jquants and not self._paper_mode:
            try:
                today_str = date.today().isoformat()
                quotes = await self._jquants.get_prices_intraday(ticker, today_str)
                if quotes:
                    df = pd.DataFrame(quotes)
                    if "Close" in df.columns and "Volume" in df.columns:
                        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                        df = df.dropna(subset=["Close", "Volume"])
                        if not df.empty and df["Volume"].sum() > 0:
                            vwap = (df["Close"] * df["Volume"]).sum() / df["Volume"].sum()
                            return float(vwap)
            except Exception as e:
                logger.debug("Intraday VWAP calculation failed for {}: {}", ticker, e)

        # Fallback: estimate from daily OHLCV (typical price)
        ohlcv = await self.get_ohlcv(ticker, "daily", 1)
        if ohlcv.empty:
            return await self.get_current_price(ticker)

        row = ohlcv.iloc[-1]
        typical_price = (float(row["High"]) + float(row["Low"]) + float(row["Close"])) / 3.0
        return typical_price

    async def get_market_condition(self) -> MarketCondition:
        """
        Assess the current overall market condition.

        Uses Nikkei 225 proxy (1321 or market index data) to determine
        the market regime.

        Returns:
            MarketCondition enum value.
        """
        try:
            # Use Nikkei 225 ETF (1321) as proxy
            ohlcv = await self.get_ohlcv("1321", "daily", 20)
            if ohlcv.empty or len(ohlcv) < 5:
                return MarketCondition.NEUTRAL

            closes = ohlcv["Close"].astype(float)

            # Calculate short-term return
            ret_5d = (closes.iloc[-1] / closes.iloc[-5] - 1) * 100
            # Calculate volatility
            daily_returns = closes.pct_change().dropna()
            volatility = daily_returns.std() * 100

            if volatility > 2.5:
                return MarketCondition.HIGH_VOLATILITY
            elif ret_5d > 3:
                return MarketCondition.STRONG_BULL
            elif ret_5d > 1:
                return MarketCondition.BULL
            elif ret_5d < -3:
                return MarketCondition.STRONG_BEAR
            elif ret_5d < -1:
                return MarketCondition.BEAR
            else:
                return MarketCondition.NEUTRAL

        except Exception as e:
            logger.warning("Failed to assess market condition: {}", e)
            return MarketCondition.NEUTRAL

    # ------------------------------------------------------------------
    # Feature calculation
    # ------------------------------------------------------------------

    async def calculate_features(
        self,
        ticker: str,
        ohlcv: Optional[pd.DataFrame] = None,
    ) -> dict[str, Any]:
        """
        Calculate technical features for a given ticker.

        Features computed:
        - vwap: Volume Weighted Average Price
        - rsi_14: 14-period RSI
        - rsi_5: 5-period RSI
        - macd: MACD line value
        - macd_signal: MACD signal line
        - macd_histogram: MACD histogram
        - bb_upper: Bollinger Band upper (20, 2)
        - bb_middle: Bollinger Band middle
        - bb_lower: Bollinger Band lower
        - atr: 14-period Average True Range
        - volume_ratio: Current volume / 20-day average volume
        - gap_pct: Gap from previous close (%)
        - distance_from_vwap: (price - vwap) / vwap * 100
        - spread: Bid-ask spread (if available, else NaN)
        - sector_relative_strength: Relative strength vs sector (if available)

        Args:
            ticker: Stock code.
            ohlcv: Pre-fetched OHLCV DataFrame. If None, will be fetched.

        Returns:
            Dict mapping feature name to float value.
        """
        if ohlcv is None:
            ohlcv = await self.get_ohlcv(ticker, "daily", 60)

        if ohlcv.empty or len(ohlcv) < 20:
            logger.warning("Insufficient data to calculate features for {}", ticker)
            return {}

        closes = ohlcv["Close"].astype(float)
        highs = ohlcv["High"].astype(float)
        lows = ohlcv["Low"].astype(float)
        volumes = ohlcv["Volume"].astype(float)
        opens = ohlcv["Open"].astype(float)

        features: dict[str, Any] = {}

        # VWAP (daily approximation)
        if volumes.sum() > 0:
            typical = (highs + lows + closes) / 3
            features["vwap"] = float((typical * volumes).sum() / volumes.sum())
        else:
            features["vwap"] = float(closes.iloc[-1])

        # RSI
        features["rsi_14"] = self._compute_rsi(closes, 14)
        features["rsi_5"] = self._compute_rsi(closes, 5)

        # MACD (12, 26, 9)
        macd_line, signal_line, histogram = self._compute_macd(closes, 12, 26, 9)
        features["macd"] = macd_line
        features["macd_signal"] = signal_line
        features["macd_histogram"] = histogram

        # Bollinger Bands (20, 2)
        bb_mid = closes.rolling(20).mean().iloc[-1]
        bb_std = closes.rolling(20).std().iloc[-1]
        features["bb_upper"] = float(bb_mid + 2 * bb_std)
        features["bb_middle"] = float(bb_mid)
        features["bb_lower"] = float(bb_mid - 2 * bb_std)

        # ATR (14-period)
        features["atr"] = self._compute_atr(highs, lows, closes, 14)

        # Volume ratio (current vs 20-day average)
        avg_volume_20 = volumes.tail(20).mean()
        current_volume = volumes.iloc[-1]
        features["volume_ratio"] = (
            float(current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0
        )

        # Gap percentage from previous close
        if len(closes) >= 2:
            prev_close = float(closes.iloc[-2])
            current_open = float(opens.iloc[-1])
            features["gap_pct"] = (
                ((current_open - prev_close) / prev_close * 100) if prev_close > 0 else 0.0
            )
        else:
            features["gap_pct"] = 0.0

        # Distance from VWAP
        current_price = float(closes.iloc[-1])
        vwap = features["vwap"]
        features["distance_from_vwap"] = (
            ((current_price - vwap) / vwap * 100) if vwap > 0 else 0.0
        )

        # Spread (not available from daily data)
        features["spread"] = float("nan")

        # Sector relative strength (placeholder -- requires sector index data)
        features["sector_relative_strength"] = float("nan")

        return features

    # ------------------------------------------------------------------
    # Technical indicator calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(closes: pd.Series, period: int) -> float:
        """Compute RSI using exponential moving average of gains/losses."""
        if len(closes) < period + 1:
            return 50.0

        delta = closes.diff()
        gains = delta.clip(lower=0)
        losses = (-delta.clip(upper=0))

        avg_gain = gains.ewm(span=period, min_periods=period, adjust=False).mean()
        avg_loss = losses.ewm(span=period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        rsi = 100 - (100 / (1 + rs))
        value = rsi.iloc[-1]
        return float(value) if pd.notna(value) else 50.0

    @staticmethod
    def _compute_macd(
        closes: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        """Compute MACD line, signal line, and histogram."""
        if len(closes) < slow + signal:
            return 0.0, 0.0, 0.0

        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return (
            float(macd_line.iloc[-1]),
            float(signal_line.iloc[-1]),
            float(histogram.iloc[-1]),
        )

    @staticmethod
    def _compute_atr(
        highs: pd.Series,
        lows: pd.Series,
        closes: pd.Series,
        period: int = 14,
    ) -> float:
        """Compute Average True Range."""
        if len(closes) < period + 1:
            return 0.0

        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        value = atr.iloc[-1]
        return float(value) if pd.notna(value) else 0.0

    # ------------------------------------------------------------------
    # Paper trading price simulation
    # ------------------------------------------------------------------

    def _simulate_price(self, ticker: str) -> float:
        """
        Generate a simulated price using random walk around last known close.

        Uses geometric Brownian motion with small drift and volatility
        to produce realistic-looking price changes for paper trading.
        """
        if ticker in self._price_cache:
            base_price = self._price_cache[ticker]
        elif ticker in self._last_close_cache:
            base_price = self._last_close_cache[ticker]
        else:
            # Generate a plausible base price from ticker code
            # (deterministic seed so the same ticker always starts at the same price)
            rng = random.Random(int(ticker))
            base_price = rng.uniform(500, 5000)
            self._last_close_cache[ticker] = base_price

        # Random walk: small percentage change
        drift = 0.0001  # slight upward bias
        volatility = 0.001  # 0.1% per tick
        shock = random.gauss(drift, volatility)
        new_price = base_price * (1 + shock)

        # Round to nearest yen (for stocks over 1000) or 0.1 yen
        if new_price >= 1000:
            new_price = round(new_price)
        else:
            new_price = round(new_price, 1)

        self._price_cache[ticker] = new_price
        return new_price

    def _generate_paper_ohlcv(self, ticker: str, periods: int) -> pd.DataFrame:
        """Generate synthetic OHLCV data for paper trading testing."""
        rng = random.Random(int(ticker))
        base = rng.uniform(500, 5000)

        dates = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        price = base
        current_date = date.today() - timedelta(days=int(periods * 1.5))

        generated = 0
        while generated < periods:
            # Skip weekends
            if current_date.weekday() < 5:
                daily_return = rng.gauss(0.0002, 0.015)
                open_price = price
                close_price = price * (1 + daily_return)
                high_price = max(open_price, close_price) * (1 + abs(rng.gauss(0, 0.005)))
                low_price = min(open_price, close_price) * (1 - abs(rng.gauss(0, 0.005)))
                volume = int(rng.gauss(500000, 200000))
                volume = max(volume, 10000)

                dates.append(current_date.isoformat())
                opens.append(round(open_price, 1))
                highs.append(round(high_price, 1))
                lows.append(round(low_price, 1))
                closes.append(round(close_price, 1))
                volumes.append(volume)

                price = close_price
                generated += 1

            current_date += timedelta(days=1)

        df = pd.DataFrame({
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        })
        return df
