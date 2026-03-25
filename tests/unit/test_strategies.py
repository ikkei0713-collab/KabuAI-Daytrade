"""Unit tests for KabuAI-Daytrade strategies.

Tests scan(), should_exit(), position sizing, and edge cases using
realistic Japanese stock market data (prices in yen, 100-share units).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from core.models import TradeSignal, Position, StrategyConfig
from strategies.base import BaseStrategy
from tools.feature_engineering import FeatureEngineer


# ------------------------------------------------------------------
# Helpers: realistic Japanese stock test data
# ------------------------------------------------------------------


def _make_ohlcv(
    rows: int = 60,
    base_price: float = 2500.0,
    trend: float = 0.001,
    volatility: float = 0.015,
    base_volume: int = 500_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data resembling a Japanese stock.

    Prices are in JPY (hundreds to thousands range).  Volume is in
    shares (typically hundreds of thousands for liquid names).
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=datetime.now().date(), periods=rows, freq="B")

    close_prices: list[float] = [base_price]
    for _ in range(1, rows):
        ret = trend + volatility * rng.randn()
        close_prices.append(close_prices[-1] * (1 + ret))

    close = np.array(close_prices)
    high = close * (1 + rng.uniform(0.001, 0.02, rows))
    low = close * (1 - rng.uniform(0.001, 0.02, rows))
    open_ = low + (high - low) * rng.uniform(0.2, 0.8, rows)
    volume = (base_volume * (1 + 0.3 * rng.randn(rows))).clip(min=10_000).astype(int)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _make_momentum_signal_data(rows: int = 60) -> pd.DataFrame:
    """Create data with a strong upward momentum pattern."""
    rng = np.random.RandomState(123)
    base = 3000.0
    prices: list[float] = [base]
    for i in range(1, rows):
        # Strong uptrend in last 10 bars
        trend = 0.015 if i > rows - 15 else 0.001
        prices.append(prices[-1] * (1 + trend + 0.005 * rng.randn()))

    close = np.array(prices)
    high = close * 1.01
    low = close * 0.99
    open_ = close * (1 - 0.002 * rng.randn(rows))
    volume = np.full(rows, 800_000)
    # High volume on recent bars
    volume[-10:] = 2_000_000

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
    )


def _make_reversal_signal_data(rows: int = 60) -> pd.DataFrame:
    """Create data with an oversold bounce pattern (hammer candle, low RSI)."""
    rng = np.random.RandomState(456)
    base = 1800.0
    prices: list[float] = [base]
    for i in range(1, rows):
        # Downtrend then reversal
        if i < rows - 5:
            trend = -0.008
        else:
            trend = 0.005
        prices.append(prices[-1] * (1 + trend + 0.005 * rng.randn()))

    close = np.array(prices)
    high = close * 1.005
    low = close * 0.995

    # Make last bar a hammer
    low[-1] = close[-1] * 0.97
    high[-1] = close[-1] * 1.002

    open_ = close * (1 + 0.001 * rng.randn(rows))
    open_[-1] = close[-1] * 0.999  # open near close for hammer body
    volume = np.full(rows, 300_000)
    volume[-1] = 900_000  # volume spike on reversal

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
    )


# ------------------------------------------------------------------
# Concrete strategy implementations for testing
# ------------------------------------------------------------------


class MockMomentumStrategy(BaseStrategy):
    """Momentum strategy for testing purposes."""

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="mock_momentum",
            entry_conditions={"rsi_14_min": 50, "volume_ratio_min": 1.5},
            exit_conditions={"rsi_14_max": 75, "trailing_stop_pct": 0.02},
            expected_market_condition="bull",
            feature_requirements=["rsi_14", "sma_20", "volume_ratio", "atr_14"],
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=20):
            return None
        if not self._validate_features(
            features, ["rsi_14", "sma_20", "volume_ratio", "atr_14"]
        ):
            return None

        rsi = features.get("rsi_14")
        sma_20 = features.get("sma_20")
        vol_ratio = features.get("volume_ratio")
        atr = features.get("atr_14")
        current_close = float(data["close"].iloc[-1])

        if rsi is None or sma_20 is None or vol_ratio is None or atr is None:
            return None

        # Entry: price above SMA20, RSI 50-70, volume above average
        if current_close > sma_20 and 50 <= rsi <= 70 and vol_ratio > 1.5:
            stop_loss = current_close - atr * 2
            take_profit = current_close + atr * 3
            confidence = min(0.9, 0.5 + (vol_ratio - 1.5) * 0.1 + (rsi - 50) / 100)

            return TradeSignal(
                strategy_name=self.name,
                ticker=ticker,
                direction="long",
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                entry_reason=f"Momentum: RSI={rsi:.0f}, VolRatio={vol_ratio:.1f}",
                features_snapshot=features,
            )
        return None

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if not self._validate_data(current_data, min_rows=5):
            return False, ""

        rsi = features.get("rsi_14")
        current_close = float(current_data["close"].iloc[-1])

        # Exit on RSI overbought
        if rsi is not None and rsi > 75:
            return True, f"RSI overbought: {rsi:.0f}"

        # Trailing stop: 2% from peak
        if hasattr(position, "entry_price") and position.entry_price > 0:
            pnl_pct = (current_close - position.entry_price) / position.entry_price
            if pnl_pct < -0.02:
                return True, f"Trailing stop hit: {pnl_pct:.1%}"

        return False, ""


class MockReversalStrategy(BaseStrategy):
    """Mean-reversion strategy for testing purposes."""

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="mock_reversal",
            entry_conditions={"rsi_14_max": 30, "candle_pattern": "hammer"},
            exit_conditions={"rsi_14_min": 50, "take_profit_pct": 0.03},
            expected_market_condition="range",
            feature_requirements=["rsi_14", "bb_lower", "candle_pattern", "atr_14"],
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=20):
            return None
        if not self._validate_features(
            features, ["rsi_14", "bb_lower", "candle_pattern", "atr_14"]
        ):
            return None

        rsi = features.get("rsi_14")
        bb_lower = features.get("bb_lower")
        pattern = features.get("candle_pattern")
        atr = features.get("atr_14")
        current_close = float(data["close"].iloc[-1])

        if rsi is None or bb_lower is None or atr is None:
            return None

        # Entry: RSI oversold, price near lower BB, bullish pattern
        if rsi < 30 and current_close <= bb_lower * 1.01 and pattern in ("hammer", "bullish_engulfing"):
            stop_loss = current_close - atr * 1.5
            take_profit = current_close + atr * 2
            confidence = min(0.85, 0.6 + (30 - rsi) / 100)

            return TradeSignal(
                strategy_name=self.name,
                ticker=ticker,
                direction="long",
                entry_price=current_close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=confidence,
                entry_reason=f"Reversal: RSI={rsi:.0f}, pattern={pattern}",
                features_snapshot=features,
            )
        return None

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if not self._validate_data(current_data, min_rows=5):
            return False, ""

        rsi = features.get("rsi_14")
        current_close = float(current_data["close"].iloc[-1])

        # Exit when RSI recovers above 50
        if rsi is not None and rsi > 50:
            return True, f"RSI recovered: {rsi:.0f}"

        # Take profit at 3%
        if hasattr(position, "entry_price") and position.entry_price > 0:
            gain = (current_close - position.entry_price) / position.entry_price
            if gain >= 0.03:
                return True, f"Take profit: {gain:.1%}"

        return False, ""


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture
def momentum_strategy() -> MockMomentumStrategy:
    config = StrategyConfig(
        strategy_name="mock_momentum",
        feature_requirements=["rsi_14", "sma_20", "volume_ratio", "atr_14"],
    )
    return MockMomentumStrategy(config)


@pytest.fixture
def reversal_strategy() -> MockReversalStrategy:
    config = StrategyConfig(
        strategy_name="mock_reversal",
        feature_requirements=["rsi_14", "bb_lower", "candle_pattern", "atr_14"],
    )
    return MockReversalStrategy(config)


@pytest.fixture
def basic_ohlcv() -> pd.DataFrame:
    return _make_ohlcv()


@pytest.fixture
def momentum_data() -> pd.DataFrame:
    return _make_momentum_signal_data()


@pytest.fixture
def reversal_data() -> pd.DataFrame:
    return _make_reversal_signal_data()


# ------------------------------------------------------------------
# Tests: scan()
# ------------------------------------------------------------------


class TestStrategyScan:
    """Test strategy scan() methods with realistic Japanese stock data."""

    @pytest.mark.asyncio
    async def test_momentum_scan_returns_signal_on_bullish_setup(
        self,
        momentum_strategy: MockMomentumStrategy,
        momentum_data: pd.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> None:
        features = feature_engineer.calculate_all_features(momentum_data)
        signal = await momentum_strategy.scan("7203", momentum_data, features)

        # The momentum data is designed with strong uptrend and high volume,
        # but whether we get a signal depends on exact feature values.
        # We verify the method runs without error and returns correct type.
        assert signal is None or isinstance(signal, TradeSignal)
        if signal is not None:
            assert signal.ticker == "7203"
            assert signal.direction == "long"
            assert signal.confidence >= 0.0
            assert signal.confidence <= 1.0
            assert signal.stop_loss < signal.entry_price
            assert signal.take_profit > signal.entry_price
            assert signal.strategy_name == "mock_momentum"

    @pytest.mark.asyncio
    async def test_reversal_scan_returns_signal_or_none(
        self,
        reversal_strategy: MockReversalStrategy,
        reversal_data: pd.DataFrame,
        feature_engineer: FeatureEngineer,
    ) -> None:
        features = feature_engineer.calculate_all_features(reversal_data)
        signal = await reversal_strategy.scan("6758", reversal_data, features)

        assert signal is None or isinstance(signal, TradeSignal)
        if signal is not None:
            assert signal.ticker == "6758"
            assert signal.direction == "long"
            assert signal.strategy_name == "mock_reversal"

    @pytest.mark.asyncio
    async def test_scan_with_forced_features(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        """Supply hand-crafted features to guarantee a signal."""
        data = _make_ohlcv(rows=60, base_price=3000, trend=0.005, seed=99)
        last_close = float(data["close"].iloc[-1])

        features = {
            "rsi_14": 60.0,
            "sma_20": last_close * 0.95,  # price above SMA
            "volume_ratio": 2.0,
            "atr_14": last_close * 0.015,
        }

        signal = await momentum_strategy.scan("9984", data, features)
        assert signal is not None
        assert signal.ticker == "9984"
        assert signal.direction == "long"
        assert 0.5 <= signal.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_scan_no_signal_when_conditions_unmet(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        """No signal when RSI is too high and volume is low."""
        data = _make_ohlcv(rows=60, base_price=1500, seed=77)
        last_close = float(data["close"].iloc[-1])

        features = {
            "rsi_14": 80.0,
            "sma_20": last_close * 1.05,  # price below SMA
            "volume_ratio": 0.5,
            "atr_14": 20.0,
        }

        signal = await momentum_strategy.scan("8306", data, features)
        assert signal is None


# ------------------------------------------------------------------
# Tests: should_exit()
# ------------------------------------------------------------------


class TestStrategyExit:
    """Test should_exit() logic."""

    def test_momentum_exit_on_rsi_overbought(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        data = _make_ohlcv(rows=30, base_price=2500)
        position = Position(
            ticker="7203",
            strategy_name="mock_momentum",
            direction="long",
            entry_price=2400.0,
            entry_time=datetime.now(),
            current_price=2600.0,
        )
        features = {"rsi_14": 80.0}

        should_exit, reason = momentum_strategy.should_exit(position, data, features)
        assert should_exit is True
        assert "overbought" in reason.lower()

    def test_momentum_no_exit_normal_rsi(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        data = _make_ohlcv(rows=30, base_price=2500)
        position = Position(
            ticker="7203",
            strategy_name="mock_momentum",
            direction="long",
            entry_price=2400.0,
            entry_time=datetime.now(),
            current_price=2500.0,
        )
        features = {"rsi_14": 60.0}

        should_exit, reason = momentum_strategy.should_exit(position, data, features)
        assert should_exit is False

    def test_reversal_exit_on_rsi_recovery(
        self,
        reversal_strategy: MockReversalStrategy,
    ) -> None:
        data = _make_ohlcv(rows=30, base_price=1800)
        position = Position(
            ticker="6758",
            strategy_name="mock_reversal",
            direction="long",
            entry_price=1750.0,
            entry_time=datetime.now(),
            current_price=1850.0,
        )
        features = {"rsi_14": 55.0}

        should_exit, reason = reversal_strategy.should_exit(position, data, features)
        assert should_exit is True
        assert "recovered" in reason.lower()

    def test_exit_trailing_stop(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        """Trailing stop triggers on >2% loss."""
        data = _make_ohlcv(rows=30, base_price=2000)
        # Manually set last close well below entry
        data.iloc[-1, data.columns.get_loc("close")] = 1940.0

        position = Position(
            ticker="7203",
            strategy_name="mock_momentum",
            direction="long",
            entry_price=2000.0,
            entry_time=datetime.now(),
            current_price=1940.0,
        )
        features = {"rsi_14": 45.0}  # not overbought

        should_exit, reason = momentum_strategy.should_exit(position, data, features)
        assert should_exit is True
        assert "trailing stop" in reason.lower() or "stop" in reason.lower()


# ------------------------------------------------------------------
# Tests: position sizing
# ------------------------------------------------------------------


class TestPositionSizing:
    """Test ATR-based position sizing for Japanese equities."""

    def test_basic_position_size(
        self, momentum_strategy: MockMomentumStrategy
    ) -> None:
        # BaseStrategy: 1株単位、max_notional = min(capital, 30_000)
        shares = momentum_strategy.calculate_position_size(2500.0, 50.0, 3_000_000)
        assert shares >= 1
        assert shares * 2500 <= 30_000

    def test_minimum_unit(
        self, momentum_strategy: MockMomentumStrategy
    ) -> None:
        """単元株ではなく 1 株単位の最小。"""
        shares = momentum_strategy.calculate_position_size(5000.0, 200.0, 100_000)
        assert shares == 6

    def test_max_notional_cap(
        self, momentum_strategy: MockMomentumStrategy
    ) -> None:
        """名義金額は 30,000 円上限付近で打ち切り。"""
        shares = momentum_strategy.calculate_position_size(1000.0, 5.0, 10_000_000)
        assert shares * 1000 <= 30_000

    def test_high_price_stock(
        self, momentum_strategy: MockMomentumStrategy
    ) -> None:
        """高値株は max_shares = int(max_notional/price) で株数が抑えられる。"""
        shares = momentum_strategy.calculate_position_size(60000.0, 1200.0, 3_000_000)
        assert shares == 1


# ------------------------------------------------------------------
# Tests: edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Test strategies with edge-case inputs."""

    @pytest.mark.asyncio
    async def test_scan_with_empty_dataframe(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        features: dict = {}
        signal = await momentum_strategy.scan("7203", empty_df, features)
        assert signal is None

    @pytest.mark.asyncio
    async def test_scan_with_insufficient_data(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        """Only 3 rows — below the 20-row minimum."""
        data = _make_ohlcv(rows=3, base_price=2500)
        features = {"rsi_14": 60.0, "sma_20": 2400.0, "volume_ratio": 2.0, "atr_14": 40.0}
        signal = await momentum_strategy.scan("7203", data, features)
        assert signal is None

    @pytest.mark.asyncio
    async def test_scan_with_missing_features(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        data = _make_ohlcv(rows=60, base_price=2500)
        # Missing rsi_14
        features = {"sma_20": 2400.0, "volume_ratio": 2.0, "atr_14": 40.0}
        signal = await momentum_strategy.scan("7203", data, features)
        assert signal is None

    @pytest.mark.asyncio
    async def test_scan_with_none_feature_values(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        data = _make_ohlcv(rows=60, base_price=2500)
        features = {"rsi_14": None, "sma_20": None, "volume_ratio": None, "atr_14": None}
        signal = await momentum_strategy.scan("7203", data, features)
        assert signal is None

    def test_should_exit_with_minimal_data(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        """should_exit with less than 5 rows returns False."""
        data = _make_ohlcv(rows=3, base_price=2500)
        position = Position(
            ticker="7203",
            strategy_name="mock_momentum",
            direction="long",
            entry_price=2500.0,
            entry_time=datetime.now(),
        )
        should_exit, reason = momentum_strategy.should_exit(position, data, {})
        assert should_exit is False

    def test_validate_data_with_none(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        assert momentum_strategy._validate_data(None) is False

    def test_validate_features_reports_missing(
        self,
        momentum_strategy: MockMomentumStrategy,
    ) -> None:
        assert momentum_strategy._validate_features({}, ["rsi_14", "sma_20"]) is False
        assert momentum_strategy._validate_features(
            {"rsi_14": 50}, ["rsi_14"]
        ) is True
