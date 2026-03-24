"""Base strategy class for all KabuAI-Daytrade strategies."""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from loguru import logger

from core.models import TradeSignal, StrategyConfig


class BaseStrategy(ABC):
    """Abstract base class that all trading strategies must extend."""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.strategy_name

    @abstractmethod
    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        """Scan a ticker for entry signals.

        Args:
            ticker: Stock ticker symbol (e.g. "7203" for Toyota).
            data: OHLCV DataFrame with columns open/high/low/close/volume.
            features: Pre-computed feature dict (indicators, orderbook stats, etc.).

        Returns:
            A TradeSignal if conditions are met, otherwise None.
        """
        pass

    @abstractmethod
    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        """Check if an open position should be exited.

        Args:
            position: The current Position object.
            current_data: Latest OHLCV DataFrame.
            features: Latest feature dict.

        Returns:
            Tuple of (should_exit, reason_string).
        """
        pass

    @abstractmethod
    def get_default_config(self) -> StrategyConfig:
        """Return the default StrategyConfig for this strategy."""
        pass

    def calculate_position_size(
        self, price: float, atr: float, capital: float
    ) -> int:
        """ATR-based position sizing for Japanese equities.

        Risk is capped at 1% of capital per trade.  Shares are rounded to
        the standard 100-share trading unit (単元株).  A hard cap of
        500,000 JPY notional per position is applied.
        """
        risk_per_trade = capital * 0.005  # 0.5% risk (1%→0.5%: 最大損失抑制)
        shares = int(risk_per_trade / (atr * 2))
        unit = 100  # Japanese stock trading unit
        shares = max(unit, (shares // unit) * unit)
        max_shares = int(250_000 / price / unit) * unit  # 500k→250k: POSITION_SIZE と統一
        return min(shares, max(unit, max_shares))

    # ------------------------------------------------------------------
    # Convenience helpers available to all strategies
    # ------------------------------------------------------------------

    def _validate_data(self, data: pd.DataFrame, min_rows: int = 5) -> bool:
        """Return True if data has enough rows for analysis."""
        if data is None or len(data) < min_rows:
            logger.debug(
                f"[{self.name}] Insufficient data rows: "
                f"{0 if data is None else len(data)} < {min_rows}"
            )
            return False
        return True

    def _validate_features(self, features: dict, required: list[str]) -> bool:
        """Return True if all required feature keys are present."""
        missing = [k for k in required if k not in features]
        if missing:
            logger.debug(
                f"[{self.name}] Missing features: {missing}"
            )
            return False
        return True
