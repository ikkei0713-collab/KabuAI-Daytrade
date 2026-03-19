"""Opening Range Breakout strategy (5分足ブレイク).

Defines the opening range as the high and low of the first 5-minute
candle.  Enters long on a break above the high or short on a break
below the low, with a 1.5x range-size target.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class ORBStrategy(BaseStrategy):
    """5分足ブレイク – Opening Range Breakout."""

    REQUIRED_FEATURES = [
        "opening_range_high",
        "opening_range_low",
        "volume_ratio",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="orb",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_volume_ratio": 1.5,
                "min_atr": 10.0,
                "target_range_multiple": 1.5,
                "min_range_yen": 5.0,
                "max_range_pct": 3.0,
                "breakout_buffer_yen": 1.0,
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=2):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        params = self.config.parameter_set
        or_high: float = features["opening_range_high"]
        or_low: float = features["opening_range_low"]
        volume_ratio: float = features["volume_ratio"]
        atr: float = features["atr"]

        or_size = or_high - or_low
        mid_price = (or_high + or_low) / 2.0
        or_pct = or_size / mid_price * 100 if mid_price > 0 else 0

        # Filters
        if volume_ratio < params.get("min_volume_ratio", 1.5):
            logger.debug(f"[orb] {ticker}: volume_ratio {volume_ratio:.2f} too low")
            return None
        if atr < params.get("min_atr", 10.0):
            logger.debug(f"[orb] {ticker}: ATR {atr:.1f} below threshold")
            return None
        if or_size < params.get("min_range_yen", 5.0):
            logger.debug(f"[orb] {ticker}: range {or_size:.1f} yen too narrow")
            return None
        if or_pct > params.get("max_range_pct", 3.0):
            logger.debug(f"[orb] {ticker}: range {or_pct:.2f}% too wide")
            return None

        # Determine breakout direction from latest bar
        latest = data.iloc[-1]
        current_close = float(latest["close"])
        current_high = float(latest["high"])
        current_low = float(latest["low"])
        buffer = params.get("breakout_buffer_yen", 1.0)
        target_mult = params.get("target_range_multiple", 1.5)

        direction = None
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0

        if current_close > or_high + buffer:
            direction = "long"
            entry_price = or_high + buffer
            stop_price = or_low - buffer
            target_price = entry_price + or_size * target_mult
        elif current_close < or_low - buffer:
            direction = "short"
            entry_price = or_low - buffer
            stop_price = or_high + buffer
            target_price = entry_price - or_size * target_mult

        if direction is None:
            return None

        # Confirm with candle body
        body = abs(float(latest["close"]) - float(latest["open"]))
        candle_range = float(latest["high"]) - float(latest["low"])
        if candle_range > 0 and body / candle_range < 0.4:
            logger.debug(f"[orb] {ticker}: breakout candle has weak body ratio")
            return None

        # Confidence
        confidence = 0.50
        if volume_ratio > 2.5:
            confidence += 0.10
        if or_pct > 0.5:
            confidence += 0.05
        if body / candle_range > 0.7 if candle_range > 0 else False:
            confidence += 0.10
        if atr > 20:
            confidence += 0.05
        confidence = min(confidence, 0.90)

        shares = self.calculate_position_size(entry_price, atr, 10_000_000)

        signal = TradeSignal(
            ticker=ticker,
            direction=direction,
            strategy_name=self.name,
            entry_price=round(entry_price, 1),
            stop_loss=round(stop_price, 1),
            take_profit=round(target_price, 1),
            confidence=round(confidence, 2),
            entry_reason=(
                f"5分足ブレイク{'↑' if direction == 'long' else '↓'}: "
                f"OR [{or_low:.0f}-{or_high:.0f}] ({or_size:.0f}円), "
                f"出来高比{volume_ratio:.1f}x, ATR={atr:.0f}"
            ),
            features_snapshot=features,
        )
        logger.info(f"[orb] SIGNAL {ticker} {direction} entry={entry_price:.0f}")
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        or_high = features.get("opening_range_high", 0)
        or_low = features.get("opening_range_low", 0)
        or_mid = (or_high + or_low) / 2.0 if or_high and or_low else 0

        # Stop loss
        if position.direction == "long" and current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"
        if position.direction == "short" and current_price >= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if position.direction == "long" and current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"
        if position.direction == "short" and current_price <= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        # Failed breakout: price re-enters range
        if position.direction == "long" and or_mid > 0 and current_price < or_mid:
            return True, f"ブレイクアウト失敗 – レンジ内回帰 ({current_price:.0f})"
        if position.direction == "short" and or_mid > 0 and current_price > or_mid:
            return True, f"ブレイクアウト失敗 – レンジ内回帰 ({current_price:.0f})"

        # Time: ORB works best in first 30 min
        if position.holding_minutes > 30:
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            if position.direction == "short":
                pnl_pct = -pnl_pct
            if pnl_pct < 0.3:
                return True, "時間切れ30分"

        return False, ""
