"""VWAP Bounce strategy.

Enters when a stock in an uptrend pulls back to VWAP and bounces
with a bullish candle pattern and supportive volume.  Stop is placed
0.5 ATR below VWAP; target is the previous high or 1x ATR.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class VWAPBounceStrategy(BaseStrategy):
    """VWAPバウンス – buy on VWAP support bounce."""

    REQUIRED_FEATURES = [
        "vwap",
        "vwap_touches_today",
        "trend_direction",
        "volume_ratio",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="vwap_bounce",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "max_vwap_touches": 4,
                "min_vwap_touches": 1,
                "trend_direction_min": 0.3,
                "min_volume_ratio": 1.2,
                "target_atr_multiple": 1.0,
                "stop_atr_below_vwap": 0.5,
                "bounce_proximity_pct": 0.3,
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=3):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        params = self.config.parameter_set
        vwap: float = features["vwap"]
        touches: int = features["vwap_touches_today"]
        trend: float = features["trend_direction"]
        volume_ratio: float = features["volume_ratio"]
        atr: float = features["atr"]

        # Must be in an uptrend
        if trend < params.get("trend_direction_min", 0.3):
            logger.debug(f"[vwap_bounce] {ticker}: trend {trend:.2f} too weak")
            return None

        # VWAP touch count: not too few (no test), not too many (broken)
        min_t = params.get("min_vwap_touches", 1)
        max_t = params.get("max_vwap_touches", 4)
        if touches < min_t or touches > max_t:
            logger.debug(f"[vwap_bounce] {ticker}: VWAP touches {touches} outside [{min_t},{max_t}]")
            return None

        if volume_ratio < params.get("min_volume_ratio", 1.2):
            return None

        # Price must be near VWAP (within proximity %)
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        proximity = params.get("bounce_proximity_pct", 0.3)
        dist_pct = abs(current_price - vwap) / vwap * 100 if vwap > 0 else 999
        if dist_pct > proximity:
            logger.debug(
                f"[vwap_bounce] {ticker}: distance from VWAP {dist_pct:.2f}% > {proximity}%"
            )
            return None

        # Bullish candle: close > open and close near high
        candle_open = float(latest["open"])
        candle_high = float(latest["high"])
        candle_low = float(latest["low"])
        candle_range = candle_high - candle_low

        is_bullish = current_price > candle_open
        if not is_bullish:
            logger.debug(f"[vwap_bounce] {ticker}: latest candle not bullish")
            return None

        # Low should have wicked near or below VWAP
        if candle_low > vwap * 1.002:
            logger.debug(f"[vwap_bounce] {ticker}: candle low {candle_low:.0f} too far above VWAP")
            return None

        # Calculate levels
        stop_offset = atr * params.get("stop_atr_below_vwap", 0.5)
        entry_price = current_price
        stop_price = vwap - stop_offset
        day_high = float(data["high"].max())
        atr_target = entry_price + atr * params.get("target_atr_multiple", 1.0)
        target_price = max(day_high, atr_target)

        if stop_price >= entry_price:
            return None

        # Confidence
        confidence = 0.50
        if trend > 0.6:
            confidence += 0.10
        if volume_ratio > 2.0:
            confidence += 0.10
        if candle_range > 0 and (current_price - candle_open) / candle_range > 0.6:
            confidence += 0.10
        if touches >= 2:
            confidence += 0.05  # VWAP tested and held
        confidence = min(confidence, 0.90)

        shares = self.calculate_position_size(entry_price, atr, 10_000_000)

        signal = TradeSignal(
            ticker=ticker,
            direction="long",
            strategy_name=self.name,
            entry_price=round(entry_price, 1),
            stop_loss=round(stop_price, 1),
            take_profit=round(target_price, 1),
            confidence=round(confidence, 2),
            entry_reason=(
                f"VWAPバウンス: VWAP({vwap:.0f})で反発, "
                f"本日{touches}回テスト, "
                f"トレンド{trend:+.2f}, "
                f"出来高比{volume_ratio:.1f}x"
            ),
            features_snapshot=features,
        )
        logger.info(f"[vwap_bounce] SIGNAL {ticker} long entry={entry_price:.0f}")
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        vwap = features.get("vwap", 0)
        atr = features.get("atr", 0)

        # Stop
        if current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        # VWAP break: close below VWAP with margin
        if vwap > 0 and atr > 0:
            if current_price < vwap - atr * 0.3:
                return True, f"VWAP割れ ({current_price:.0f} < VWAP {vwap:.0f})"

        # Trailing: move stop to breakeven after 0.5 ATR profit
        if atr > 0:
            profit = current_price - position.entry_price
            if profit > atr * 0.5:
                breakeven = position.entry_price + 1.0
                if current_price <= breakeven:
                    return True, f"ブレイクイーブンストップ ({current_price:.0f})"

        # Time
        if position.holding_minutes > 45:
            if current_price < position.entry_price * 1.002:
                return True, "時間切れ45分"

        return False, ""
