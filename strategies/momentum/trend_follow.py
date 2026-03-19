"""Intraday Trend Follow strategy.

Identifies an intraday trend using EMA(9) > EMA(21) > VWAP (for longs).
Enters on a pullback to EMA(9) in the trend direction.  Stop is placed
below EMA(21); target uses a trailing stop at 1.5 ATR.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class TrendFollowStrategy(BaseStrategy):
    """日中トレンドフォロー – ride the intraday trend."""

    REQUIRED_FEATURES = [
        "ema_9",
        "ema_21",
        "vwap",
        "trend_strength",
        "volume_trend",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    @staticmethod
    def is_trending(features: dict) -> tuple[bool, str, float]:
        """Check if market is trending. Used as a filter by other strategies.

        Returns:
            (is_trending, direction, strength) where direction is 'up'/'down'/'none'
            and strength is 0-1.
        """
        ema_9 = features.get("ema_9", 0)
        ema_21 = features.get("ema_21", 0)
        vwap = features.get("vwap", 0)
        trend_str = features.get("trend_strength", 0)

        if not all([ema_9, ema_21, vwap]):
            return False, "none", 0.0

        buf = ema_9 * 0.0005  # 0.05% buffer

        if ema_9 > ema_21 + buf and ema_21 > vwap + buf:
            return True, "up", min(abs(trend_str), 1.0)
        elif ema_9 < ema_21 - buf and ema_21 < vwap - buf:
            return True, "down", min(abs(trend_str), 1.0)

        return False, "none", 0.0

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="trend_follow",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_trend_strength": 0.4,
                "min_volume_trend": 1.0,
                "trailing_atr_multiple": 1.5,
                "pullback_ema": "ema_9",
                "stop_ema": "ema_21",
                "ema_alignment_buffer_pct": 0.05,
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=5):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        params = self.config.parameter_set
        ema_9: float = features["ema_9"]
        ema_21: float = features["ema_21"]
        vwap: float = features["vwap"]
        trend_str: float = features["trend_strength"]
        vol_trend: float = features["volume_trend"]
        atr: float = features["atr"]

        latest = data.iloc[-1]
        current_price = float(latest["close"])
        current_low = float(latest["low"])
        buf = ema_9 * params.get("ema_alignment_buffer_pct", 0.05) / 100

        # --- Determine direction from EMA alignment ---
        long_aligned = ema_9 > ema_21 + buf and ema_21 > vwap + buf
        short_aligned = ema_9 < ema_21 - buf and ema_21 < vwap - buf

        if not long_aligned and not short_aligned:
            logger.debug(
                f"[trend_follow] {ticker}: EMAs not aligned "
                f"(EMA9={ema_9:.0f}, EMA21={ema_21:.0f}, VWAP={vwap:.0f})"
            )
            return None

        # Trend strength filter
        if abs(trend_str) < params.get("min_trend_strength", 0.4):
            logger.debug(f"[trend_follow] {ticker}: trend_strength {trend_str:.2f} too weak")
            return None

        # Volume trend: rising volume in trend direction
        if vol_trend < params.get("min_volume_trend", 1.0):
            logger.debug(f"[trend_follow] {ticker}: volume_trend {vol_trend:.2f} declining")
            return None

        # --- Pullback detection ---
        if long_aligned:
            # Price should be near EMA9 (pulled back to it)
            pullback_dist = (current_price - ema_9) / atr if atr > 0 else 999
            if pullback_dist > 0.5 or pullback_dist < -0.3:
                logger.debug(
                    f"[trend_follow] {ticker}: long pullback distance "
                    f"{pullback_dist:.2f} ATR not ideal"
                )
                return None

            # Candle must show bounce (close above open)
            if current_price < float(latest["open"]):
                return None

            entry_price = current_price
            stop_price = ema_21 - atr * 0.2
            target_price = entry_price + atr * params.get("trailing_atr_multiple", 1.5) * 2
            direction = "long"

        else:  # short_aligned
            pullback_dist = (ema_9 - current_price) / atr if atr > 0 else 999
            if pullback_dist > 0.5 or pullback_dist < -0.3:
                return None

            if current_price > float(latest["open"]):
                return None

            entry_price = current_price
            stop_price = ema_21 + atr * 0.2
            target_price = entry_price - atr * params.get("trailing_atr_multiple", 1.5) * 2
            direction = "short"

        risk = abs(entry_price - stop_price)
        if risk < 1.0:
            return None

        # Confidence
        confidence = 0.50
        if abs(trend_str) > 0.7:
            confidence += 0.10
        if vol_trend > 1.5:
            confidence += 0.10
        ema_spread = abs(ema_9 - ema_21)
        if atr > 0 and ema_spread / atr > 0.3:
            confidence += 0.05
        # Strong body on pullback candle
        body = abs(current_price - float(latest["open"]))
        candle_range = float(latest["high"]) - current_low
        if candle_range > 0 and body / candle_range > 0.6:
            confidence += 0.10
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
                f"トレンドフォロー{'↑' if direction == 'long' else '↓'}: "
                f"EMA9({ema_9:.0f})>EMA21({ema_21:.0f})>VWAP({vwap:.0f}), "
                f"トレンド強度{trend_str:+.2f}, "
                f"出来高トレンド{vol_trend:.1f}x"
            ),
            features_snapshot=features,
        )
        logger.info(f"[trend_follow] SIGNAL {ticker} {direction} entry={entry_price:.0f}")
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        ema_21 = features.get("ema_21", 0)
        atr = features.get("atr", 0)
        trail_mult = self.config.parameter_set.get("trailing_atr_multiple", 1.5)

        # Hard stop
        if position.direction == "long" and current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"
        if position.direction == "short" and current_price >= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # EMA21 break (trend structure broken)
        if position.direction == "long" and ema_21 > 0 and current_price < ema_21:
            return True, f"EMA21割れ – トレンド崩壊 ({current_price:.0f})"
        if position.direction == "short" and ema_21 > 0 and current_price > ema_21:
            return True, f"EMA21突破 – トレンド崩壊 ({current_price:.0f})"

        # Trailing stop based on ATR
        if atr > 0:
            if position.direction == "long":
                day_high = float(current_data["high"].max())
                trail = day_high - atr * trail_mult
                if trail > position.entry_price and current_price <= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"
            else:
                day_low = float(current_data["low"].min())
                trail = day_low + atr * trail_mult
                if trail < position.entry_price and current_price >= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"

        # Target
        if position.direction == "long" and current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"
        if position.direction == "short" and current_price <= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        return False, ""
