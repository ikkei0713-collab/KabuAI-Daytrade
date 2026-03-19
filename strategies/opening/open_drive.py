"""Open Drive strategy (寄り付きドライブ).

Detects a strong directional move in the first 5 minutes of trading.
Entry is taken on the first pullback after the initial drive.
Stop is placed below the opening range; target is 1.5-2x the
opening range size.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class OpenDriveStrategy(BaseStrategy):
    """寄り付きドライブ – momentum continuation off the open."""

    REQUIRED_FEATURES = [
        "opening_range_size",
        "volume_first_5min",
        "tick_direction",
        "atr",
        "vwap",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="open_drive",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_drive_pct": 0.8,
                "min_volume_ratio_5min": 2.0,
                "target_range_multiple": 1.75,
                "max_pullback_pct": 0.5,
                "min_tick_direction": 0.6,
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
        or_size: float = features["opening_range_size"]
        vol_5min: float = features["volume_first_5min"]
        tick_dir: float = features["tick_direction"]
        atr: float = features["atr"]
        vwap: float = features["vwap"]

        first_bar = data.iloc[0]
        open_price = float(first_bar["open"])
        first_high = float(first_bar["high"])
        first_low = float(first_bar["low"])

        # Drive must be significant
        drive_pct = or_size / open_price * 100 if open_price > 0 else 0
        if drive_pct < params.get("min_drive_pct", 0.8):
            logger.debug(f"[open_drive] {ticker}: drive {drive_pct:.2f}% too small")
            return None

        # Volume confirmation
        if vol_5min < params.get("min_volume_ratio_5min", 2.0):
            return None

        # Tick direction: > 0.6 means strong buying (ratio of upticks)
        min_tick = params.get("min_tick_direction", 0.6)
        is_long = tick_dir >= min_tick
        is_short = tick_dir <= (1 - min_tick)
        if not is_long and not is_short:
            logger.debug(f"[open_drive] {ticker}: tick_dir {tick_dir:.2f} indecisive")
            return None

        # Wait for pullback: current price should have pulled back
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        range_mult = params.get("target_range_multiple", 1.75)
        max_pb = params.get("max_pullback_pct", 0.5)

        if is_long:
            pullback_depth = (first_high - current_price) / or_size if or_size > 0 else 0
            if pullback_depth < 0.15 or pullback_depth > max_pb:
                logger.debug(
                    f"[open_drive] {ticker}: pullback depth {pullback_depth:.2f} "
                    f"not in [0.15, {max_pb}]"
                )
                return None

            entry_price = current_price + 1.0
            stop_price = first_low - 1.0
            target_price = entry_price + or_size * range_mult
            direction = "long"
        else:
            pullback_depth = (current_price - first_low) / or_size if or_size > 0 else 0
            if pullback_depth < 0.15 or pullback_depth > max_pb:
                return None

            entry_price = current_price - 1.0
            stop_price = first_high + 1.0
            target_price = entry_price - or_size * range_mult
            direction = "short"

        if direction == "long" and stop_price >= entry_price:
            return None
        if direction == "short" and stop_price <= entry_price:
            return None

        # Confidence
        confidence = 0.50
        if drive_pct > 1.5:
            confidence += 0.10
        if vol_5min > 3.0:
            confidence += 0.10
        tick_strength = tick_dir if is_long else (1 - tick_dir)
        if tick_strength > 0.75:
            confidence += 0.10
        if direction == "long" and current_price > vwap:
            confidence += 0.05
        confidence = min(confidence, 0.95)

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
                f"寄り付きドライブ{'↑' if is_long else '↓'}: "
                f"レンジ{or_size:.0f}円({drive_pct:.1f}%), "
                f"出来高比{vol_5min:.1f}x, "
                f"Tick方向{tick_dir:.2f}"
            ),
            features_snapshot=features,
        )
        logger.info(f"[open_drive] SIGNAL {ticker} {direction} entry={entry_price:.0f}")
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        atr = features.get("atr", 0)

        # Stop loss
        if position.direction == "long" and current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"
        if position.direction == "short" and current_price >= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if position.direction == "long" and current_price >= position.take_profit:
            return True, f"利確到達 ({current_price:.0f})"
        if position.direction == "short" and current_price <= position.take_profit:
            return True, f"利確到達 ({current_price:.0f})"

        # Trailing stop: 1 ATR
        if atr > 0:
            if position.direction == "long":
                profit = current_price - position.entry_price
                if profit > atr:
                    trail = current_price - atr
                    if current_price <= trail:
                        return True, f"トレーリングストップ ({current_price:.0f})"
            else:
                profit = position.entry_price - current_price
                if profit > atr:
                    trail = current_price + atr
                    if current_price >= trail:
                        return True, f"トレーリングストップ ({current_price:.0f})"

        # Time: opening drive should resolve within 45 min
        if position.holding_minutes > 45:
            return True, "時間切れ45分 – ポジション決済"

        return False, ""
