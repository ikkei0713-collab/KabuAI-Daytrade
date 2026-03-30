"""Overextension Reversal strategy.

Fires when price moves more than 3 ATR away from VWAP and RSI is
extreme (> 80 or < 20).  Waits for a reversal candle pattern
(hammer or engulfing) before entry.  Stop is placed beyond the
extreme; target is VWAP or the mean.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class OverextensionStrategy(BaseStrategy):
    """過伸び反転 – mean reversion on extreme overextension."""

    REQUIRED_FEATURES = [
        "atr_distance_from_vwap",
        "rsi_14",
        "candle_pattern",
        "volume_climax",
        "vwap",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="overextension",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="range",
            parameter_set={
                "min_atr_distance": 3.0,
                "rsi_overbought": 80,
                "rsi_oversold": 20,
                "required_patterns": ["hammer", "engulfing", "doji_star", "pin_bar"],
                "min_volume_climax": 2.0,
                "stop_buffer_atr": 0.5,
                "blocked_regimes": ["trend_down"],
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

        if not self._check_regime_filter(features):
            return None

        atr_dist: float = features["atr_distance_from_vwap"]
        rsi_14: float = features["rsi_14"]
        candle_pat: str = features["candle_pattern"]
        vol_climax: float = features["volume_climax"]
        vwap: float = features["vwap"]
        atr: float = features["atr"]

        min_dist = params.get("min_atr_distance", 3.0)
        rsi_ob = params.get("rsi_overbought", 80)
        rsi_os = params.get("rsi_oversold", 20)
        required_pats = params.get("required_patterns", ["hammer", "engulfing"])

        latest = data.iloc[-1]
        current_price = float(latest["close"])

        # Determine direction of overextension
        is_overextended_up = atr_dist > min_dist and rsi_14 > rsi_ob
        is_overextended_down = atr_dist < -min_dist and rsi_14 < rsi_os

        if not is_overextended_up and not is_overextended_down:
            logger.debug(
                f"[overextension] {ticker}: atr_dist={atr_dist:.1f}, "
                f"rsi={rsi_14:.0f} – not extreme"
            )
            return None

        # Need a reversal candle pattern
        if candle_pat not in required_pats:
            logger.debug(
                f"[overextension] {ticker}: candle pattern '{candle_pat}' "
                f"not in {required_pats}"
            )
            return None

        # Volume climax: selling/buying climax often marks extremes
        min_vol_climax = params.get("min_volume_climax", 2.0)
        if vol_climax < min_vol_climax:
            logger.debug(f"[overextension] {ticker}: vol_climax {vol_climax:.1f} < {min_vol_climax}")
            return None

        stop_buf = atr * params.get("stop_buffer_atr", 0.5)

        if is_overextended_up:
            # Short / fade-sell reversal
            direction = "short"
            extreme_high = float(data["high"].max())
            entry_price = current_price
            stop_price = extreme_high + stop_buf
            target_price = vwap
            if target_price >= entry_price:
                target_price = entry_price - abs(atr_dist) * atr * 0.5
        else:
            # Long reversal
            direction = "long"
            extreme_low = float(data["low"].min())
            entry_price = current_price
            stop_price = extreme_low - stop_buf
            target_price = vwap
            if target_price <= entry_price:
                target_price = entry_price + abs(atr_dist) * atr * 0.5

        risk = abs(entry_price - stop_price)
        if risk < 1.0:
            return None

        # Confidence
        confidence = 0.45
        if abs(atr_dist) > 4.0:
            confidence += 0.10
        if vol_climax > 3.0:
            confidence += 0.10
        if candle_pat == "engulfing":
            confidence += 0.10
        elif candle_pat == "hammer":
            confidence += 0.05
        if (is_overextended_up and rsi_14 > 85) or (is_overextended_down and rsi_14 < 15):
            confidence += 0.10
        confidence = min(confidence, 0.85)

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
                f"過伸び反転{'↓売' if direction == 'short' else '↑買'}: "
                f"VWAP距離{atr_dist:.1f}ATR, RSI={rsi_14:.0f}, "
                f"パターン={candle_pat}, 出来高クライマックス{vol_climax:.1f}x"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[overextension] SIGNAL {ticker} {direction} "
            f"entry={entry_price:.0f} atr_dist={atr_dist:.1f}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        vwap = features.get("vwap", 0)
        rsi = features.get("rsi_14", 50)
        atr = features.get("atr", 0)

        # Stop
        if position.direction == "long" and current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"
        if position.direction == "short" and current_price >= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if position.direction == "long" and current_price >= position.take_profit:
            return True, f"利確 – VWAP/平均到達 ({current_price:.0f})"
        if position.direction == "short" and current_price <= position.take_profit:
            return True, f"利確 – VWAP/平均到達 ({current_price:.0f})"

        # RSI normalised: reversal is working
        if 40 < rsi < 60:
            if position.direction == "long" and current_price > position.entry_price:
                return True, f"RSI正常化 ({rsi:.0f}) – 利益確定"
            if position.direction == "short" and current_price < position.entry_price:
                return True, f"RSI正常化 ({rsi:.0f}) – 利益確定"

        # VWAP reached
        if vwap > 0:
            if position.direction == "long" and current_price >= vwap * 0.998:
                return True, f"VWAP到達 ({current_price:.0f})"
            if position.direction == "short" and current_price <= vwap * 1.002:
                return True, f"VWAP到達 ({current_price:.0f})"

        # Time: mean reversion should happen within 30 min
        if position.holding_minutes > 30:
            return True, "時間切れ30分 – 反転未完了"

        return False, ""
