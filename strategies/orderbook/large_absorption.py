"""大口板吸収 (Large Absorption) strategy.

Detects when a large resting order is absorbed (eaten through) without
the price moving significantly, indicating hidden buying or selling
pressure.  For paper trading this is detected from volume/price
divergence patterns.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class LargeAbsorptionStrategy(BaseStrategy):
    """大口板吸収 – trade in direction of hidden large-order absorption."""

    REQUIRED_FEATURES = [
        "volume_price_divergence",
        "large_trade_detection",
        "atr",
        "vwap",
        "volume_ratio",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="large_absorption",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_volume_price_divergence": 2.0,
                "large_trade_threshold": 0.7,
                "target_atr_multiple": 1.2,
                "stop_atr_multiple": 0.8,
                "min_volume_ratio": 1.5,
                "confirmation_bars": 2,
                "blocked_regimes": ["trend_down"],
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=4):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        params = self.config.parameter_set

        if not self._check_regime_filter(features):
            return None

        vp_div: float = features["volume_price_divergence"]
        large_td: float = features["large_trade_detection"]
        atr: float = features["atr"]
        vwap: float = features["vwap"]
        volume_ratio: float = features["volume_ratio"]

        min_div = params.get("min_volume_price_divergence", 2.0)
        large_thresh = params.get("large_trade_threshold", 0.7)
        min_vol = params.get("min_volume_ratio", 1.5)
        conf_bars = params.get("confirmation_bars", 2)

        # Volume/price divergence: high volume but little price movement
        # Positive divergence = buying absorption, negative = selling absorption
        if abs(vp_div) < min_div:
            logger.debug(
                f"[large_absorption] {ticker}: VP divergence {vp_div:.2f} "
                f"below threshold {min_div}"
            )
            return None

        # Large trade detection confidence
        if large_td < large_thresh:
            logger.debug(
                f"[large_absorption] {ticker}: large_trade_detection "
                f"{large_td:.2f} < {large_thresh}"
            )
            return None

        if volume_ratio < min_vol:
            return None

        # Determine absorption direction from divergence sign and price action
        latest = data.iloc[-1]
        current_price = float(latest["close"])

        # Check confirmation bars: volume high but price stable
        if len(data) >= conf_bars + 1:
            recent = data.iloc[-conf_bars:]
            price_range = float(recent["high"].max()) - float(recent["low"].min())
            avg_vol = float(data.iloc[:-conf_bars]["volume"].mean()) if len(data) > conf_bars else 1
            recent_vol = float(recent["volume"].mean())

            # Price range should be small relative to ATR
            if price_range > atr * 0.5:
                logger.debug(
                    f"[large_absorption] {ticker}: price range {price_range:.0f} "
                    f"too wide for absorption"
                )
                return None
        else:
            recent_vol = float(latest["volume"])
            avg_vol = recent_vol

        # Direction: positive VP divergence = hidden buying
        if vp_div > 0:
            direction = "long"
        else:
            direction = "short"

        stop_mult = params.get("stop_atr_multiple", 0.8)
        target_mult = params.get("target_atr_multiple", 1.2)

        if direction == "long":
            entry_price = current_price
            stop_price = entry_price - atr * stop_mult
            target_price = entry_price + atr * target_mult
        else:
            entry_price = current_price
            stop_price = entry_price + atr * stop_mult
            target_price = entry_price - atr * target_mult

        risk = abs(entry_price - stop_price)
        if risk < 1.0:
            return None

        # Confidence
        confidence = 0.45
        if abs(vp_div) > 3.0:
            confidence += 0.10
        if large_td > 0.85:
            confidence += 0.10
        if volume_ratio > 2.5:
            confidence += 0.05
        # VWAP alignment
        if direction == "long" and current_price >= vwap:
            confidence += 0.05
        elif direction == "short" and current_price <= vwap:
            confidence += 0.05
        # Volume confirmation
        if avg_vol > 0 and recent_vol / avg_vol > 2.5:
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
                f"大口板吸収{'買い' if direction == 'long' else '売り'}: "
                f"VP乖離={vp_div:.2f}, "
                f"大口検出={large_td:.2f}, "
                f"出来高比{volume_ratio:.1f}x"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[large_absorption] SIGNAL {ticker} {direction} "
            f"VP_div={vp_div:.2f} large_td={large_td:.2f}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        vp_div = features.get("volume_price_divergence", 0)
        atr = features.get("atr", 0)

        # Stop
        if position.direction == "long" and current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"
        if position.direction == "short" and current_price >= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if position.direction == "long" and current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"
        if position.direction == "short" and current_price <= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        # Absorption signal reversed
        if position.direction == "long" and vp_div < -1.0:
            return True, f"吸収反転 – 売り圧力出現 (VP={vp_div:.2f})"
        if position.direction == "short" and vp_div > 1.0:
            return True, f"吸収反転 – 買い圧力出現 (VP={vp_div:.2f})"

        # Price breakout from absorption zone
        if atr > 0:
            profit = current_price - position.entry_price
            if position.direction == "short":
                profit = -profit
            if profit > atr * 0.8:
                # Trail stop at 0.5 ATR
                trail = atr * 0.5
                if position.direction == "long":
                    if current_price < (position.entry_price + profit - trail):
                        return True, f"トレーリングストップ ({current_price:.0f})"
                else:
                    if current_price > (position.entry_price - profit + trail):
                        return True, f"トレーリングストップ ({current_price:.0f})"

        # Time
        if position.holding_minutes > 20:
            return True, "時間切れ20分 – 板シグナルポジション決済"

        return False, ""
