"""Orderbook Imbalance strategy.

Enters in the direction of a significant bid/ask imbalance
(ratio > 2.0 for longs, < 0.5 for shorts).  For paper trading,
the orderbook data is simulated from volume and price patterns.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class ImbalanceStrategy(BaseStrategy):
    """板不均衡 – trade in the direction of orderbook imbalance."""

    REQUIRED_FEATURES = [
        "bid_ask_ratio",
        "spread",
        "depth_imbalance",
        "atr",
        "vwap",
        "volume_ratio",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="orderbook_imbalance",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                # 大規模BT (2026-04-06): OOS PF=1.50 WR=55% 80件 +¥7,149
                "long_ratio_threshold": 1.5,    # 緩和: BTベスト
                "short_ratio_threshold": 0.5,
                "max_spread_pct": 0.3,
                "min_depth_imbalance": 0.5,     # BTベスト
                "target_atr_multiple": 1.0,     # 手数料無料: 回転UP
                "stop_atr_multiple": 0.8,
                "min_volume_ratio": 1.0,
                "blocked_regimes": ["trend_down", "low_vol"],  # volatile解除
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

        if not self._check_regime_filter(features):
            return None

        ba_ratio: float = features["bid_ask_ratio"]
        spread: float = features["spread"]
        depth_imb: float = features["depth_imbalance"]
        atr: float = features["atr"]
        vwap: float = features["vwap"]
        volume_ratio: float = features["volume_ratio"]

        latest = data.iloc[-1]
        current_price = float(latest["close"])

        # Spread must not be too wide
        spread_pct = spread / current_price * 100 if current_price > 0 else 999
        max_spread = params.get("max_spread_pct", 0.3)
        if spread_pct > max_spread:
            logger.debug(f"[imbalance] {ticker}: spread {spread_pct:.2f}% > {max_spread}%")
            return None

        if volume_ratio < params.get("min_volume_ratio", 1.0):
            return None

        long_thresh = params.get("long_ratio_threshold", 2.0)
        short_thresh = params.get("short_ratio_threshold", 0.5)
        min_depth = params.get("min_depth_imbalance", 0.3)

        direction = None
        if ba_ratio >= long_thresh and depth_imb >= min_depth:
            direction = "long"
        elif ba_ratio <= short_thresh and depth_imb <= -min_depth:
            direction = "short"

        if direction is None:
            logger.debug(
                f"[imbalance] {ticker}: BA ratio {ba_ratio:.2f}, "
                f"depth_imb {depth_imb:.2f} – no signal"
            )
            return None

        # Confirm with price action: price moving in direction of imbalance
        prev_bar = data.iloc[-2] if len(data) >= 2 else latest
        price_change = current_price - float(prev_bar["close"])
        if direction == "long" and price_change < 0:
            logger.debug(f"[imbalance] {ticker}: bid imbalance but price dropping")
            return None
        if direction == "short" and price_change > 0:
            logger.debug(f"[imbalance] {ticker}: ask imbalance but price rising")
            return None

        stop_mult = params.get("stop_atr_multiple", 0.8)
        target_mult = params.get("target_atr_multiple", 1.0)

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
        if ba_ratio > 3.0 or ba_ratio < 0.33:
            confidence += 0.10
        if abs(depth_imb) > 0.6:
            confidence += 0.10
        if spread_pct < 0.1:
            confidence += 0.05
        if volume_ratio > 1.5:
            confidence += 0.05
        # VWAP alignment
        if direction == "long" and current_price > vwap:
            confidence += 0.05
        elif direction == "short" and current_price < vwap:
            confidence += 0.05
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
                f"板不均衡{'買い優勢' if direction == 'long' else '売り優勢'}: "
                f"Bid/Ask比={ba_ratio:.2f}, "
                f"板深度偏り={depth_imb:+.2f}, "
                f"スプレッド{spread_pct:.2f}%"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[imbalance] SIGNAL {ticker} {direction} "
            f"BA={ba_ratio:.2f} depth={depth_imb:.2f}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        ba_ratio = features.get("bid_ask_ratio", 1.0)
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

        # Imbalance reversed
        if position.direction == "long" and ba_ratio < 0.8:
            return True, f"板不均衡反転 – 売り優勢化 (BA={ba_ratio:.2f})"
        if position.direction == "short" and ba_ratio > 1.2:
            return True, f"板不均衡反転 – 買い優勢化 (BA={ba_ratio:.2f})"

        # Orderbook signals are fast: 15 min max
        if position.holding_minutes > 15:
            return True, "時間切れ15分 – 板シグナルポジション決済"

        return False, ""
