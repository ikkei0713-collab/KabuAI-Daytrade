"""スプレッド縮小エントリー strategy.

Enters when the bid-ask spread narrows significantly, indicating
an imminent move.  The entry is taken on the breakout after
spread contraction.  Works well on liquid large-cap stocks.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class SpreadEntryStrategy(BaseStrategy):
    """スプレッド縮小エントリー – breakout after spread tightening."""

    REQUIRED_FEATURES = [
        "spread_percentile",
        "volume_building",
        "price_compression",
        "atr",
        "vwap",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    @staticmethod
    def get_spread_boost(features: dict) -> float:
        """Return confidence boost (0 to 0.15) based on spread conditions.

        Used as auxiliary signal to boost other strategies' confidence
        when spread conditions are favorable.
        """
        spread_pct = features.get("spread_percentile", 50)
        vol_build = features.get("volume_building", 1.0)
        price_comp = features.get("price_compression", 0.0)

        boost = 0.0
        if spread_pct < 20:
            boost += 0.05
        if vol_build > 2.0:
            boost += 0.05
        if price_comp > 0.5:
            boost += 0.05

        return min(boost, 0.15)

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="spread_entry",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="range",
            parameter_set={
                "spread_percentile_max": 20,
                "min_volume_building": 1.8,  # 最適化結果: 2.0→1.8 (OOS PF 1.68)
                "min_price_compression": 0.5,
                "target_atr_multiple": 1.5,  # 最適化結果: 1.0→1.5
                "stop_atr_multiple": 0.5,    # 最適化結果: 0.7→0.5
                "breakout_threshold_pct": 0.3,  # 最適化結果: 0.15→0.3
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
        spread_pct: float = features["spread_percentile"]
        vol_build: float = features["volume_building"]
        price_comp: float = features["price_compression"]
        atr: float = features["atr"]
        vwap: float = features["vwap"]

        # Spread must be in lower percentile (tight)
        max_pctile = params.get("spread_percentile_max", 20)
        if spread_pct > max_pctile:
            logger.debug(f"[spread_entry] {ticker}: spread percentile {spread_pct:.0f} > {max_pctile}")
            return None

        # Volume should be building (accumulation)
        min_vol = params.get("min_volume_building", 1.3)
        if vol_build < min_vol:
            logger.debug(f"[spread_entry] {ticker}: volume_building {vol_build:.2f} < {min_vol}")
            return None

        # Price compression: recent range narrow relative to ATR
        min_comp = params.get("min_price_compression", 0.5)
        if price_comp < min_comp:
            logger.debug(f"[spread_entry] {ticker}: price_compression {price_comp:.2f} < {min_comp}")
            return None

        # Detect breakout direction from latest candle
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        current_open = float(latest["open"])
        current_high = float(latest["high"])
        current_low = float(latest["low"])

        # Recent compression range
        lookback = min(len(data), 6)
        recent = data.iloc[-lookback:]
        comp_high = float(recent["high"].max())
        comp_low = float(recent["low"].min())
        comp_range = comp_high - comp_low
        breakout_thresh = current_price * params.get("breakout_threshold_pct", 0.15) / 100

        direction = None
        if current_price > comp_high - breakout_thresh and current_price > current_open:
            direction = "long"
        elif current_price < comp_low + breakout_thresh and current_price < current_open:
            direction = "short"

        if direction is None:
            # Not yet breaking out
            logger.debug(
                f"[spread_entry] {ticker}: no breakout from compression "
                f"[{comp_low:.0f}-{comp_high:.0f}]"
            )
            return None

        stop_mult = params.get("stop_atr_multiple", 0.7)
        target_mult = params.get("target_atr_multiple", 1.0)

        if direction == "long":
            entry_price = current_price
            stop_price = comp_low - atr * 0.2  # Below compression zone
            target_price = entry_price + max(comp_range, atr * target_mult)
        else:
            entry_price = current_price
            stop_price = comp_high + atr * 0.2
            target_price = entry_price - max(comp_range, atr * target_mult)

        risk = abs(entry_price - stop_price)
        if risk < 1.0:
            return None

        # Confidence
        confidence = 0.45
        if spread_pct < 10:
            confidence += 0.10
        if vol_build > 2.0:
            confidence += 0.10
        if price_comp > 0.7:
            confidence += 0.05
        # Strong breakout candle
        body = abs(current_price - current_open)
        candle_range = current_high - current_low
        if candle_range > 0 and body / candle_range > 0.65:
            confidence += 0.10
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
                f"スプレッド縮小ブレイク{'↑' if direction == 'long' else '↓'}: "
                f"スプレッド{spread_pct:.0f}パーセンタイル, "
                f"出来高蓄積{vol_build:.1f}x, "
                f"価格圧縮{price_comp:.2f}"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[spread_entry] SIGNAL {ticker} {direction} "
            f"spread_pct={spread_pct:.0f} vol_build={vol_build:.1f}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        spread_pct = features.get("spread_percentile", 50)
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

        # Spread widening: breakout may be failing
        if spread_pct > 70:
            return True, f"スプレッド拡大 ({spread_pct:.0f}パーセンタイル) – 流動性低下"

        # False breakout: price returned into compression zone
        if atr > 0:
            profit = current_price - position.entry_price
            if position.direction == "short":
                profit = -profit
            if profit < -atr * 0.3:
                return True, f"偽ブレイクアウト ({current_price:.0f})"

        # Time
        if position.holding_minutes > 20:
            profit = current_price - position.entry_price
            if position.direction == "short":
                profit = -profit
            if profit < atr * 0.2:
                return True, "時間切れ20分"

        return False, ""
