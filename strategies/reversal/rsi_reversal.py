"""RSI短期逆張り strategy.

Fires when RSI(5) reaches extreme levels (< 15 or > 85) on the
5-minute chart, confirmed by a volume spike.  Entry is taken when
RSI crosses back toward 50.  Stop is 1 ATR from entry; target is
the price level corresponding to RSI returning to 50.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class RSIReversalStrategy(BaseStrategy):
    """RSI短期逆張り – mean reversion on extreme short-term RSI."""

    REQUIRED_FEATURES = [
        "rsi_5",
        "volume_spike",
        "price_vs_bollinger",
        "atr",
        "vwap",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="rsi_reversal",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="range",
            parameter_set={
                "rsi_oversold": 15,
                "rsi_overbought": 85,
                "rsi_crossback_threshold": 5,
                "min_volume_spike": 2.0,
                "stop_atr_multiple": 1.0,
                "bollinger_confirmation": True,
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

        rsi_5: float = features["rsi_5"]
        vol_spike: float = features["volume_spike"]
        bb_pos: float = features["price_vs_bollinger"]  # -1 to 1 scale
        atr: float = features["atr"]
        vwap: float = features["vwap"]

        rsi_os = params.get("rsi_oversold", 15)
        rsi_ob = params.get("rsi_overbought", 85)
        crossback = params.get("rsi_crossback_threshold", 5)
        min_vol_spike = params.get("min_volume_spike", 2.0)

        latest = data.iloc[-1]
        current_price = float(latest["close"])

        # Check for RSI extremes
        is_oversold = rsi_5 < rsi_os + crossback  # RSI was extreme and is crossing back
        is_overbought = rsi_5 > rsi_ob - crossback

        if not is_oversold and not is_overbought:
            return None

        # For long: RSI should be crossing *back up* from oversold
        # For short: RSI should be crossing *back down* from overbought
        if is_oversold and rsi_5 < rsi_os:
            # Still deeply oversold - wait for crossback
            logger.debug(f"[rsi_reversal] {ticker}: RSI {rsi_5:.1f} still deeply oversold, waiting")
            return None
        if is_overbought and rsi_5 > rsi_ob:
            logger.debug(f"[rsi_reversal] {ticker}: RSI {rsi_5:.1f} still deeply overbought, waiting")
            return None

        # Volume spike confirmation
        if vol_spike < min_vol_spike:
            logger.debug(f"[rsi_reversal] {ticker}: volume_spike {vol_spike:.1f} < {min_vol_spike}")
            return None

        # Bollinger band confirmation
        use_bb = params.get("bollinger_confirmation", True)
        if use_bb:
            if is_oversold and bb_pos > -0.5:
                logger.debug(f"[rsi_reversal] {ticker}: BB position {bb_pos:.2f} not confirming oversold")
                return None
            if is_overbought and bb_pos < 0.5:
                logger.debug(f"[rsi_reversal] {ticker}: BB position {bb_pos:.2f} not confirming overbought")
                return None

        # Entry / levels
        stop_mult = params.get("stop_atr_multiple", 1.0)

        if is_oversold:
            direction = "long"
            entry_price = current_price
            stop_price = entry_price - atr * stop_mult
            # Target: approximate price when RSI would reach ~50
            recent_range = float(data.iloc[-6:]["high"].max()) - float(data.iloc[-6:]["low"].min())
            target_price = entry_price + recent_range * 0.4
            target_price = max(target_price, vwap) if vwap > entry_price else target_price
        else:
            direction = "short"
            entry_price = current_price
            stop_price = entry_price + atr * stop_mult
            recent_range = float(data.iloc[-6:]["high"].max()) - float(data.iloc[-6:]["low"].min())
            target_price = entry_price - recent_range * 0.4
            target_price = min(target_price, vwap) if vwap < entry_price else target_price

        risk = abs(entry_price - stop_price)
        if risk < 1.0:
            return None

        # Confidence
        confidence = 0.45
        if vol_spike > 3.0:
            confidence += 0.10
        if abs(bb_pos) > 0.8:
            confidence += 0.10
        # RSI just crossed back (closer to threshold = stronger)
        if is_oversold:
            rsi_distance_from_extreme = rsi_5 - rsi_os
            if rsi_distance_from_extreme < 3:
                confidence += 0.10
        else:
            rsi_distance_from_extreme = rsi_ob - rsi_5
            if rsi_distance_from_extreme < 3:
                confidence += 0.10
        # Bullish candle on long, bearish on short
        is_favorable = (direction == "long" and current_price > float(latest["open"])) or \
                       (direction == "short" and current_price < float(latest["open"]))
        if is_favorable:
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
                f"RSI逆張り{'↑' if direction == 'long' else '↓'}: "
                f"RSI(5)={rsi_5:.1f}{'売られすぎ' if is_oversold else '買われすぎ'}クロスバック, "
                f"出来高スパイク{vol_spike:.1f}x, "
                f"BB位置={bb_pos:+.2f}"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[rsi_reversal] SIGNAL {ticker} {direction} "
            f"RSI={rsi_5:.1f} entry={entry_price:.0f}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        rsi = features.get("rsi_5", 50)
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

        # RSI normalised to 50 region
        if 45 < rsi < 55:
            profit = current_price - position.entry_price
            if position.direction == "short":
                profit = -profit
            if profit > 0:
                return True, f"RSI正常化({rsi:.0f}) – 利益確定"

        # RSI re-extreme in wrong direction
        if position.direction == "long" and rsi > 80:
            return True, f"RSI逆方向過熱({rsi:.0f}) – 決済"
        if position.direction == "short" and rsi < 20:
            return True, f"RSI逆方向過熱({rsi:.0f}) – 決済"

        # Time
        if position.holding_minutes > 25:
            return True, "時間切れ25分 – 短期逆張りポジション決済"

        return False, ""
