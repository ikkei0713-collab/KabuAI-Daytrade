"""Opening Range Breakout strategy (5分足ブレイク).

Defines the opening range as the high and low of the first N-minute
window (configurable: 5/15/30 min).  Enters long on a break above
the high or short on a break below the low.

Enhancements:
- Configurable OR window (5min/15min/30min); for daily data uses first bar
- Integration with RegimeDetector (via features["regime_result"])
- Integration with StockSelector score (via features["selector_score"])
- ATR-based stops and targets
- Time-based exit with configurable max holding
- Max trades per stock per day
- Multi-factor confidence calculation
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

    # Optional features that boost confidence / adjust behavior
    OPTIONAL_FEATURES = [
        "regime_result",      # RegimeResult from RegimeDetector
        "selector_score",     # 0-1 score from StockSelector
        "or_window_minutes",  # 5, 15, or 30 — defaults to 5
        "daily_trade_count",  # how many ORB trades already taken on this ticker today
        "market_trend",       # "up" / "down" / "neutral" from daily data
        "prev_day_atr",       # previous day ATR for stop calculation
        "sma20_slope",        # slope of 20-day SMA (positive = uptrend)
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
                # New parameters
                "or_window_minutes": 5,         # default OR window
                "max_trades_per_stock": 2,      # max ORB entries per ticker per day
                "max_holding_minutes": 60,       # time-based exit (extended from 30)
                "atr_stop_multiple": 1.5,        # stop = entry -/+ ATR * multiple
                "atr_target_multiple": 2.0,      # target = entry +/- ATR * multiple
                "use_atr_stops": True,           # use ATR-based stops instead of range-based
                "min_confidence": 0.45,          # minimum confidence to generate signal
                "regime_weight_threshold": 0.3,  # skip if regime weight for orb < this
                "min_selector_score": 0.0,       # minimum StockSelector score (0=no filter)
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

        # --- Max trades per stock per day ---
        daily_trade_count = features.get("daily_trade_count", 0)
        max_trades = params.get("max_trades_per_stock", 2)
        if daily_trade_count >= max_trades:
            logger.debug(f"[orb] {ticker}: max trades reached ({daily_trade_count}/{max_trades})")
            return None

        # --- Regime-based gating ---
        regime_result = features.get("regime_result")
        regime_orb_weight = 1.0  # default: no penalty
        position_scale = 1.0
        if regime_result is not None:
            # regime_result is a RegimeResult dataclass
            regime_orb_weight = regime_result.strategy_weights.get("orb", 0.5)
            position_scale = regime_result.position_scale
            threshold = params.get("regime_weight_threshold", 0.3)
            if regime_orb_weight < threshold:
                logger.debug(
                    f"[orb] {ticker}: regime '{regime_result.regime}' "
                    f"orb weight {regime_orb_weight:.2f} below threshold {threshold}"
                )
                return None

        # --- StockSelector score filter ---
        selector_score = features.get("selector_score", 1.0)
        min_selector = params.get("min_selector_score", 0.0)
        if selector_score < min_selector:
            logger.debug(f"[orb] {ticker}: selector_score {selector_score:.2f} below {min_selector}")
            return None

        # --- OR window info (for logging/reason) ---
        or_window = features.get("or_window_minutes", params.get("or_window_minutes", 5))

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
        use_atr_stops = params.get("use_atr_stops", True)
        atr_stop_mult = params.get("atr_stop_multiple", 1.5)
        atr_target_mult = params.get("atr_target_multiple", 2.0)

        direction = None
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0

        if current_close > or_high + buffer:
            direction = "long"
            entry_price = or_high + buffer
            if use_atr_stops:
                stop_price = entry_price - atr * atr_stop_mult
                target_price = entry_price + atr * atr_target_mult
            else:
                stop_price = or_low - buffer
                target_price = entry_price + or_size * target_mult
        elif current_close < or_low - buffer:
            direction = "short"
            entry_price = or_low - buffer
            if use_atr_stops:
                stop_price = entry_price + atr * atr_stop_mult
                target_price = entry_price - atr * atr_target_mult
            else:
                stop_price = or_high + buffer
                target_price = entry_price - or_size * target_mult

        # Continuation filter: require prior bar to also show direction
        if len(data) >= 3:
            prev_bar = data.iloc[-2]
            prev_close = float(prev_bar["close"])
            prev_open = float(prev_bar["open"])
            if direction == "long" and prev_close < prev_open:
                # Previous bar was bearish — this is a reversal, not continuation
                # Still allow but reduce confidence later
                pass
            elif direction == "short" and prev_close > prev_open:
                pass

        if direction is None:
            return None

        # Confirm with candle body
        body = abs(float(latest["close"]) - float(latest["open"]))
        candle_range = float(latest["high"]) - float(latest["low"])
        if candle_range > 0 and body / candle_range < 0.4:
            logger.debug(f"[orb] {ticker}: breakout candle has weak body ratio")
            return None

        # --- Multi-factor confidence calculation ---
        confidence = 0.40  # base confidence

        # Factor 1: Volume strength (0 to +0.15)
        if volume_ratio > 3.0:
            confidence += 0.15
        elif volume_ratio > 2.5:
            confidence += 0.10
        elif volume_ratio > 2.0:
            confidence += 0.05

        # Factor 2: Candle body quality (0 to +0.10)
        body_ratio = body / candle_range if candle_range > 0 else 0
        if body_ratio > 0.7:
            confidence += 0.10
        elif body_ratio > 0.5:
            confidence += 0.05

        # Factor 3: OR range relative to ATR (0 to +0.10)
        # Best when OR is 30-70% of ATR (meaningful but not exhausted)
        or_atr_ratio = or_size / atr if atr > 0 else 0
        if 0.3 <= or_atr_ratio <= 0.7:
            confidence += 0.10
        elif 0.2 <= or_atr_ratio <= 0.8:
            confidence += 0.05

        # Factor 4: Regime alignment (0 to +0.10)
        if regime_result is not None:
            if regime_orb_weight >= 0.8:
                confidence += 0.10
            elif regime_orb_weight >= 0.5:
                confidence += 0.05
            # Penalize if regime is unfavorable
            if regime_orb_weight < 0.4:
                confidence -= 0.05

        # Factor 5: StockSelector score (0 to +0.10)
        if selector_score >= 0.8:
            confidence += 0.10
        elif selector_score >= 0.6:
            confidence += 0.05

        # Factor 6: Trend alignment from daily data (0 to +0.05)
        market_trend = features.get("market_trend")
        if market_trend is not None:
            if direction == "long" and market_trend == "up":
                confidence += 0.05
            elif direction == "short" and market_trend == "down":
                confidence += 0.05
            elif direction == "long" and market_trend == "down":
                confidence -= 0.05
            elif direction == "short" and market_trend == "up":
                confidence -= 0.05

        # Factor 7: Continuation confirmation (0 to +0.10)
        if len(data) >= 3:
            prev_bar = data.iloc[-2]
            prev_close_f = float(prev_bar["close"])
            prev_open_f = float(prev_bar["open"])
            if direction == "long" and prev_close_f > prev_open_f:
                confidence += 0.10  # Prior bar bullish = continuation confirmed
            elif direction == "short" and prev_close_f < prev_open_f:
                confidence += 0.10
            else:
                confidence -= 0.05  # Reversal breakout = lower confidence

        confidence = round(min(max(confidence, 0.1), 0.95), 2)

        # --- Minimum confidence gate ---
        min_confidence = params.get("min_confidence", 0.45)
        if confidence < min_confidence:
            logger.debug(f"[orb] {ticker}: confidence {confidence:.2f} below minimum {min_confidence}")
            return None

        # --- Position sizing with regime scale ---
        capital = 10_000_000
        shares = self.calculate_position_size(entry_price, atr, capital)
        if position_scale != 1.0:
            # Adjust shares by regime position scale, round to 100-share units
            adjusted = int(shares * position_scale)
            adjusted = max(100, (adjusted // 100) * 100)
            shares = adjusted

        signal = TradeSignal(
            ticker=ticker,
            direction=direction,
            strategy_name=self.name,
            entry_price=round(entry_price, 1),
            stop_loss=round(stop_price, 1),
            take_profit=round(target_price, 1),
            confidence=confidence,
            entry_reason=(
                f"{or_window}分足ブレイク{'↑' if direction == 'long' else '↓'}: "
                f"OR [{or_low:.0f}-{or_high:.0f}] ({or_size:.0f}円), "
                f"出来高比{volume_ratio:.1f}x, ATR={atr:.0f}"
                + (f", レジーム={regime_result.regime}" if regime_result else "")
                + (f", 選定スコア={selector_score:.0%}" if selector_score < 1.0 else "")
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[orb] SIGNAL {ticker} {direction} entry={entry_price:.0f} "
            f"conf={confidence:.0%} regime_w={regime_orb_weight:.2f}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        params = self.config.parameter_set
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

        # Trailing stop: move stop to breakeven after 50% of target reached
        atr = features.get("atr", 0)
        if atr > 0 and position.entry_price > 0:
            if position.direction == "long":
                profit = current_price - position.entry_price
                half_target = (position.take_profit - position.entry_price) * 0.5
                if profit >= half_target and current_price < position.entry_price + atr * 0.3:
                    return True, f"トレーリング – 利益縮小 ({current_price:.0f})"
            elif position.direction == "short":
                profit = position.entry_price - current_price
                half_target = (position.entry_price - position.take_profit) * 0.5
                if profit >= half_target and current_price > position.entry_price - atr * 0.3:
                    return True, f"トレーリング – 利益縮小 ({current_price:.0f})"

        # Time-based exit with configurable max holding
        max_holding = params.get("max_holding_minutes", 60)
        if position.holding_minutes > max_holding:
            return True, f"時間切れ{max_holding}分"

        # Earlier time exit if not profitable
        if position.holding_minutes > max_holding * 0.5:
            pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
            if position.direction == "short":
                pnl_pct = -pnl_pct
            if pnl_pct < 0.3:
                return True, f"時間切れ{int(max_holding * 0.5)}分（含み益不足）"

        return False, ""
