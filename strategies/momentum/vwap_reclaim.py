"""VWAP Reclaim strategy.

Fires when a stock drops below VWAP, stays below for at least 15 minutes,
then reclaims VWAP with strong volume.  Entry is on the close above VWAP;
stop is below the recent low; target is the day's high or 1.5x ATR.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class VWAPReclaimStrategy(BaseStrategy):
    """VWAP奪回 – buy on VWAP reclaim after extended time below."""

    REQUIRED_FEATURES = [
        "vwap",
        "vwap_distance",
        "time_below_vwap",
        "volume_at_reclaim",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="vwap_reclaim",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_time_below_vwap_min": 15,       # NOTE: 擬似特徴量 (日足推定=30固定)
                "min_volume_at_reclaim": 1.5,         # NOTE: 擬似特徴量 (vol*1.2)
                "target_atr_multiple": 1.2,           # 1.5→1.2: 利確を手前に寄せる (保守的)
                "reclaim_buffer_pct": 0.15,           # 0.1→0.15: 飛びつき抑制
                "max_distance_from_vwap_pct": 0.8,    # 2.0→0.8: VWAP近傍のみ許可
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
        vwap: float = features["vwap"]
        vwap_dist: float = features["vwap_distance"]
        time_below: float = features["time_below_vwap"]
        vol_reclaim: float = features["volume_at_reclaim"]
        atr: float = features["atr"]

        min_time = params.get("min_time_below_vwap_min", 15)
        min_vol = params.get("min_volume_at_reclaim", 1.5)

        # Must have been below VWAP for enough time
        if time_below < min_time:
            logger.debug(f"[vwap_reclaim] {ticker}: below VWAP only {time_below:.0f} min < {min_time}")
            return None

        # Current price must now be above VWAP
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        reclaim_buf = vwap * params.get("reclaim_buffer_pct", 0.1) / 100
        if current_price < vwap + reclaim_buf:
            logger.debug(f"[vwap_reclaim] {ticker}: price {current_price:.0f} not yet above VWAP {vwap:.0f}")
            return None

        # Previous bar must have been below VWAP (confirming the reclaim just happened)
        prev_bar = data.iloc[-2]
        if float(prev_bar["close"]) > vwap:
            logger.debug(f"[vwap_reclaim] {ticker}: previous bar already above VWAP – stale signal")
            return None

        # Volume confirmation
        if vol_reclaim < min_vol:
            logger.debug(f"[vwap_reclaim] {ticker}: reclaim volume ratio {vol_reclaim:.2f} < {min_vol}")
            return None

        # Don't chase if too far above VWAP already
        max_dist = params.get("max_distance_from_vwap_pct", 2.0)
        dist_pct = (current_price - vwap) / vwap * 100
        if dist_pct > max_dist:
            return None

        # Find recent low for stop
        lookback = min(len(data), 6)
        recent_low = float(data.iloc[-lookback:]["low"].min())
        day_high = float(data["high"].max())

        entry_price = current_price
        stop_price = recent_low - 1.0
        atr_target = entry_price + atr * params.get("target_atr_multiple", 1.5)
        target_price = max(day_high, atr_target)

        if stop_price >= entry_price:
            return None

        # Confidence
        confidence = 0.50
        if time_below > 30:
            confidence += 0.10
        if vol_reclaim > 2.5:
            confidence += 0.10
        if atr > 15:
            confidence += 0.05
        # Stronger if reclaim candle has big body
        body = abs(float(latest["close"]) - float(latest["open"]))
        candle_range = float(latest["high"]) - float(latest["low"])
        if candle_range > 0 and body / candle_range > 0.6:
            confidence += 0.10
        # Event weighting: boost confidence if TDnet event present
        event_type = features.get("event_type", "")
        if event_type and event_type not in ("", 0, 0.0):
            confidence += 0.15  # Strong boost for event-driven VWAP reclaim
            if features.get("event_magnitude", 0) > 0.5:
                confidence += 0.05

        # Regime alignment
        regime_result = features.get("regime_result")
        if regime_result is not None:
            vwap_weight = regime_result.strategy_weights.get("vwap_reclaim", 0.5)
            if vwap_weight >= 0.7:
                confidence += 0.05
            elif vwap_weight < 0.3:
                confidence -= 0.10

        # Trend follow filter gate (保守的チューニング)
        # EMA9>EMA21, close>VWAP, strength>0.45, vol_trend>1.2 で加点
        # 不通過時は大幅減点 → MIN_CONFIDENCE 0.65 で実質ブロック
        is_trending = features.get("_is_trending", False)
        trend_dir = features.get("_trend_direction", "none")
        if is_trending and trend_dir == "up":
            trend_str = features.get("trend_strength", 0)
            vol_trend = features.get("volume_trend", 1.0)
            if trend_str > 0.45 and vol_trend > 1.2:
                confidence += 0.10  # trend filter 完全通過
            else:
                confidence += 0.03  # 部分通過
        else:
            confidence -= 0.15  # trend 不通過 → 厳しく減点

        # selector_score フィルタ: 上位銘柄のみ許可
        selector_score = features.get("selector_score", 0)
        if selector_score < 0.25:
            confidence -= 0.10

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
                f"VWAP奪回: {time_below:.0f}分間VWAP下→再突破, "
                f"出来高{vol_reclaim:.1f}x, "
                f"VWAP={vwap:.0f}"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[vwap_reclaim] SIGNAL {ticker} long entry={entry_price:.0f} "
            f"stop={stop_price:.0f} target={target_price:.0f}"
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
        atr = features.get("atr", 0)

        # Stop
        if current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if current_price >= position.take_profit:
            return True, f"利確到達 ({current_price:.0f})"

        # Lost VWAP again: give a small buffer
        if vwap > 0 and current_price < vwap - atr * 0.3:
            return True, f"VWAP再度割れ ({current_price:.0f} < {vwap:.0f})"

        # Trailing stop
        if atr > 0:
            profit = current_price - position.entry_price
            if profit > atr:
                trail = current_price - atr * 0.8
                if current_price <= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"

        # Time
        if position.holding_minutes > 60:
            pnl = current_price - position.entry_price
            if pnl < atr * 0.3:
                return True, "時間切れ60分 – 利益不十分"

        return False, ""
