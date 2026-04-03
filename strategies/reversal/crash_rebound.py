"""急落リバウンド strategy.

Fires when a stock drops more than 5 % intraday within 30 minutes
and volume surges above 3x average.  Entry is on the first green
candle after selling exhaustion.  Stop is below the crash low;
target is a 50 % retracement of the drop.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class CrashReboundStrategy(BaseStrategy):
    """急落リバウンド – catch the dead-cat bounce."""

    REQUIRED_FEATURES = [
        "intraday_drop_pct",
        "volume_surge",
        "selling_exhaustion",
        "support_level",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="crash_rebound",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="volatile",
            parameter_set={
                "min_drop_pct": 5.0,
                "max_drop_pct": 20.0,
                "min_volume_surge": 2.5,
                "selling_exhaustion_threshold": 0.7,
                "retracement_target": 0.50,
                "max_drop_minutes": 30,
                "stop_buffer_pct": 0.5,
                "blocked_regimes": [],  # crash_rebound はどのレジームでも発火可能
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

        drop_pct: float = features["intraday_drop_pct"]
        vol_surge: float = features["volume_surge"]
        sell_exhaust: float = features["selling_exhaustion"]
        support: float = features["support_level"]
        atr: float = features["atr"]

        min_drop = params.get("min_drop_pct", 5.0)
        max_drop = params.get("max_drop_pct", 20.0)
        min_vol = params.get("min_volume_surge", 3.0)
        exhaust_thresh = params.get("selling_exhaustion_threshold", 0.7)

        # Must have a significant rapid drop
        if abs(drop_pct) < min_drop or abs(drop_pct) > max_drop:
            logger.debug(f"[crash_rebound] {ticker}: drop {drop_pct:.1f}% outside range")
            return None

        # Volume must confirm panic selling
        if vol_surge < min_vol:
            logger.debug(f"[crash_rebound] {ticker}: volume surge {vol_surge:.1f} < {min_vol}")
            return None

        # Selling exhaustion indicator (0-1, higher = more exhausted)
        if sell_exhaust < exhaust_thresh:
            logger.debug(
                f"[crash_rebound] {ticker}: selling exhaustion "
                f"{sell_exhaust:.2f} < {exhaust_thresh}"
            )
            return None

        # Must see a green candle (first sign of bounce)
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        current_open = float(latest["open"])

        if current_price <= current_open:
            logger.debug(f"[crash_rebound] {ticker}: latest candle not green")
            return None

        # Find crash low and pre-crash high
        crash_low = float(data["low"].min())
        day_high = float(data["high"].max())
        drop_size = day_high - crash_low

        if drop_size <= 0:
            return None

        # Entry at current price (first green candle)
        entry_price = current_price
        stop_buffer = entry_price * params.get("stop_buffer_pct", 0.5) / 100
        stop_price = crash_low - stop_buffer

        # Target: 50% retracement of the drop
        retrace = params.get("retracement_target", 0.50)
        target_price = crash_low + drop_size * retrace

        if target_price <= entry_price:
            # Already retraced past target
            target_price = entry_price + atr
        if stop_price >= entry_price:
            return None

        # Near support adds confidence
        near_support = support > 0 and abs(crash_low - support) / support < 0.01

        # Confidence
        confidence = 0.40  # Inherently risky strategy
        if sell_exhaust > 0.85:
            confidence += 0.10
        if vol_surge > 5.0:
            confidence += 0.10
        if near_support:
            confidence += 0.10
        # Green candle quality
        body = current_price - current_open
        wick_below = current_open - float(latest["low"])
        if body > 0 and wick_below > body:
            confidence += 0.05  # Hammer-like
        if abs(drop_pct) > 8:
            confidence += 0.05  # Bigger drops = more rebound potential
        confidence = min(confidence, 0.80)

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
                f"急落リバウンド: {drop_pct:.1f}%急落後の初陽線, "
                f"出来高{vol_surge:.1f}x急増, "
                f"売り疲れ{sell_exhaust:.2f}, "
                f"{'サポート付近' if near_support else ''}"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[crash_rebound] SIGNAL {ticker} long entry={entry_price:.0f} "
            f"crash_low={crash_low:.0f} target_retrace={retrace:.0%}"
        )
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        atr = features.get("atr", 0)

        # Hard stop: below crash low
        if current_price <= position.stop_loss:
            return True, f"ストップロス – 安値割れ ({current_price:.0f})"

        # Target hit
        if current_price >= position.take_profit:
            return True, f"利確 – リトレースメント到達 ({current_price:.0f})"

        # Renewed selling: new lower low
        if len(current_data) >= 3:
            recent_lows = [float(r["low"]) for _, r in current_data.iloc[-3:].iterrows()]
            if all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows))):
                return True, f"再下落 – 安値切り下げ ({current_price:.0f})"

        # Partial profit: lock in if gained > 50% of target move
        if atr > 0:
            profit = current_price - position.entry_price
            if profit > atr * 0.8:
                # Tighten stop to entry
                if current_price < position.entry_price + atr * 0.2:
                    return True, f"利益保護 – ブレイクイーブン割れ"

        # Time: rebound should happen quickly
        if position.holding_minutes > 20:
            if current_price <= position.entry_price:
                return True, "時間切れ20分 – リバウンドなし"
        if position.holding_minutes > 40:
            return True, "時間切れ40分 – ポジション決済"

        return False, ""
