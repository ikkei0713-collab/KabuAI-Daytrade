"""決算モメンタム (Earnings Momentum) strategy.

Enters on post-earnings price momentum when a stock gaps up after
reporting strong earnings and shows continuation.  Stop is below
the post-earnings low; target uses a 2x ATR trailing stop.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class EarningsMomentumStrategy(BaseStrategy):
    """決算モメンタム – ride post-earnings momentum."""

    REQUIRED_FEATURES = [
        "earnings_surprise_pct",
        "revenue_growth",
        "guidance_change",
        "gap_pct",
        "atr",
        "volume_ratio",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="earnings_momentum",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_earnings_surprise_pct": 5.0,
                "min_revenue_growth": 0.0,
                "min_gap_pct": 1.0,
                "min_volume_ratio": 1.5,
                "trailing_atr_multiple": 2.0,
                "stop_below_post_earnings_low": True,
                "guidance_weight": 1.5,
                "blocked_regimes": [],  # 決算イベントはレジーム不問
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

        surprise: float = features["earnings_surprise_pct"]
        rev_growth: float = features["revenue_growth"]
        guidance: float = features["guidance_change"]
        gap_pct: float = features["gap_pct"]
        atr: float = features["atr"]
        volume_ratio: float = features["volume_ratio"]

        min_surprise = params.get("min_earnings_surprise_pct", 5.0)
        min_rev = params.get("min_revenue_growth", 0.0)
        min_gap = params.get("min_gap_pct", 1.0)
        min_vol = params.get("min_volume_ratio", 1.5)

        # Earnings must have beaten expectations
        if surprise < min_surprise:
            logger.debug(f"[earnings_mom] {ticker}: surprise {surprise:.1f}% < {min_surprise}%")
            return None

        # Revenue growth positive
        if rev_growth < min_rev:
            logger.debug(f"[earnings_mom] {ticker}: revenue growth {rev_growth:.1f}% < {min_rev}%")
            return None

        # Must have gapped up
        if gap_pct < min_gap:
            logger.debug(f"[earnings_mom] {ticker}: gap {gap_pct:.1f}% < {min_gap}%")
            return None

        # Volume confirmation
        if volume_ratio < min_vol:
            logger.debug(f"[earnings_mom] {ticker}: volume_ratio {volume_ratio:.1f} < {min_vol}")
            return None

        # Continuation check: price should be holding above gap level
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        first_bar = data.iloc[0]
        open_price = float(first_bar["open"])

        if current_price < open_price:
            logger.debug(f"[earnings_mom] {ticker}: price below open – no continuation")
            return None

        # Post-earnings low for stop
        post_earnings_low = float(data["low"].min())
        entry_price = current_price
        stop_price = post_earnings_low - 1.0

        # Target: trailing stop based (initial target = 2x ATR above)
        trail_mult = params.get("trailing_atr_multiple", 2.0)
        target_price = entry_price + atr * trail_mult * 2

        if stop_price >= entry_price:
            return None

        # Guidance boost
        guidance_w = params.get("guidance_weight", 1.5)
        guidance_score = guidance * guidance_w

        # Confidence
        confidence = 0.50
        if surprise > 10:
            confidence += 0.10
        if rev_growth > 10:
            confidence += 0.05
        if guidance > 0:
            confidence += 0.10
        if volume_ratio > 3.0:
            confidence += 0.10
        if gap_pct > 3.0:
            confidence += 0.05
        # Price holding above VWAP area (use open as proxy)
        if current_price > open_price * 1.005:
            confidence += 0.05
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
                f"決算モメンタム: "
                f"サプライズ{surprise:+.1f}%, "
                f"売上成長{rev_growth:+.1f}%, "
                f"ガイダンス{guidance:+.1f}, "
                f"ギャップ{gap_pct:+.1f}%, "
                f"出来高比{volume_ratio:.1f}x"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[earnings_mom] SIGNAL {ticker} long "
            f"surprise={surprise:.1f}% gap={gap_pct:.1f}%"
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
        trail_mult = self.config.parameter_set.get("trailing_atr_multiple", 2.0)

        # Stop
        if current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Trailing stop
        if atr > 0:
            day_high = float(current_data["high"].max())
            trail = day_high - atr * trail_mult
            if trail > position.entry_price and current_price <= trail:
                return True, f"トレーリングストップ ({current_price:.0f})"

        # Gap fill: price returned to pre-gap level
        gap_pct = features.get("gap_pct", 0)
        if gap_pct > 0:
            pre_gap = position.entry_price / (1 + gap_pct / 100)
            if current_price <= pre_gap:
                return True, f"ギャップ埋め – 決算反応消滅 ({current_price:.0f})"

        # Momentum check: volume drying up
        volume_ratio = features.get("volume_ratio", 1.0)
        if volume_ratio < 0.5 and position.holding_minutes > 30:
            if current_price < position.entry_price * 1.01:
                return True, f"出来高急減 (比率{volume_ratio:.1f}) – モメンタム消失"

        # Hard target
        if current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        # Time: earnings momentum can last all day
        if position.holding_minutes > 180:
            return True, f"時間切れ3時間 – 決済 ({current_price:.0f})"

        return False, ""
