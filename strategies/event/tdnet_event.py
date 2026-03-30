"""TDnet Event strategy.

Monitors TDnet for material disclosures such as 上方修正 (upward revision)
and 自社株買い (share buyback).  Enters immediately on a material positive
disclosure.  Target varies by event type.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


# Event type configuration with expected targets
EVENT_PROFILES = {
    "上方修正": {
        "target_pct": 7.5,
        "stop_pct": 2.0,
        "base_confidence": 0.60,
        "description": "業績上方修正",
    },
    "自社株買い": {
        "target_pct": 4.0,
        "stop_pct": 2.0,
        "base_confidence": 0.55,
        "description": "自社株買い発表",
    },
    "増配": {
        "target_pct": 5.0,
        "stop_pct": 2.0,
        "base_confidence": 0.55,
        "description": "増配発表",
    },
    "株式分割": {
        "target_pct": 5.0,
        "stop_pct": 2.5,
        "base_confidence": 0.50,
        "description": "株式分割発表",
    },
    "業務提携": {
        "target_pct": 4.0,
        "stop_pct": 2.5,
        "base_confidence": 0.45,
        "description": "業務提携発表",
    },
    "新製品": {
        "target_pct": 3.5,
        "stop_pct": 2.0,
        "base_confidence": 0.40,
        "description": "新製品・サービス発表",
    },
}


class TDnetEventStrategy(BaseStrategy):
    """TDnetイベント – trade on material TDnet disclosures."""

    REQUIRED_FEATURES = [
        "event_type",
        "event_magnitude",
        "market_cap",
        "historical_event_response",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="tdnet_event",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "supported_events": list(EVENT_PROFILES.keys()),
                "min_market_cap_billion": 10.0,
                "max_market_cap_billion": 5000.0,
                "min_event_magnitude": 0.3,
                "min_historical_response": 0.0,
                "blocked_regimes": [],  # イベント駆動はレジーム不問
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=1):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        params = self.config.parameter_set

        if not self._check_regime_filter(features):
            return None

        event_type: str = features["event_type"]
        event_mag: float = features["event_magnitude"]
        market_cap: float = features["market_cap"]  # in billions of yen
        hist_response: float = features["historical_event_response"]
        atr: float = features["atr"]

        supported = params.get("supported_events", list(EVENT_PROFILES.keys()))
        if event_type not in supported:
            logger.debug(f"[tdnet_event] {ticker}: event '{event_type}' not supported")
            return None

        # Market cap filter
        min_cap = params.get("min_market_cap_billion", 10.0)
        max_cap = params.get("max_market_cap_billion", 5000.0)
        if market_cap < min_cap or market_cap > max_cap:
            logger.debug(
                f"[tdnet_event] {ticker}: market_cap {market_cap:.0f}B outside "
                f"[{min_cap}, {max_cap}]"
            )
            return None

        # Event magnitude filter
        min_mag = params.get("min_event_magnitude", 0.3)
        if event_mag < min_mag:
            logger.debug(f"[tdnet_event] {ticker}: magnitude {event_mag:.2f} < {min_mag}")
            return None

        # Historical response check
        min_hist = params.get("min_historical_response", 0.0)
        if hist_response < min_hist:
            logger.debug(
                f"[tdnet_event] {ticker}: historical response "
                f"{hist_response:.2f} < {min_hist}"
            )
            return None

        # Get event profile
        profile = EVENT_PROFILES.get(event_type, EVENT_PROFILES["業務提携"])
        latest = data.iloc[-1]
        current_price = float(latest["close"])

        entry_price = current_price
        target_pct = profile["target_pct"]
        stop_pct = profile["stop_pct"]

        # Scale target by event magnitude
        target_pct *= (1 + event_mag)
        target_pct = min(target_pct, 15.0)

        target_price = entry_price * (1 + target_pct / 100)
        stop_price = entry_price * (1 - stop_pct / 100)

        # Confidence
        confidence = profile["base_confidence"]

        # Magnitude boost
        if event_mag > 0.7:
            confidence += 0.10
        elif event_mag > 0.5:
            confidence += 0.05

        # Historical response boost
        if hist_response > 3.0:
            confidence += 0.10
        elif hist_response > 1.5:
            confidence += 0.05

        # Small cap tends to move more
        if market_cap < 100:
            confidence += 0.05

        # Volume confirmation from latest bar
        if len(data) >= 2:
            prev_vol = float(data.iloc[-2]["volume"])
            curr_vol = float(latest["volume"])
            if prev_vol > 0 and curr_vol / prev_vol > 2.0:
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
                f"TDnet{profile['description']}: "
                f"マグニチュード{event_mag:.2f}, "
                f"時価総額{market_cap:.0f}億円, "
                f"過去反応{hist_response:+.1f}%"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[tdnet_event] SIGNAL {ticker} {event_type} "
            f"mag={event_mag:.2f} target={target_pct:.1f}%"
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

        # Stop
        if current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        # Trailing stop after significant move
        if atr > 0:
            profit = current_price - position.entry_price
            if profit > atr * 2:
                trail = current_price - atr * 1.0
                if trail > position.entry_price and current_price <= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"

        # Momentum fading: consecutive red candles after initial pop
        if len(current_data) >= 3:
            recent = current_data.iloc[-3:]
            red_count = sum(
                1 for _, r in recent.iterrows()
                if float(r["close"]) < float(r["open"])
            )
            if red_count >= 3 and current_price < position.entry_price * 1.01:
                return True, "モメンタム減退 – 3連続陰線"

        # Event trades can hold longer: 2 hours
        if position.holding_minutes > 120:
            if current_price > position.entry_price:
                return True, f"時間切れ2時間 – 利益確定 ({current_price:.0f})"
            return True, f"時間切れ2時間 – 決済 ({current_price:.0f})"

        return False, ""
