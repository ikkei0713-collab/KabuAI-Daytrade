"""材料初動 (Catalyst Initial Move) strategy.

Captures the first move on a new catalyst or breaking news.
Entry is taken early in the move with volume confirmation.
Stop is 1.5 % below entry; target scales with catalyst magnitude.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


# Catalyst magnitude profiles
CATALYST_PROFILES = {
    "regulatory_approval": {"target_pct": 8.0, "base_confidence": 0.55},
    "major_contract": {"target_pct": 6.0, "base_confidence": 0.50},
    "patent_grant": {"target_pct": 4.0, "base_confidence": 0.45},
    "partnership": {"target_pct": 4.0, "base_confidence": 0.45},
    "sector_tailwind": {"target_pct": 3.0, "base_confidence": 0.40},
    "analyst_upgrade": {"target_pct": 3.0, "base_confidence": 0.45},
    "government_policy": {"target_pct": 5.0, "base_confidence": 0.45},
    "general_positive": {"target_pct": 3.0, "base_confidence": 0.40},
}


class CatalystInitialStrategy(BaseStrategy):
    """材料初動 – capture the first move on a new catalyst."""

    REQUIRED_FEATURES = [
        "news_sentiment",
        "volume_surge",
        "price_acceleration",
        "historical_catalyst_response",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="catalyst_initial",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_news_sentiment": 0.5,
                "min_volume_surge": 2.0,
                "min_price_acceleration": 0.3,
                "stop_pct": 1.5,
                "max_chase_pct": 5.0,
                "catalyst_type_key": "catalyst_type",
                "blocked_regimes": [],  # カタリストはレジーム不問
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

        sentiment: float = features["news_sentiment"]
        vol_surge: float = features["volume_surge"]
        price_accel: float = features["price_acceleration"]
        hist_response: float = features["historical_catalyst_response"]
        atr: float = features["atr"]

        min_sent = params.get("min_news_sentiment", 0.5)
        min_vol = params.get("min_volume_surge", 2.0)
        min_accel = params.get("min_price_acceleration", 0.3)

        # Positive news sentiment required
        if sentiment < min_sent:
            logger.debug(f"[catalyst] {ticker}: sentiment {sentiment:.2f} < {min_sent}")
            return None

        # Volume surge confirms market attention
        if vol_surge < min_vol:
            logger.debug(f"[catalyst] {ticker}: volume surge {vol_surge:.1f} < {min_vol}")
            return None

        # Price acceleration: move is picking up speed
        if price_accel < min_accel:
            logger.debug(f"[catalyst] {ticker}: acceleration {price_accel:.2f} < {min_accel}")
            return None

        # Don't chase: check if we're too late
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        first_price = float(data.iloc[0]["open"])
        move_pct = (current_price - first_price) / first_price * 100 if first_price > 0 else 0
        max_chase = params.get("max_chase_pct", 5.0)
        if move_pct > max_chase:
            logger.debug(
                f"[catalyst] {ticker}: already moved {move_pct:.1f}% – too late to chase"
            )
            return None

        # Get catalyst profile
        cat_type = features.get(
            params.get("catalyst_type_key", "catalyst_type"),
            "general_positive",
        )
        profile = CATALYST_PROFILES.get(cat_type, CATALYST_PROFILES["general_positive"])

        stop_pct = params.get("stop_pct", 1.5)
        entry_price = current_price
        stop_price = entry_price * (1 - stop_pct / 100)

        # Scale target by magnitude and historical response
        base_target = profile["target_pct"]
        if hist_response > 3.0:
            base_target *= 1.3
        elif hist_response > 1.5:
            base_target *= 1.1
        target_price = entry_price * (1 + base_target / 100)

        if stop_price >= entry_price:
            return None

        # Confidence
        confidence = profile["base_confidence"]
        if sentiment > 0.8:
            confidence += 0.10
        if vol_surge > 4.0:
            confidence += 0.10
        if price_accel > 0.6:
            confidence += 0.05
        if hist_response > 2.0:
            confidence += 0.05
        # Early in the move
        if move_pct < 2.0:
            confidence += 0.05
        confidence = min(confidence, 0.85)

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
                f"材料初動({cat_type}): "
                f"センチメント{sentiment:.2f}, "
                f"出来高{vol_surge:.1f}x急増, "
                f"加速度{price_accel:.2f}, "
                f"既存変動{move_pct:+.1f}%"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[catalyst] SIGNAL {ticker} long catalyst={cat_type} "
            f"sentiment={sentiment:.2f} vol={vol_surge:.1f}x"
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
        vol_surge = features.get("volume_surge", 1.0)

        # Stop
        if current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target
        if current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        # Trailing stop after significant profit
        if atr > 0:
            profit = current_price - position.entry_price
            if profit > atr * 1.5:
                trail = current_price - atr * 1.0
                if trail > position.entry_price and current_price <= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"

        # Volume drying up: catalyst fading
        if vol_surge < 0.8 and position.holding_minutes > 15:
            profit_pct = (current_price - position.entry_price) / position.entry_price * 100
            if profit_pct < 1.0:
                return True, f"出来高減少 – 材料反応薄 (出来高比{vol_surge:.1f})"

        # Negative price action: consecutive bearish candles
        if len(current_data) >= 4:
            recent = current_data.iloc[-4:]
            bearish = sum(
                1 for _, r in recent.iterrows()
                if float(r["close"]) < float(r["open"])
            )
            if bearish >= 4:
                return True, "4連続陰線 – モメンタム消失"

        # Time
        if position.holding_minutes > 60:
            profit_pct = (current_price - position.entry_price) / position.entry_price * 100
            if profit_pct < 0.5:
                return True, f"時間切れ60分 – 利益不十分 ({profit_pct:.1f}%)"
        if position.holding_minutes > 120:
            return True, f"時間切れ2時間 – 決済 ({current_price:.0f})"

        return False, ""
