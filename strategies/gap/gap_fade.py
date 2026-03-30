"""Gap Fade strategy (ギャップフェード).

Triggers when a stock gaps up more than 3 % but shows exhaustion signals.
Since short-selling is restricted in the Japanese cash market for retail,
this strategy implements a "fade buy" at a lower level after the gap
fills, or signals to avoid a long entry on an over-extended gap.

Best in: range market, over-extended gaps with low follow-through.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class GapFadeStrategy(BaseStrategy):
    """ギャップフェード – fade the over-extended gap."""

    REQUIRED_FEATURES = [
        "gap_pct",
        "volume_ratio",
        "rsi_5",
        "vwap_distance",
        "vwap",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="gap_fade",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="range",
            parameter_set={
                "min_gap_pct": 3.0,
                "max_gap_pct": 15.0,
                "rsi_exhaustion": 75.0,
                "vwap_fail_minutes": 15,
                "min_volume_ratio": 1.2,
                "target_type": "vwap",
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

        gap_pct: float = features["gap_pct"]
        volume_ratio: float = features["volume_ratio"]
        rsi_5: float = features["rsi_5"]
        vwap_dist: float = features["vwap_distance"]
        vwap: float = features["vwap"]
        atr: float = features["atr"]

        min_gap = params.get("min_gap_pct", 3.0)
        max_gap = params.get("max_gap_pct", 15.0)

        # Must have a meaningful gap up
        if gap_pct < min_gap or gap_pct > max_gap:
            return None

        # RSI must show exhaustion
        rsi_threshold = params.get("rsi_exhaustion", 75.0)
        if rsi_5 < rsi_threshold:
            logger.debug(f"[gap_fade] {ticker}: RSI {rsi_5:.1f} not exhausted (< {rsi_threshold})")
            return None

        # Price must have failed to hold above VWAP (trading below VWAP now)
        latest = data.iloc[-1]
        current_price = float(latest["close"])
        if current_price > vwap:
            logger.debug(f"[gap_fade] {ticker}: price {current_price:.0f} still above VWAP {vwap:.0f}")
            return None

        # Require enough bars to represent ~15 min
        vwap_fail_bars = params.get("vwap_fail_minutes", 15) // 5
        if len(data) < vwap_fail_bars:
            return None

        # Check recent candles – price should have been above VWAP initially
        early_candles = data.iloc[:vwap_fail_bars]
        was_above_vwap = any(float(row["close"]) > vwap for _, row in early_candles.iterrows())
        if not was_above_vwap:
            logger.debug(f"[gap_fade] {ticker}: never traded above VWAP early on")
            return None

        # --- Fade buy: enter long at lower level expecting bounce to VWAP ---
        prev_close_est = current_price / (1 + gap_pct / 100.0)
        gap_high = float(data["high"].max())

        # Entry: current price (already below VWAP)
        entry_price = current_price
        # Stop: below previous close (full gap fill risk)
        stop_price = prev_close_est - atr * 0.5
        # Target: VWAP or previous close depending on config
        target_type = params.get("target_type", "vwap")
        if target_type == "vwap":
            target_price = vwap
        else:
            target_price = prev_close_est

        if target_price <= entry_price:
            # If VWAP is below entry, target the gap midpoint
            target_price = entry_price + (gap_high - entry_price) * 0.5
        if stop_price >= entry_price:
            return None

        # Confidence calculation
        confidence = 0.45
        if rsi_5 > 80:
            confidence += 0.1
        if volume_ratio > 2.0:
            confidence += 0.1
        if vwap_dist < -1.0:
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
                f"ギャップ{gap_pct:.1f}%フェード: "
                f"RSI({rsi_5:.0f})過熱後VWAP割れ, "
                f"出来高比{volume_ratio:.1f}x, "
                f"VWAP回帰狙い"
            ),
            features_snapshot=features,
        )
        logger.info(
            f"[gap_fade] SIGNAL {ticker} fade-buy entry={entry_price:.0f} "
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

        # Stop loss
        if current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # Target reached
        if current_price >= position.take_profit:
            return True, f"利確 – VWAP/目標到達 ({current_price:.0f})"

        # VWAP reclaim – take partial exit signal
        if vwap > 0 and current_price >= vwap * 0.998:
            return True, f"VWAP到達 ({current_price:.0f} ~ {vwap:.0f})"

        # Time stop: gap fades should resolve within 30 min
        if position.holding_minutes > 30:
            if current_price < position.entry_price:
                return True, "時間切れ30分 – 含み損のため損切り"
            return True, "時間切れ30分 – ポジション決済"

        # Further breakdown – price dropping well below entry
        if atr > 0 and current_price < position.entry_price - atr * 1.5:
            return True, f"想定外の下落 ({current_price:.0f})"

        return False, ""
