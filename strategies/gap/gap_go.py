"""Gap Go strategy (ギャップアップブレイク).

Triggers when a stock gaps up more than 2 % from the previous close.
Waits for the first 5-minute candle to close above the gap level, then
enters on a break above the 5-min high.  Stop is placed below the
5-min low and the target is 2x ATR or the gap size, whichever is larger.

Best in: bull market, high-volume sessions.
"""

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class GapGoStrategy(BaseStrategy):
    """ギャップアップブレイク – ride the gap continuation."""

    REQUIRED_FEATURES = [
        "gap_pct",
        "volume_ratio",
        "pre_market_volume",
        "sector_momentum",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    # -----------------------------------------------------------------
    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="gap_go",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                "min_gap_pct": 2.0,
                "min_volume_ratio": 1.5,
                "target_atr_multiple": 2.0,
                "max_gap_pct": 10.0,
                "min_pre_market_volume": 50000,
                "confirmation_candles": 1,
                "blocked_regimes": ["trend_down"],
            },
        )

    # -----------------------------------------------------------------
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

        gap_pct: float = features["gap_pct"]
        volume_ratio: float = features["volume_ratio"]
        pre_market_vol: float = features.get("pre_market_volume", 0)
        atr: float = features["atr"]
        sector_mom: float = features.get("sector_momentum", 0.0)

        min_gap = params.get("min_gap_pct", 2.0)
        max_gap = params.get("max_gap_pct", 10.0)
        min_vol_ratio = params.get("min_volume_ratio", 1.5)
        min_pre_vol = params.get("min_pre_market_volume", 50000)

        # --- Filter conditions ---
        if gap_pct < min_gap or gap_pct > max_gap:
            logger.debug(f"[gap_go] {ticker}: gap {gap_pct:.2f}% outside [{min_gap}, {max_gap}]")
            return None
        if volume_ratio < min_vol_ratio:
            logger.debug(f"[gap_go] {ticker}: volume_ratio {volume_ratio:.2f} < {min_vol_ratio}")
            return None
        if pre_market_vol < min_pre_vol:
            logger.debug(f"[gap_go] {ticker}: pre_market_volume {pre_market_vol} < {min_pre_vol}")
            return None

        # --- First 5-min candle must close above gap level ---
        first_candle = data.iloc[0]
        prev_close = first_candle["open"] / (1 + gap_pct / 100.0)
        gap_level = prev_close * (1 + gap_pct / 100.0)

        if first_candle["close"] < gap_level:
            logger.debug(f"[gap_go] {ticker}: first candle close below gap level")
            return None

        # --- Entry / stop / target ---
        entry_price = float(first_candle["high"]) + 1.0  # 1 yen above 5-min high
        stop_price = float(first_candle["low"]) - 1.0
        gap_size = entry_price - prev_close
        atr_target = atr * params.get("target_atr_multiple", 2.0)
        target_distance = max(gap_size, atr_target)
        target_price = entry_price + target_distance

        if stop_price >= entry_price:
            return None

        # --- Confidence ---
        confidence = 0.5
        if volume_ratio > 2.5:
            confidence += 0.1
        if sector_mom > 0:
            confidence += 0.1
        if gap_pct > 3.0:
            confidence += 0.1
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
                f"ギャップアップ{gap_pct:.1f}% ブレイク: "
                f"出来高比{volume_ratio:.1f}x, "
                f"セクターモメンタム{sector_mom:+.2f}"
            ),
            features_snapshot=features,
        )
        logger.info(f"[gap_go] SIGNAL {ticker} entry={entry_price} stop={stop_price} target={target_price}")
        return signal

    # -----------------------------------------------------------------
    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        atr = features.get("atr", 0)

        # Hard stop
        if current_price <= position.stop_loss:
            logger.info(f"[gap_go] EXIT {position.ticker}: stop hit at {current_price}")
            return True, f"ストップロス到達 ({current_price:.0f} <= {position.stop_loss:.0f})"

        # Target hit
        if current_price >= position.take_profit:
            logger.info(f"[gap_go] EXIT {position.ticker}: target hit at {current_price}")
            return True, f"利確到達 ({current_price:.0f} >= {position.take_profit:.0f})"

        # Trail stop: if price moved > 1 ATR in profit, tighten stop
        if atr > 0:
            profit = current_price - position.entry_price
            if profit > atr:
                trailing_stop = current_price - atr
                if trailing_stop > position.stop_loss and current_price <= trailing_stop:
                    return True, f"トレーリングストップ ({current_price:.0f})"

        # Gap fill – price returned to pre-gap level
        gap_pct = features.get("gap_pct", 0)
        if gap_pct > 0:
            prev_close = position.entry_price / (1 + gap_pct / 100.0)
            if current_price <= prev_close:
                return True, f"ギャップ埋め ({current_price:.0f})"

        # Time exit: if position held > 60 min and profit is marginal
        holding = position.holding_minutes
        if holding > 60 and current_price < position.entry_price * 1.005:
            return True, f"時間切れ – 60分超でほぼ利益なし"

        return False, ""
