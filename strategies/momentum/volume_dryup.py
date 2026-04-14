"""
Volume Dry-Up (出来高枯れ押し目買い) 戦略

パターン:
1. 高値更新 + 出来高急増の陽線 (ブレイクアウト)
2. 出来高減少の陰線 (利確・調整の押し目)
3. ここで買い → 上昇トレンド継続を狙う

ポイント:
- プライスアクションと出来高の推移が鍵
- 上昇トレンド入りの最初の押し目は勝率が高い
- テーマ株・注目株でよく出る買いシグナル
"""

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class VolumeDryUpStrategy(BaseStrategy):
    """出来高枯れ押し目買い — 高値更新陽線→出来高減陰線で買い"""

    REQUIRED_FEATURES = [
        "atr",
        "volume_ratio",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="volume_dryup",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                # ブレイクアウト足の条件
                "min_volume_spike": 1.5,        # 20日平均の1.5倍以上
                "min_breakout_body_pct": 0.3,   # 実体がレンジの30%以上 (陽線)
                "lookback_high": 20,            # N日高値を更新したか

                # 押し目足の条件
                "max_pullback_volume_ratio": 0.8,  # ブレイクアウト足の80%以下に出来高減少
                "max_pullback_depth_pct": 50,      # ブレイクアウト足の値幅の50%以内の押し

                # エグジット
                "target_atr_multiple": 1.2,     # 手数料無料期間: TP近め
                "stop_atr_multiple": 1.5,       # SL

                # レジームフィルタ
                "blocked_regimes": ["trend_down"],
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=22):
            return None
        if not self._check_regime_filter(features):
            return None

        params = self.config.parameter_set
        atr = features.get("atr", 0)
        if atr <= 0:
            return None

        # 直近3本を見る: [-3]=ブレイクアウト候補, [-2]=押し目候補, [-1]=現在
        if len(data) < 3:
            return None

        bars = data.iloc[-3:]
        breakout_bar = bars.iloc[0]   # 2日前
        pullback_bar = bars.iloc[1]   # 1日前 (昨日)
        current_bar = bars.iloc[2]    # 今日

        # --- ブレイクアウト足の判定 ---
        bo_open = float(breakout_bar["open"])
        bo_close = float(breakout_bar["close"])
        bo_high = float(breakout_bar["high"])
        bo_low = float(breakout_bar["low"])
        bo_volume = float(breakout_bar["volume"])

        # 陽線チェック
        if bo_close <= bo_open:
            logger.debug(f"[volume_dryup] {ticker}: breakout bar is not bullish")
            return None

        # 実体比率チェック
        bo_range = bo_high - bo_low
        if bo_range <= 0:
            return None
        bo_body = bo_close - bo_open
        body_pct = bo_body / bo_range
        if body_pct < params.get("min_breakout_body_pct", 0.3):
            logger.debug(f"[volume_dryup] {ticker}: body ratio {body_pct:.2f} too small")
            return None

        # 高値更新チェック
        lookback = params.get("lookback_high", 20)
        if len(data) > lookback + 2:
            prior_high = float(data.iloc[-(lookback + 2):-2]["high"].max())
            if bo_high <= prior_high:
                logger.debug(f"[volume_dryup] {ticker}: no new high ({bo_high:.0f} <= {prior_high:.0f})")
                return None

        # 出来高急増チェック
        vol_window = min(len(data) - 2, 20)
        avg_volume = float(data.iloc[-(vol_window + 2):-2]["volume"].mean())
        if avg_volume <= 0:
            return None
        volume_spike = bo_volume / avg_volume
        min_spike = params.get("min_volume_spike", 1.5)
        if volume_spike < min_spike:
            logger.debug(f"[volume_dryup] {ticker}: volume spike {volume_spike:.1f}x < {min_spike}x")
            return None

        # --- 押し目足の判定 ---
        pb_open = float(pullback_bar["open"])
        pb_close = float(pullback_bar["close"])
        pb_volume = float(pullback_bar["volume"])

        # 陰線チェック (または小さい陽線でもOK: 実質は出来高減少が重要)
        is_bearish = pb_close < pb_open
        is_small = abs(pb_close - pb_open) < bo_body * 0.5

        if not (is_bearish or is_small):
            logger.debug(f"[volume_dryup] {ticker}: pullback bar not bearish/small")
            return None

        # 出来高減少チェック
        max_vol_ratio = params.get("max_pullback_volume_ratio", 0.8)
        if bo_volume > 0 and pb_volume / bo_volume > max_vol_ratio:
            logger.debug(
                f"[volume_dryup] {ticker}: pullback volume ratio "
                f"{pb_volume / bo_volume:.2f} > {max_vol_ratio}"
            )
            return None

        # 押し目の深さチェック (ブレイクアウト足の値幅に対して)
        max_depth_pct = params.get("max_pullback_depth_pct", 50)
        pullback_depth = bo_close - pb_close  # 正ならbo_closeから下がった
        if bo_body > 0 and (pullback_depth / bo_body * 100) > max_depth_pct:
            logger.debug(f"[volume_dryup] {ticker}: pullback too deep ({pullback_depth / bo_body * 100:.0f}%)")
            return None

        # --- エントリー ---
        current_price = features.get("current_price", float(current_bar["close"]))
        entry_price = current_price

        target_mult = params.get("target_atr_multiple", 1.2)
        stop_mult = params.get("stop_atr_multiple", 1.5)
        target_price = entry_price + atr * target_mult
        stop_price = min(float(pullback_bar["low"]) - 1.0, entry_price - atr * stop_mult)

        if stop_price >= entry_price:
            return None

        # --- Confidence ---
        confidence = 0.55

        # 出来高スパイクが大きいほど信頼度UP
        if volume_spike > 2.5:
            confidence += 0.10
        elif volume_spike > 2.0:
            confidence += 0.05

        # 押し目が浅いほど信頼度UP
        if bo_body > 0:
            depth_ratio = pullback_depth / bo_body
            if depth_ratio < 0.25:
                confidence += 0.10  # 浅い押し目は強い
            elif depth_ratio < 0.38:
                confidence += 0.05

        # 出来高の減り方が大きいほど信頼度UP (セリングプレッシャー減退)
        if bo_volume > 0:
            vol_decline = 1.0 - (pb_volume / bo_volume)
            if vol_decline > 0.5:
                confidence += 0.10  # 出来高半減以上
            elif vol_decline > 0.3:
                confidence += 0.05

        # 低ボラペナルティ
        atr_pct = (atr / entry_price * 100) if entry_price > 0 else 0
        if atr_pct < 1.5:
            confidence -= 0.10

        confidence = max(0.0, min(1.0, confidence))

        reason = (
            f"出来高枯れ押し目: "
            f"BO陽線 vol={volume_spike:.1f}x body={body_pct:.0%}, "
            f"PB陰線 vol={pb_volume / bo_volume:.0%} depth={pullback_depth / bo_body * 100 if bo_body > 0 else 0:.0f}%"
        )

        logger.info(
            f"[volume_dryup] {ticker}: SIGNAL conf={confidence:.2f} "
            f"entry={entry_price:.0f} stop={stop_price:.0f} target={target_price:.0f} "
            f"{reason}"
        )

        return TradeSignal(
            ticker=ticker,
            direction="long",
            confidence=confidence,
            entry_price=round(entry_price, 1),
            stop_loss=round(stop_price, 1),
            take_profit=round(target_price, 1),
            strategy_name=self.name,
            entry_reason=reason,
            features_at_entry=features,
        )

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        current_price = features.get("current_price", 0)
        if current_price <= 0:
            return False, ""
        if current_price <= position.stop_loss:
            return True, f"損切り: ¥{current_price:.0f} <= SL ¥{position.stop_loss:.0f}"
        if current_price >= position.take_profit:
            return True, f"利確: ¥{current_price:.0f} >= TP ¥{position.take_profit:.0f}"
        return False, ""
