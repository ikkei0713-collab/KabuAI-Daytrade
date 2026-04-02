"""Intraday Trend Follow strategy.

Identifies an intraday trend using EMA(9) > EMA(21) > VWAP (for longs).
Enters on a pullback to EMA(9) in the trend direction.  Stop is placed
below EMA(21); target uses a trailing stop at 1.5 ATR.
"""

from typing import Optional

import pandas as pd
from loguru import logger

from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy


class TrendFollowStrategy(BaseStrategy):
    """日中トレンドフォロー – ride the intraday trend."""

    REQUIRED_FEATURES = [
        "ema_9",
        "ema_21",
        "vwap",
        "trend_strength",
        "volume_trend",
        "atr",
    ]

    def __init__(self, config: Optional[StrategyConfig] = None):
        super().__init__(config or self.get_default_config())

    @staticmethod
    def is_trending(features: dict) -> tuple[bool, str, float]:
        """Check if market is trending. Used as a filter by other strategies.

        Returns:
            (is_trending, direction, strength) where direction is 'up'/'down'/'none'
            and strength is 0-1.
        """
        ema_9 = features.get("ema_9", 0)
        ema_21 = features.get("ema_21", 0)
        vwap = features.get("vwap", 0)
        trend_str = features.get("trend_strength", 0)

        if not all([ema_9, ema_21, vwap]):
            return False, "none", 0.0

        buf = ema_9 * 0.0005  # 0.05% buffer

        if ema_9 > ema_21 + buf and ema_21 > vwap + buf:
            return True, "up", min(abs(trend_str), 1.0)
        elif ema_9 < ema_21 - buf and ema_21 < vwap - buf:
            return True, "down", min(abs(trend_str), 1.0)

        return False, "none", 0.0

    @staticmethod
    def pm_trend_filter(features: dict) -> tuple[bool, str]:
        """後場 PM-VWAP reclaim 用フィルタ（単独エントリーには使わない）。

        EMA9>EMA21, 終値>日次VWAP, trend_strength / volume_trend 閾値、
        価格>PM-VWAP、PM-VWAP 下落傾きすぎを拒否、地合い極端悪化を軽く拒否。
        """
        from core.config import settings

        ema_9 = float(features.get("ema_9") or 0)
        ema_21 = float(features.get("ema_21") or 0)
        vwap = float(features.get("vwap") or 0)
        close = float(
            features.get("current_price")
            or features.get("close")
            or features.get("last_close")
            or 0
        )
        trend_str = float(features.get("trend_strength") or 0)
        vol_trend = float(features.get("volume_trend") or 1.0)
        pm_vwap = float(features.get("pm_vwap") or 0)

        buf = max(ema_9 * 0.0005, 0.01)
        if ema_9 <= ema_21 + buf:
            return False, "ema9_not_above_ema21"
        if vwap > 0 and close <= vwap + buf * 0.1:
            return False, "close_not_above_vwap"
        if abs(trend_str) <= 0.45:
            return False, "trend_strength_low"
        if vol_trend <= 1.2:
            return False, "volume_trend_low"
        if pm_vwap > 0 and close <= pm_vwap + buf * 0.1:
            return False, "close_not_above_pm_vwap"
        slope = float(features.get("pm_vwap_slope") or 0)
        if slope < -settings.PM_VWAP_SLOPE_MAX_NEG:
            return False, "pm_vwap_slope_down"
        regime_result = features.get("regime_result")
        if regime_result is not None:
            if (
                getattr(regime_result, "regime", "") == "trend_down"
                and float(getattr(regime_result, "confidence", 0) or 0) > 0.55
            ):
                return False, "regime_trend_down"
        return True, "ok"

    @staticmethod
    def convergence_filter(features: dict) -> tuple[bool, float, str]:
        """収束フィルタ: 拡散飛び乗りを拒否し、収束後のみ許可する.

        レジーム情報がある場合は、レジーム別閾値を適用する。
        論文 (Arai 2013): トレンド相場とレンジ相場で有効なインジケータが異なる。
        → トレンド時は MA 拡散を許容、レンジ時は収束を厳格に要求。

        Returns:
            (passed, adjustment, reason)
            - passed: フィルタ通過したか
            - adjustment: confidence への加減算値
            - reason: 判定理由
        """
        from core.config import settings

        ma_spread_pct = features.get("ma_spread_pct")
        ma_conv_score = features.get("ma_convergence_score")
        ma_conv_trend = features.get("ma_convergence_trend", 0)
        range_comp = features.get("range_compression_score")
        vol_comp = features.get("volatility_compression_score")
        ext_score = features.get("extension_from_ma_score")
        post_cross_exp = features.get("post_cross_expansion_flag", False)
        post_cross_con = features.get("post_cross_consolidation_flag", False)

        # データ不足の場合はフィルタ通過 (制約しない)
        if ma_spread_pct is None or ma_conv_score is None:
            return True, 0.0, "convergence_data_insufficient"

        # レジーム別閾値の取得 (Arai 2013 の知見を反映)
        regime_result = features.get("regime_result")
        if regime_result is not None:
            from tools.market_regime import RegimeDetector
            rp = RegimeDetector.get_convergence_params(regime_result.regime)
            max_spread = rp["max_ma_spread_pct"]
            min_conv = rp["min_convergence_score"]
            min_range = rp["min_range_compression"]
            min_vol = rp["min_vol_compression"]
            conv_boost = rp["convergence_boost"]
            exp_penalty = rp["expansion_penalty"]
        else:
            # デフォルト (config 値)
            max_spread = settings.MAX_MA_SPREAD_PCT_FOR_ENTRY
            min_conv = settings.MIN_MA_CONVERGENCE_SCORE
            min_range = settings.MIN_RANGE_COMPRESSION_SCORE
            min_vol = settings.MIN_VOLATILITY_COMPRESSION_SCORE
            conv_boost = settings.CONVERGENCE_CONFIDENCE_BOOST
            exp_penalty = settings.EXPANSION_PENALTY_AFTER_CROSS

        adjustment = 0.0
        reasons = []

        # 1. MA 拡散しすぎ → 拒否
        if ma_spread_pct > max_spread:
            return False, -0.10, f"ma_spread_too_wide({ma_spread_pct:.4f}>{max_spread:.4f})"

        # 2. MA 乖離が大きすぎる → 拒否
        # BT結果: 勝ち=0.69 vs 負け=0.11 → 0.30→0.35 に引き上げ
        if ext_score is not None and ext_score < 0.35:
            return False, -0.10, f"extension_too_far({ext_score:.2f})"

        # 3. GC/DC 直後の拡散 → 拒否
        if post_cross_exp:
            return False, -exp_penalty, "post_cross_expansion"

        # 4. GC/DC 後の収束 → 加点
        if post_cross_con:
            adjustment += conv_boost * 0.5
            reasons.append("post_cross_consolidation")

        # 5. 収束スコアが高い → 加点
        if ma_conv_score >= min_conv:
            adjustment += conv_boost * 0.5
            reasons.append(f"convergence_high({ma_conv_score:.2f})")

        # 6. 収束トレンドが改善中 → 加点
        if ma_conv_trend > 0.3:
            adjustment += 0.03
            reasons.append("convergence_improving")

        # 7. レンジ圧縮・ボラ縮小 → 加点
        if range_comp is not None and range_comp >= min_range:
            adjustment += 0.02
            reasons.append("range_compressed")
        if vol_comp is not None and vol_comp >= min_vol:
            adjustment += 0.02
            reasons.append("vol_compressed")

        reason_str = ",".join(reasons) if reasons else "convergence_neutral"
        return True, round(adjustment, 3), reason_str

    def get_default_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_name="trend_follow",
            is_active=True,
            feature_requirements=self.REQUIRED_FEATURES,
            expected_market_condition="bull",
            parameter_set={
                # 大規模BT (2026-03-27, 168パターン, OOS PF=2.63, 27件)
                "min_trend_strength": 0.1,
                "min_volume_trend": 1.2,
                "trailing_atr_multiple": 1.0,
                "pullback_ema": "ema_9",
                "stop_ema": "ema_21",
                "ema_alignment_buffer_pct": 0.05,
                # レジームフィルタ: trend_down/volatile で損失
                "blocked_regimes": ["trend_down"],
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=5):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        params = self.config.parameter_set

        if not self._check_regime_filter(features):
            return None

        ema_9: float = features["ema_9"]
        ema_21: float = features["ema_21"]
        vwap: float = features["vwap"]
        trend_str: float = features["trend_strength"]
        vol_trend: float = features["volume_trend"]
        atr: float = features["atr"]

        latest = data.iloc[-1]
        current_price = float(latest["close"])
        current_low = float(latest["low"])
        buf = ema_9 * params.get("ema_alignment_buffer_pct", 0.05) / 100

        # --- Determine direction from EMA alignment ---
        long_aligned = ema_9 > ema_21 + buf and ema_21 > vwap + buf
        short_aligned = ema_9 < ema_21 - buf and ema_21 < vwap - buf

        if not long_aligned and not short_aligned:
            logger.debug(
                f"[trend_follow] {ticker}: EMAs not aligned "
                f"(EMA9={ema_9:.0f}, EMA21={ema_21:.0f}, VWAP={vwap:.0f})"
            )
            return None

        # Trend strength filter
        if abs(trend_str) < params.get("min_trend_strength", 0.4):
            logger.debug(f"[trend_follow] {ticker}: trend_strength {trend_str:.2f} too weak")
            return None

        # Volume trend: rising volume in trend direction
        if vol_trend < params.get("min_volume_trend", 1.0):
            logger.debug(f"[trend_follow] {ticker}: volume_trend {vol_trend:.2f} declining")
            return None

        # --- Pullback detection ---
        if long_aligned:
            # Price should be near EMA9 (pulled back to it)
            pullback_dist = (current_price - ema_9) / atr if atr > 0 else 999
            if pullback_dist > 0.5 or pullback_dist < -0.3:
                logger.debug(
                    f"[trend_follow] {ticker}: long pullback distance "
                    f"{pullback_dist:.2f} ATR not ideal"
                )
                return None

            # Candle must show bounce (close above open)
            if current_price < float(latest["open"]):
                return None

            entry_price = current_price
            stop_price = ema_21 - atr * 0.2
            target_price = entry_price + atr * params.get("trailing_atr_multiple", 1.5) * 2
            direction = "long"

        else:  # short_aligned
            pullback_dist = (ema_9 - current_price) / atr if atr > 0 else 999
            if pullback_dist > 0.5 or pullback_dist < -0.3:
                return None

            if current_price > float(latest["open"]):
                return None

            entry_price = current_price
            stop_price = ema_21 + atr * 0.2
            target_price = entry_price - atr * params.get("trailing_atr_multiple", 1.5) * 2
            direction = "short"

        risk = abs(entry_price - stop_price)
        if risk < 1.0:
            return None

        # Confidence
        confidence = 0.50
        if abs(trend_str) > 0.7:
            confidence += 0.10
        if vol_trend > 1.5:
            confidence += 0.10
        ema_spread = abs(ema_9 - ema_21)
        if atr > 0 and ema_spread / atr > 0.3:
            confidence += 0.05
        # Strong body on pullback candle
        body = abs(current_price - float(latest["open"]))
        candle_range = float(latest["high"]) - current_low
        if candle_range > 0 and body / candle_range > 0.6:
            confidence += 0.10

        # レジーム別ハイブリッド指標切替（論文 fit3 ハイブリッド売買）
        # レンジ/volatile相場ではオシレータ系（RSI）で確認を追加
        regime_result = features.get("regime_result")
        current_regime = getattr(regime_result, "regime", "") if regime_result else ""
        rsi_14 = features.get("rsi_14", 50)
        bb_pct = features.get("price_vs_bollinger", 0.5)

        if current_regime in ("range", "volatile", "low_vol"):
            # レンジ相場: オシレータ確認を追加
            if direction == "long" and rsi_14 > 70:
                confidence -= 0.10  # RSI過熱 → 減点
            elif direction == "long" and rsi_14 < 40:
                confidence += 0.05  # RSI余裕あり → 加点
            if direction == "short" and rsi_14 < 30:
                confidence -= 0.10
            elif direction == "short" and rsi_14 > 60:
                confidence += 0.05
            # BB上限/下限での逆張り確認
            if direction == "long" and bb_pct > 0.9:
                confidence -= 0.08  # BB上限付近 → 減点
            if direction == "short" and bb_pct < 0.1:
                confidence -= 0.08
        elif current_regime == "trend_up":
            # トレンド相場: トレンド指標をそのまま信頼
            confidence += 0.05

        confidence = min(confidence, 0.90)

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
                f"トレンドフォロー{'↑' if direction == 'long' else '↓'}: "
                f"EMA9({ema_9:.0f})>EMA21({ema_21:.0f})>VWAP({vwap:.0f}), "
                f"トレンド強度{trend_str:+.2f}, "
                f"出来高トレンド{vol_trend:.1f}x"
            ),
            features_snapshot=features,
        )
        logger.info(f"[trend_follow] SIGNAL {ticker} {direction} entry={entry_price:.0f}")
        return signal

    def should_exit(
        self, position, current_data: pd.DataFrame, features: dict
    ) -> tuple[bool, str]:
        if current_data is None or current_data.empty:
            return False, ""

        latest = current_data.iloc[-1]
        current_price = float(latest["close"])
        ema_21 = features.get("ema_21", 0)
        atr = features.get("atr", 0)
        trail_mult = self.config.parameter_set.get("trailing_atr_multiple", 1.5)

        # Hard stop
        if position.direction == "long" and current_price <= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"
        if position.direction == "short" and current_price >= position.stop_loss:
            return True, f"ストップロス ({current_price:.0f})"

        # EMA21 break (trend structure broken)
        if position.direction == "long" and ema_21 > 0 and current_price < ema_21:
            return True, f"EMA21割れ – トレンド崩壊 ({current_price:.0f})"
        if position.direction == "short" and ema_21 > 0 and current_price > ema_21:
            return True, f"EMA21突破 – トレンド崩壊 ({current_price:.0f})"

        # Trailing stop based on ATR
        if atr > 0:
            if position.direction == "long":
                day_high = float(current_data["high"].max())
                trail = day_high - atr * trail_mult
                if trail > position.entry_price and current_price <= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"
            else:
                day_low = float(current_data["low"].min())
                trail = day_low + atr * trail_mult
                if trail < position.entry_price and current_price >= trail:
                    return True, f"トレーリングストップ ({current_price:.0f})"

        # Target
        if position.direction == "long" and current_price >= position.take_profit:
            return True, f"利確 ({current_price:.0f})"
        if position.direction == "short" and current_price <= position.take_profit:
            return True, f"利確 ({current_price:.0f})"

        return False, ""
