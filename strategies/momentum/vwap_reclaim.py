"""VWAP Reclaim strategy.

Fires when a stock drops below VWAP, stays below for at least 15 minutes,
then reclaims VWAP with strong volume.  Entry is on the close above VWAP;
stop is below the recent low; target is the day's high or 1.2x ATR.

NOTE: 擬似特徴量の限界
- time_below_vwap: 日足ベースでは正確な intraday 滞在時間を計測できない (固定値30)
- volume_at_reclaim: 日足出来高 * 1.2 で推定しており、reclaim 時点の出来高ではない
- proxy_usage_rate ≈ 1.0 → 評価信頼度は限定的
- 本戦略が主力なのは他戦略がより proxy 依存が高いため (相対的に最良)
"""

from typing import Any, Optional

import pandas as pd
from loguru import logger

from core.config import settings
from core.models import StrategyConfig, TradeSignal
from strategies.base import BaseStrategy
from strategies.momentum.trend_follow import TrendFollowStrategy


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
                # 大規模BT (2026-03-27, 168パターン, OOS PF=1.42 trend_up/range)
                "min_time_below_vwap_min": 30,
                "min_volume_at_reclaim": 1.8,
                "target_atr_multiple": 2.0,
                "reclaim_buffer_pct": 0.30,
                "max_distance_from_vwap_pct": 1.0,
                # レジームフィルタ: trend_down/volatile で損失が大きい
                "blocked_regimes": ["trend_down"],
                # 後場 PM-VWAP reclaim（intraday 品質が十分なときのみ）
                "pm_reclaim_min_hold_count": 2,
                "pm_rel_volume_threshold": 1.8,
                "pm_turnover_threshold": 1_000_000_000.0,
                "pm_confidence_boost": 0.08,
                "pm_event_boost": 0.05,
                "pm_intraday_quality_min": 0.60,
                "pm_selector_score_min": 0.40,
            },
        )

    async def scan(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        if not self._validate_data(data, min_rows=4):
            return None
        if not self._validate_features(features, self.REQUIRED_FEATURES):
            return None

        classic = await self._scan_classic_vwap_reclaim(ticker, data, features)
        if classic is not None:
            return classic
        return await self._scan_pm_vwap_reclaim(ticker, data, features)

    async def _scan_classic_vwap_reclaim(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        params = self.config.parameter_set

        if not self._check_regime_filter(features):
            return None

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
        # Event weighting: LLM分析済みの magnitude/direction で加減点
        event_type = features.get("event_type", "")
        if event_type and event_type not in ("", 0, 0.0):
            event_dir = features.get("event_direction", "neutral")
            event_mag = features.get("event_magnitude", 0.5)
            if event_dir == "positive":
                confidence += 0.10 + event_mag * 0.10  # +0.10~+0.20
            elif event_dir == "negative":
                confidence -= 0.05 + event_mag * 0.10  # -0.05~-0.15
            else:
                confidence += 0.05  # neutral event: 軽い加点

        # Regime alignment — 攻撃的チューニング (2026-03-25)
        # 日足ベースのレジーム判定は限界があるため、軽い調整に留める
        # 全レジームでエントリー可能にし、得意レジームでブースト
        regime_result = features.get("regime_result")
        if regime_result is not None:
            regime = regime_result.regime
            if regime in ("volatile", "low_vol"):
                confidence += 0.10  # 得意レジーム: 積極エントリー
            elif regime == "trend_up":
                confidence += 0.05  # 上昇トレンド: 加点
            elif regime == "range":
                confidence -= 0.03  # range: 微減点のみ
            elif regime == "trend_down":
                confidence -= 0.05  # trend_down: 軽い減点のみ

        # Trend follow filter — 攻撃的: ボーナスのみ、減点なし
        is_trending = features.get("_is_trending", False)
        trend_dir = features.get("_trend_direction", "none")
        if is_trending and trend_dir == "up":
            trend_str = features.get("trend_strength", 0)
            vol_trend = features.get("volume_trend", 1.0)
            if trend_str > 0.45 and vol_trend > 1.2:
                confidence += 0.10  # trend filter 完全通過
            else:
                confidence += 0.05  # 部分通過
        # trend 不通過でも減点しない（機会を逃さない）

        # Convergence filter — 攻撃的: ボーナスのみ
        conv_passed, conv_adj, conv_reason = TrendFollowStrategy.convergence_filter(features)
        if not conv_passed:
            pass  # 不通過でも減点しない
            logger.debug(f"[vwap_reclaim] {ticker}: convergence filter blocked: {conv_reason}")
        else:
            confidence += conv_adj
            # squeeze_breakout_ready なら最優先シナリオ
            if features.get("squeeze_breakout_ready", False):
                confidence += 0.05
                logger.debug(f"[vwap_reclaim] {ticker}: squeeze_breakout_ready → +0.05")

        # selector_score: ボーナスのみ (減点なし)
        selector_score = features.get("selector_score", 0)
        if selector_score > 0.5:
            confidence += 0.05

        # Sector bias: ボーナスのみ
        sector_bias_score = features.get("_sector_bias_score", 0.0)
        if sector_bias_score > 0:
            confidence += sector_bias_score

        # Proxy penalty: 無効化 (日足ベースの限界を受容)

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

    @staticmethod
    def _position_entry_reason(position: Any) -> str:
        if isinstance(position, dict):
            sig = position.get("signal")
            if sig is not None and hasattr(sig, "entry_reason"):
                return str(sig.entry_reason or "")
        return ""

    async def _scan_pm_vwap_reclaim(
        self, ticker: str, data: pd.DataFrame, features: dict
    ) -> Optional[TradeSignal]:
        """後場 12:30 以降の PM-VWAP を再計算し、13:00以降の再定着初動のみ。"""
        params = self.config.parameter_set
        if not features.get("pm_session_active"):
            return None
        if features.get("pm_intraday_is_proxy"):
            logger.debug(f"[vwap_reclaim] {ticker}: PM skip — intraday proxy")
            return None
        qmin = params.get("pm_intraday_quality_min", settings.PM_INTRADAY_QUALITY_MIN)
        if float(features.get("pm_intraday_quality_score") or 0) < qmin:
            logger.debug(f"[vwap_reclaim] {ticker}: PM skip — low intraday quality")
            return None
        if features.get("pm_force_exit_near"):
            return None
        if float(features.get("pm_minutes_from_open") or 0) < 30.0:
            return None
        if not features.get("pm_vwap_reclaim_flag"):
            return None
        min_hold = int(params.get("pm_reclaim_min_hold_count", settings.PM_RECLAIM_MIN_HOLD_COUNT))
        if int(features.get("pm_vwap_hold_count") or 0) < min_hold:
            logger.debug(f"[vwap_reclaim] {ticker}: PM hold_count insufficient")
            return None
        rv_th = float(params.get("pm_rel_volume_threshold", settings.PM_RELATIVE_VOLUME_THRESHOLD))
        if float(features.get("pm_relative_volume") or 0) < rv_th:
            return None
        to_th = float(params.get("pm_turnover_threshold", settings.PM_TURNOVER_THRESHOLD))
        if float(features.get("pm_turnover") or 0) < to_th:
            return None
        if not features.get("pm_spread_ok", True):
            return None

        sel_min = float(params.get("pm_selector_score_min", 0.40))
        if float(features.get("selector_score") or 0) < sel_min:
            return None

        ok, why = TrendFollowStrategy.pm_trend_filter(features)
        if not ok:
            logger.debug(f"[vwap_reclaim] {ticker}: PM trend filter: {why}")
            return None

        latest = data.iloc[-1]
        current_price = float(latest["close"])
        atr: float = features["atr"]
        vwap: float = features["vwap"]
        pm_vwap = float(features.get("pm_vwap") or 0)

        lookback = min(len(data), 6)
        recent_low = float(data.iloc[-lookback:]["low"].min())
        day_high = float(data["high"].max())
        entry_price = current_price
        stop_price = min(recent_low - 1.0, pm_vwap * 0.995) if pm_vwap > 0 else recent_low - 1.0
        atr_target = entry_price + atr * params.get("target_atr_multiple", 1.5)
        target_price = max(day_high, atr_target)

        if stop_price >= entry_price:
            return None

        confidence = 0.52
        confidence += float(params.get("pm_confidence_boost", settings.PM_CONFIDENCE_BOOST))
        event_type = features.get("event_type", "")
        if event_type and event_type not in ("", 0, 0.0):
            confidence += float(params.get("pm_event_boost", settings.PM_EVENT_BOOST))
        if float(features.get("selector_score") or 0) > 0.55:
            confidence += 0.04
        confidence += float(features.get("pm_low_price_bonus") or 0.0)
        confidence = min(confidence, 0.90)

        snap = {**features, "_pm_vwap_reclaim": True}
        signal = TradeSignal(
            ticker=ticker,
            direction="long",
            strategy_name=self.name,
            entry_price=round(entry_price, 1),
            stop_loss=round(stop_price, 1),
            take_profit=round(target_price, 1),
            confidence=round(confidence, 2),
            entry_reason=(
                f"PM-VWAP奪回: 後場VWAP再定着 relVol={features.get('pm_relative_volume'):.2f} "
                f"turnover={features.get('pm_turnover'):.0f} hold={features.get('pm_vwap_hold_count')} "
                f"dayVWAP={vwap:.0f} pmVWAP={pm_vwap:.0f}"
            ),
            features_snapshot=snap,
        )
        logger.info(
            f"[vwap_reclaim] PM SIGNAL {ticker} long entry={entry_price:.0f} "
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

        if (
            features.get("pm_force_exit_near")
            and "PM-VWAP" in self._position_entry_reason(position)
        ):
            return True, "PM引け前 — 強制決済ゾーン"

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
