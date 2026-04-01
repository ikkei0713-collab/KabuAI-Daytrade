"""
相場レジーム判定

日足データからその日の相場状態を推定する。
戦略選択・ロット調整・見送り判断に使用。
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal
from loguru import logger


RegimeType = Literal["trend_up", "trend_down", "range", "volatile", "low_vol"]


@dataclass
class RegimeResult:
    """レジーム判定結果"""
    regime: RegimeType
    confidence: float          # 0-1
    metrics: dict              # 判定に使った指標
    strategy_weights: dict     # 戦略別の推奨ウェイト
    position_scale: float      # ロット倍率 (0.0-1.5)


class RegimeDetector:
    """
    日足データから相場レジームを判定する。

    判定ロジック:
    1. トレンド方向: 5日/20日SMA関係 + 5日間の方向性
    2. ボラティリティ: ATR%の過去20日での位置
    3. レンジ判定: ボリンジャーバンド幅の収縮
    4. 出来高特性: 相対出来高
    """

    def detect(self, df: pd.DataFrame) -> RegimeResult:
        """
        日足DataFrameからレジームを判定

        Args:
            df: OHLCV DataFrame (最低20行必要)

        Returns:
            RegimeResult
        """
        if len(df) < 20:
            return RegimeResult(
                regime="range",
                confidence=0.3,
                metrics={},
                strategy_weights=self._default_weights(),
                position_scale=0.5,
            )

        # カラム名の正規化
        close_col = "close" if "close" in df.columns else "Close"
        high_col = "high" if "high" in df.columns else "High"
        low_col = "low" if "low" in df.columns else "Low"
        vol_col = "volume" if "volume" in df.columns else "Volume"

        close = df[close_col].astype(float)
        high = df[high_col].astype(float)
        low = df[low_col].astype(float)
        volume = df[vol_col].astype(float)

        # 指標計算
        sma5 = close.rolling(5).mean()
        sma20 = close.rolling(20).mean()

        # ATR%
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        atr_pct = (atr_14 / close * 100).iloc[-1]

        # ATR%の過去20日でのパーセンタイル
        atr_pct_series = atr_14 / close * 100
        atr_percentile = (atr_pct_series.rank(pct=True)).iloc[-1]

        # トレンド強度: 5日リターンの方向性
        ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0

        # SMA関係
        sma5_last = sma5.iloc[-1]
        sma20_last = sma20.iloc[-1]
        sma_trend = (sma5_last / sma20_last - 1) * 100  # SMA5がSMA20より何%上か

        # ボリンジャーバンド幅（20日）
        bb_std = close.rolling(20).std()
        bb_width = (bb_std.iloc[-1] / close.iloc[-1] * 100) if close.iloc[-1] > 0 else 0
        bb_width_percentile = (bb_std / close * 100).tail(60).rank(pct=True).iloc[-1] if len(close) >= 60 else 0.5

        # 相対出来高
        vol_ratio = volume.iloc[-1] / volume.tail(20).mean() if volume.tail(20).mean() > 0 else 1.0

        # 日足の方向一貫性（5日中何日が陽線か）
        daily_returns = close.diff().tail(5)
        up_days = (daily_returns > 0).sum()
        consistency = up_days / 5  # 1.0=全陽線, 0.0=全陰線

        metrics = {
            "ret_5d": round(ret_5d, 2),
            "sma_trend": round(sma_trend, 2),
            "atr_pct": round(atr_pct, 2),
            "atr_percentile": round(atr_percentile, 2),
            "bb_width": round(bb_width, 2),
            "bb_width_percentile": round(bb_width_percentile, 2),
            "vol_ratio": round(vol_ratio, 2),
            "consistency": round(consistency, 2),
        }

        # レジーム判定
        regime, confidence = self._classify(metrics)
        strategy_weights = self._get_strategy_weights(regime)
        position_scale = self._get_position_scale(regime, confidence)

        logger.debug(
            f"レジーム判定: {regime} (確信度{confidence:.0%}) "
            f"ret5d={ret_5d:+.1f}% atr={atr_pct:.1f}% vol={vol_ratio:.1f}x"
        )

        return RegimeResult(
            regime=regime,
            confidence=confidence,
            metrics=metrics,
            strategy_weights=strategy_weights,
            position_scale=position_scale,
        )

    def _classify(self, m: dict) -> tuple[RegimeType, float]:
        """指標からレジーム分類"""
        ret = m["ret_5d"]
        sma = m["sma_trend"]
        atr_p = m["atr_percentile"]
        bb_p = m["bb_width_percentile"]
        consistency = m["consistency"]
        atr_pct = m.get("atr_pct", 0)

        # 高ボラティリティ判定
        # 低位株（ATR%が高くなりやすい）は閾値を緩和:
        # ATR% 3%以上 かつ ATRパーセンタイル上位10% の場合のみvolatile
        is_high_vol = atr_p > 0.9 or (atr_p > 0.8 and atr_pct > 3.0)
        if is_high_vol:
            if abs(ret) > 3:
                if ret > 0:
                    return "trend_up", min(0.9, atr_p)
                else:
                    return "trend_down", min(0.9, atr_p)
            return "volatile", atr_p

        # 低ボラティリティ（ATRが下位20%）
        if atr_p < 0.2:
            return "low_vol", 1.0 - atr_p

        # トレンド判定
        if ret > 2 and sma > 0.5 and consistency >= 0.6:
            return "trend_up", min(0.85, consistency)
        if ret < -2 and sma < -0.5 and consistency <= 0.4:
            return "trend_down", min(0.85, 1.0 - consistency)

        # レンジ（BB幅が収縮 or 方向性なし）
        if bb_p < 0.3 or (abs(ret) < 1 and abs(sma) < 0.3):
            return "range", 0.6

        # デフォルト
        return "range", 0.4

    def _get_strategy_weights(self, regime: RegimeType) -> dict:
        """レジーム別の戦略推奨ウェイト"""
        weights = {
            "trend_up": {
                "orb": 1.0, "gap_go": 0.9, "open_drive": 0.8,
                "trend_follow": 0.9, "vwap_reclaim": 0.7,
                "vwap_bounce": 0.5, "rsi_reversal": 0.2,
                "overextension": 0.1,
            },
            "trend_down": {
                "orb": 0.8, "gap_fade": 0.7,
                "crash_rebound": 0.6, "overextension": 0.5,
                "vwap_reclaim": 0.3, "trend_follow": 0.3,
            },
            "range": {
                "vwap_bounce": 0.9, "vwap_reclaim": 0.8,
                "rsi_reversal": 0.7, "overextension": 0.6,
                "orb": 0.4, "spread_entry": 0.5,
            },
            "volatile": {
                "orb": 0.8, "crash_rebound": 0.7,
                "gap_go": 0.6, "trend_follow": 0.5,
                "vwap_reclaim": 0.3,
            },
            "low_vol": {
                "spread_entry": 0.5, "vwap_bounce": 0.4,
                "orb": 0.2,
                # ほぼ全戦略が低推奨 → 見送り推奨
            },
        }
        return weights.get(regime, self._default_weights())

    def _default_weights(self) -> dict:
        return {"orb": 0.5, "vwap_reclaim": 0.5, "vwap_bounce": 0.5}

    @staticmethod
    def get_convergence_params(regime: RegimeType) -> dict:
        """レジーム別の収束フィルタ推奨閾値を返す.

        論文 (Arai 2013) の知見: レジームによって有効なインジケータが異なる。
        - トレンド相場: トレンドフォロー系指標 (MA) が有効 → 収束フィルタを緩和
        - レンジ相場: オシレータ系指標 (RSI, BB) が有効 → 収束フィルタを厳格化
        - 高ボラ: 拡散しやすい → 閾値を緩和 (拡散を許容)
        - 低ボラ: 動きが少ない → 圧縮条件を緩和
        """
        params = {
            "trend_up": {
                # トレンド中は MA 拡散が自然 → 閾値を緩める
                "max_ma_spread_pct": 0.035,
                "min_convergence_score": 0.40,
                "min_range_compression": 0.40,
                "min_vol_compression": 0.40,
                "convergence_boost": 0.05,
                "expansion_penalty": 0.08,
            },
            "trend_down": {
                # 下降トレンド → 拡散ロングは危険、収束を厳格に
                "max_ma_spread_pct": 0.015,
                "min_convergence_score": 0.65,
                "min_range_compression": 0.55,
                "min_vol_compression": 0.55,
                "convergence_boost": 0.08,
                "expansion_penalty": 0.15,
            },
            "range": {
                # レンジ → オシレータ的な収束・発散が有効
                "max_ma_spread_pct": 0.020,
                "min_convergence_score": 0.55,
                "min_range_compression": 0.50,
                "min_vol_compression": 0.50,
                "convergence_boost": 0.07,
                "expansion_penalty": 0.12,
            },
            "volatile": {
                # ボラ高い → 拡散は自然、厳しくしすぎない
                "max_ma_spread_pct": 0.040,
                "min_convergence_score": 0.35,
                "min_range_compression": 0.35,
                "min_vol_compression": 0.35,
                "convergence_boost": 0.04,
                "expansion_penalty": 0.06,
            },
            "low_vol": {
                # ボラ低い → 圧縮が常態 → 条件を緩める
                "max_ma_spread_pct": 0.025,
                "min_convergence_score": 0.45,
                "min_range_compression": 0.40,
                "min_vol_compression": 0.40,
                "convergence_boost": 0.05,
                "expansion_penalty": 0.10,
            },
        }
        return params.get(regime, params["range"])

    def _get_position_scale(self, regime: RegimeType, confidence: float) -> float:
        """レジーム別のロット倍率"""
        base = {
            "trend_up": 1.2,
            "trend_down": 0.8,
            "range": 0.8,
            "volatile": 0.6,   # ボラ高い→ロット下げる
            "low_vol": 0.4,    # 値動き少ない→見送り気味
        }
        scale = base.get(regime, 0.8)
        # 確信度が低ければさらに縮小
        if confidence < 0.5:
            scale *= 0.7
        return round(min(scale, 1.5), 2)
