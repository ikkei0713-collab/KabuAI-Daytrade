"""
Stocks in Play 銘柄選定

当日デイトレ候補を「打つべき銘柄」に絞り込む。
無駄打ち削減が最大の目的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class StockScore:
    """銘柄の評価結果"""
    code: str
    name: str = ""
    total_score: float = 0.0
    gap_pct: float = 0.0           # ギャップ率
    relative_volume: float = 0.0    # 相対出来高 (当日/20日平均)
    turnover: float = 0.0          # 売買代金 (円)
    atr_pct: float = 0.0           # ATR% (ボラティリティ)
    range_pct: float = 0.0         # 値幅率
    has_event: bool = False         # ニュース/決算/IR有無
    liquidity_tier: str = "C"       # A/B/C (流動性ランク)
    excluded: bool = False
    exclude_reason: str = ""
    score_breakdown: dict = field(default_factory=dict)


class StockSelector:
    """
    当日売買候補をスコアリングして絞り込む。

    評価軸:
    1. ギャップ率 (gap_pct) — 寄付きのギャップが大きい = 材料あり
    2. 相対出来高 (relative_volume) — 普段より出来高が多い = 注目銘柄
    3. 売買代金 (turnover) — 流動性の直接指標
    4. ATR% (atr_pct) — ボラティリティ = 値幅が取れる余地
    5. イベント有無 (has_event) — TDnet/決算 = カタリスト
    6. 値幅率 (range_pct) — 直近の日中レンジ
    """

    # 除外条件 – 保守的チューニング (2026-03-19)
    MIN_PRICE = 200            # 200円未満は除外
    MAX_PRICE = 50000          # 5万円超は除外
    MIN_TURNOVER = 1_000_000_000   # 5億→10億: 高流動性のみ
    MIN_VOLUME_SHARES = 500_000    # 20万→50万株: 板が薄い銘柄を除外

    # スコアリング重み (convergence は軽く加算, v3.3)
    WEIGHTS = {
        "gap": 0.19,          # ギャップ率 (0.20→0.19)
        "volume": 0.24,       # 相対出来高 (0.25→0.24)
        "turnover": 0.15,     # 売買代金
        "atr": 0.19,          # ボラティリティ (0.20→0.19)
        "event": 0.10,        # イベント有無
        "range": 0.10,        # 値幅率
        "convergence": 0.03,  # MA収束加点 (v3.3, 小さく)
    }

    def score_stock(
        self,
        code: str,
        df: pd.DataFrame,
        name: str = "",
        has_event: bool = False,
    ) -> StockScore:
        """1銘柄をスコアリング"""
        score = StockScore(code=code, name=name, has_event=has_event)

        if len(df) < 20:
            score.excluded = True
            score.exclude_reason = "データ不足（20日未満）"
            return score

        close = df["close"].iloc[-1] if "close" in df.columns else df["Close"].iloc[-1]

        # 除外チェック
        if close < self.MIN_PRICE:
            score.excluded = True
            score.exclude_reason = f"株価{close:.0f}円 < {self.MIN_PRICE}円"
            return score
        if close > self.MAX_PRICE:
            score.excluded = True
            score.exclude_reason = f"株価{close:.0f}円 > {self.MAX_PRICE}円"
            return score

        # 基本指標計算
        o = df["open"].iloc[-1] if "open" in df.columns else df["Open"].iloc[-1]
        h = df["high"].iloc[-1] if "high" in df.columns else df["High"].iloc[-1]
        l = df["low"].iloc[-1] if "low" in df.columns else df["Low"].iloc[-1]
        vol = df["volume"].iloc[-1] if "volume" in df.columns else df["Volume"].iloc[-1]
        prev_close = df["close"].iloc[-2] if "close" in df.columns else df["Close"].iloc[-2]

        vol_col = "volume" if "volume" in df.columns else "Volume"
        high_col = "high" if "high" in df.columns else "High"
        low_col = "low" if "low" in df.columns else "Low"
        close_col = "close" if "close" in df.columns else "Close"

        vol_20avg = df[vol_col].tail(20).mean()

        # ギャップ率
        score.gap_pct = abs(o - prev_close) / prev_close * 100 if prev_close > 0 else 0

        # 相対出来高
        score.relative_volume = vol / vol_20avg if vol_20avg > 0 else 1.0

        # 売買代金
        score.turnover = vol * close

        # ATR%
        tr = pd.concat([
            df[high_col] - df[low_col],
            (df[high_col] - df[close_col].shift(1)).abs(),
            (df[low_col] - df[close_col].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.tail(14).mean()
        score.atr_pct = atr_14 / close * 100 if close > 0 else 0

        # 値幅率
        score.range_pct = (h - l) / close * 100 if close > 0 else 0

        # 流動性ティア
        if score.turnover >= 5_000_000_000:  # 50億円以上
            score.liquidity_tier = "A"
        elif score.turnover >= 1_000_000_000:  # 10億円以上
            score.liquidity_tier = "B"
        else:
            score.liquidity_tier = "C"

        # 除外チェック（流動性）
        if score.turnover < self.MIN_TURNOVER:
            score.excluded = True
            score.exclude_reason = f"売買代金{score.turnover/1e8:.1f}億円 < 5億円"
            return score
        if vol < self.MIN_VOLUME_SHARES:
            score.excluded = True
            score.exclude_reason = f"出来高{vol/1e4:.1f}万株 < 20万株"
            return score

        # スコアリング（各0-1に正規化）
        gap_score = min(score.gap_pct / 5.0, 1.0)  # 5%で満点
        vol_score = min(score.relative_volume / 3.0, 1.0)  # 3倍で満点
        turnover_score = min(score.turnover / 10_000_000_000, 1.0)  # 100億円で満点
        atr_score = min(score.atr_pct / 5.0, 1.0)  # ATR5%で満点
        event_score = 1.0 if has_event else 0.0
        range_score = min(score.range_pct / 5.0, 1.0)  # 5%で満点

        # 収束スコア (v3.3) – 日足から MA 収束度を軽く加点
        convergence_score = self._calc_convergence_score(df, close_col, high_col, low_col)

        score.score_breakdown = {
            "gap": round(gap_score, 3),
            "volume": round(vol_score, 3),
            "turnover": round(turnover_score, 3),
            "atr": round(atr_score, 3),
            "event": round(event_score, 3),
            "range": round(range_score, 3),
            "convergence": round(convergence_score, 3),
        }

        score.total_score = sum(
            score.score_breakdown[k] * self.WEIGHTS[k]
            for k in self.WEIGHTS
        )
        score.total_score = round(score.total_score, 4)

        return score

    @staticmethod
    def _calc_convergence_score(df: pd.DataFrame, close_col: str,
                                 high_col: str, low_col: str) -> float:
        """MA収束度を 0-1 でスコアリング (日足ベース).

        event + MA収束 + 価格圧縮 → 加点
        拡散しすぎ → 減点
        """
        close = df[close_col]
        if len(close) < 20:
            return 0.5  # データ不足 → 中立

        ma5 = close.rolling(5, min_periods=5).mean().iloc[-1]
        ma10 = close.rolling(10, min_periods=10).mean().iloc[-1]
        ma20 = close.rolling(20, min_periods=20).mean().iloc[-1]
        last_close = close.iloc[-1]

        if any(pd.isna(x) for x in [ma5, ma10, ma20]) or last_close <= 0:
            return 0.5

        # MA spread
        spread = (max(ma5, ma10, ma20) - min(ma5, ma10, ma20)) / last_close
        # 1% 以内 → 1.0, 4% 以上 → 0.0
        conv_score = max(0.0, min(1.0, 1.0 - spread / 0.04))

        # レンジ圧縮
        if len(df) >= 10:
            recent_ranges = ((df[high_col].iloc[-5:] - df[low_col].iloc[-5:]) / close.iloc[-5:]).mean()
            prior_ranges = ((df[high_col].iloc[-10:-5] - df[low_col].iloc[-10:-5]) / close.iloc[-10:-5]).mean()
            if prior_ranges > 1e-10:
                range_ratio = float(recent_ranges / prior_ranges)
                range_score = max(0.0, min(1.0, 1.0 - (range_ratio - 0.5)))
            else:
                range_score = 0.5
        else:
            range_score = 0.5

        return (conv_score * 0.6 + range_score * 0.4)

    def select_top(
        self,
        scores: list[StockScore],
        max_stocks: int = 8,       # 10→8: 上位8銘柄に集中
        min_score: float = 0.30,   # 0.15→0.30: スコア下位を切る
    ) -> list[StockScore]:
        """スコア上位の銘柄を返す"""
        eligible = [s for s in scores if not s.excluded and s.total_score >= min_score]
        eligible.sort(key=lambda s: s.total_score, reverse=True)
        selected = eligible[:max_stocks]

        if selected:
            logger.info(
                f"銘柄選定: {len(scores)}銘柄中 {len(eligible)}銘柄が条件クリア, "
                f"上位{len(selected)}銘柄を選出"
            )
            for s in selected[:5]:
                logger.info(
                    f"  {s.code} {s.name}: score={s.total_score:.3f} "
                    f"gap={s.gap_pct:.1f}% vol={s.relative_volume:.1f}x "
                    f"turnover={s.turnover/1e8:.0f}億 atr={s.atr_pct:.1f}%"
                    f"{' [EVENT]' if s.has_event else ''}"
                )
        else:
            logger.warning("銘柄選定: 条件を満たす銘柄なし")

        return selected
