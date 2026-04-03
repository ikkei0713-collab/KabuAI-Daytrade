"""
IDTW (Indexing Dynamic Time Warping) パターンマッチング予測

論文: 中川慧 (2019) 「価格変動パターンを用いた株価予測手法の実証研究」筑波大学博士論文

手法:
1. IDTW: 株価を前期末基準で指数化 → DTWで形状の類似度を計算
2. k*-NN: 類似パターン上位k個の翌期リターンから予測方向を判定
3. confidence boost: 類似パターンの一致度をエントリー判断に活用

デイトレード適用:
- 日足の直近N日パターンを過去と照合
- 分足の直近M本パターンも照合可能
- 類似パターン後の翌日/翌数時間のリターン分布からconfidenceを算出
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Core DTW / IDTW algorithms
# ---------------------------------------------------------------------------

def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """標準DTW距離を計算する。O(N*M)。"""
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    return float(D[n, m])


def _indexify(series: np.ndarray) -> np.ndarray:
    """時系列を始点基準で指数化する (IDTW前処理)。

    I[0] = 1, I[i] = I[i-1] * (series[i] / series[i-1])
    → 始点を1に正規化した累積リターン系列。
    """
    if len(series) < 2 or series[0] == 0:
        return series.copy()
    idx = np.ones(len(series))
    for i in range(1, len(series)):
        if series[i - 1] != 0:
            idx[i] = idx[i - 1] * (series[i] / series[i - 1])
        else:
            idx[i] = idx[i - 1]
    return idx


def idtw_distance(x: np.ndarray, y: np.ndarray) -> float:
    """IDTW距離: 指数化してからDTWを適用。

    価格水準が異なる時系列同士でも「形状」の類似度を測定できる。
    """
    ix = _indexify(x)
    iy = _indexify(y)
    return _dtw_distance(ix, iy)


# ---------------------------------------------------------------------------
# Pattern Matcher
# ---------------------------------------------------------------------------

class PatternMatcher:
    """IDTWベースの価格パターンマッチング予測器。

    Usage:
        matcher = PatternMatcher(window=20, top_k=5)
        result = matcher.predict(daily_df)
        # result.confidence_boost, result.predicted_direction, result.similar_patterns
    """

    def __init__(
        self,
        window: int = 20,
        top_k: int = 5,
        min_history: int = 60,
        lookahead: int = 1,
    ) -> None:
        """
        Args:
            window: パターンマッチングのウィンドウ幅（日数/本数）
            top_k: 類似パターン上位k個を使用
            min_history: 最低限必要な履歴長
            lookahead: 予測先読み期間（日数/本数）
        """
        self.window = window
        self.top_k = top_k
        self.min_history = min_history
        self.lookahead = lookahead

    def predict(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
    ) -> PatternMatchResult:
        """日足DataFrameから直近パターンを過去と照合し、予測を返す。

        Args:
            df: OHLCV DataFrame (時系列順)
            price_col: 終値カラム名

        Returns:
            PatternMatchResult with confidence_boost and direction
        """
        if df is None or len(df) < self.min_history + self.window:
            return PatternMatchResult()

        prices = df[price_col].dropna().values.astype(float)
        if len(prices) < self.min_history + self.window:
            return PatternMatchResult()

        # 直近ウィンドウ = クエリパターン
        query = prices[-self.window:]

        # 過去のウィンドウを全てスライドして候補を構築
        # 各候補のlookahead先のリターンも記録
        candidates = []
        end_idx = len(prices) - self.window - self.lookahead
        for i in range(0, end_idx):
            candidate = prices[i : i + self.window]
            future_price = prices[i + self.window + self.lookahead - 1]
            current_price = prices[i + self.window - 1]
            if current_price > 0:
                future_return = (future_price - current_price) / current_price
                dist = idtw_distance(query, candidate)
                candidates.append((dist, future_return, i))

        if not candidates:
            return PatternMatchResult()

        # 距離順にソートして上位k個を取得
        candidates.sort(key=lambda x: x[0])
        top = candidates[: self.top_k]

        distances = [c[0] for c in top]
        returns = [c[1] for c in top]
        indices = [c[2] for c in top]

        # 距離の逆数で重み付き平均リターンを算出
        weights = []
        for d in distances:
            w = 1.0 / (d + 1e-8)
            weights.append(w)
        total_w = sum(weights)
        if total_w == 0:
            return PatternMatchResult()

        weighted_return = sum(r * w for r, w in zip(returns, weights)) / total_w
        up_count = sum(1 for r in returns if r > 0)
        win_rate = up_count / len(returns)

        # confidence boost: 類似度が高く方向が一致するほど大きい
        avg_dist = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_dist)  # 0~1に正規化

        # 方向の合意度 (全員が同じ方向なら1.0)
        consensus = abs(2 * win_rate - 1.0)  # 0~1

        # confidence_boost = similarity * consensus * 基本スケール
        confidence_boost = similarity * consensus * 0.15  # 最大+0.15

        # 予測方向
        if weighted_return > 0:
            direction = "long"
        elif weighted_return < 0:
            direction = "short"
        else:
            direction = "neutral"

        result = PatternMatchResult(
            confidence_boost=confidence_boost,
            predicted_direction=direction,
            weighted_return=weighted_return,
            win_rate=win_rate,
            similarity=similarity,
            consensus=consensus,
            avg_distance=avg_dist,
            top_k_returns=returns,
            top_k_distances=distances,
            top_k_indices=indices,
        )

        logger.debug(
            "[IDTW] dir={} wr={:.0%} sim={:.3f} cons={:.2f} boost={:+.3f} wret={:+.4f}",
            direction, win_rate, similarity, consensus,
            confidence_boost, weighted_return,
        )

        return result

    def predict_intraday(
        self,
        intraday_df: pd.DataFrame,
        window: int = 30,
        lookahead: int = 10,
        price_col: str = "close",
    ) -> PatternMatchResult:
        """分足DataFrameからパターンマッチング予測。

        Args:
            intraday_df: 分足OHLCV DataFrame
            window: パターンウィンドウ（本数）
            lookahead: 先読み本数
            price_col: 終値カラム名
        """
        if intraday_df is None or len(intraday_df) < window + lookahead + 30:
            return PatternMatchResult()

        prices = intraday_df[price_col].dropna().values.astype(float)
        if len(prices) < window + lookahead + 30:
            return PatternMatchResult()

        query = prices[-window:]

        candidates = []
        end_idx = len(prices) - window - lookahead
        for i in range(0, end_idx):
            candidate = prices[i : i + window]
            future_price = prices[i + window + lookahead - 1]
            current_price = prices[i + window - 1]
            if current_price > 0:
                future_return = (future_price - current_price) / current_price
                dist = idtw_distance(query, candidate)
                candidates.append((dist, future_return, i))

        if not candidates:
            return PatternMatchResult()

        candidates.sort(key=lambda x: x[0])
        top = candidates[: self.top_k]

        distances = [c[0] for c in top]
        returns = [c[1] for c in top]

        weights = [1.0 / (d + 1e-8) for d in distances]
        total_w = sum(weights)
        if total_w == 0:
            return PatternMatchResult()

        weighted_return = sum(r * w for r, w in zip(returns, weights)) / total_w
        up_count = sum(1 for r in returns if r > 0)
        win_rate = up_count / len(returns)

        avg_dist = np.mean(distances)
        similarity = 1.0 / (1.0 + avg_dist)
        consensus = abs(2 * win_rate - 1.0)
        confidence_boost = similarity * consensus * 0.10  # 分足は控えめ

        direction = "long" if weighted_return > 0 else ("short" if weighted_return < 0 else "neutral")

        return PatternMatchResult(
            confidence_boost=confidence_boost,
            predicted_direction=direction,
            weighted_return=weighted_return,
            win_rate=win_rate,
            similarity=similarity,
            consensus=consensus,
            avg_distance=avg_dist,
            top_k_returns=returns,
            top_k_distances=distances,
        )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class PatternMatchResult:
    """パターンマッチング予測結果。"""

    def __init__(
        self,
        confidence_boost: float = 0.0,
        predicted_direction: str = "neutral",
        weighted_return: float = 0.0,
        win_rate: float = 0.5,
        similarity: float = 0.0,
        consensus: float = 0.0,
        avg_distance: float = float("inf"),
        top_k_returns: Optional[list[float]] = None,
        top_k_distances: Optional[list[float]] = None,
        top_k_indices: Optional[list[int]] = None,
    ) -> None:
        self.confidence_boost = confidence_boost
        self.predicted_direction = predicted_direction
        self.weighted_return = weighted_return
        self.win_rate = win_rate
        self.similarity = similarity
        self.consensus = consensus
        self.avg_distance = avg_distance
        self.top_k_returns = top_k_returns or []
        self.top_k_distances = top_k_distances or []
        self.top_k_indices = top_k_indices or []

    @property
    def is_valid(self) -> bool:
        return self.confidence_boost > 0 and self.predicted_direction != "neutral"

    def __repr__(self) -> str:
        return (
            f"PatternMatch(dir={self.predicted_direction}, "
            f"boost={self.confidence_boost:+.3f}, "
            f"wr={self.win_rate:.0%}, sim={self.similarity:.3f})"
        )
