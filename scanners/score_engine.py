"""
Strategy scoring and selection engine.

Evaluates and ranks trading strategies for a given ticker/condition pair,
using historical performance, feature alignment, market condition match,
and recency-weighted win rates.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional, Protocol

from loguru import logger

from core.config import settings
from core.models import (
    MarketCondition,
    StrategyConfig,
    StrategyPerformance,
    TradeResult,
)


# ---------------------------------------------------------------------------
# Weighting factors
# ---------------------------------------------------------------------------

HISTORICAL_WIN_RATE_WEIGHT = 0.30
CONDITION_MATCH_WEIGHT = 0.30
FEATURE_ALIGNMENT_WEIGHT = 0.25
RECENCY_WEIGHT = 0.15

# Minimum trades required before trusting historical performance
MIN_TRADES_FOR_STATS = 5

# Recency decay: trades older than this many days get discounted
RECENCY_HALF_LIFE_DAYS = 14


class StrategyProtocol(Protocol):
    """Minimal interface expected from a strategy object."""

    @property
    def name(self) -> str: ...

    @property
    def config(self) -> StrategyConfig: ...


class ScoreEngine:
    """
    Scores and selects the best strategies for a given ticker and market state.

    The score is a weighted combination of:
    - historical_win_rate (0.30): Past win rate for this strategy
    - condition_match (0.30): How well current market conditions match the
      strategy's expected conditions
    - feature_alignment (0.25): How well the ticker's current features
      align with the strategy's requirements
    - recency (0.15): Recent performance weighted more heavily

    Usage::

        engine = ScoreEngine(strategy_performances)
        score = engine.score_strategy(strategy, ticker, features, market_condition)
        best = engine.select_best_strategies(ticker, features, market_condition, strategies)
    """

    def __init__(
        self,
        performances: Optional[dict[str, StrategyPerformance]] = None,
        trade_history: Optional[dict[str, list[TradeResult]]] = None,
    ) -> None:
        self._performances: dict[str, StrategyPerformance] = performances or {}
        self._trade_history: dict[str, list[TradeResult]] = trade_history or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_performance(
        self, strategy_name: str, performance: StrategyPerformance,
    ) -> None:
        """Update cached performance data for a strategy."""
        self._performances[strategy_name] = performance

    def update_trade_history(
        self, strategy_name: str, trades: list[TradeResult],
    ) -> None:
        """Update cached trade history for a strategy."""
        self._trade_history[strategy_name] = trades

    def score_strategy(
        self,
        strategy: StrategyProtocol,
        ticker: str,
        features: dict[str, float],
        market_condition: MarketCondition,
    ) -> float:
        """
        Score a single strategy for the given ticker and conditions.

        Returns a float between 0.0 and 1.0.
        Higher = better match.
        """
        name = strategy.name
        config = strategy.config

        # Component 1: Historical win rate
        win_rate_score = self._score_historical_win_rate(name)

        # Component 2: Market condition match
        condition_score = self._score_condition_match(config, market_condition)

        # Component 3: Feature alignment
        alignment_score = self._score_feature_alignment(config, features)

        # Component 4: Recency of performance
        recency_score = self._score_recency(name)

        # Weighted combination
        total = (
            win_rate_score * HISTORICAL_WIN_RATE_WEIGHT
            + condition_score * CONDITION_MATCH_WEIGHT
            + alignment_score * FEATURE_ALIGNMENT_WEIGHT
            + recency_score * RECENCY_WEIGHT
        )

        # Clamp to [0, 1]
        total = max(0.0, min(1.0, total))

        logger.debug(
            "ScoreEngine: {} for {} -> {:.3f} "
            "(win_rate={:.2f}, condition={:.2f}, alignment={:.2f}, recency={:.2f})",
            name, ticker, total,
            win_rate_score, condition_score, alignment_score, recency_score,
        )

        return total

    def select_best_strategies(
        self,
        ticker: str,
        features: dict[str, float],
        market_condition: MarketCondition,
        strategies: list[StrategyProtocol],
        top_n: int = 3,
    ) -> list[tuple[StrategyProtocol, float]]:
        """
        Evaluate all strategies and return the top N sorted by score.

        Only returns strategies that exceed the STRATEGY_SCORE_THRESHOLD.
        """
        scored: list[tuple[StrategyProtocol, float]] = []

        for strategy in strategies:
            if not strategy.config.is_active:
                continue

            score = self.score_strategy(strategy, ticker, features, market_condition)

            if score >= settings.STRATEGY_SCORE_THRESHOLD:
                scored.append((strategy, score))

        # Sort descending
        scored.sort(key=lambda x: x[1], reverse=True)

        result = scored[:top_n]
        logger.info(
            "ScoreEngine: selected {} strategies for {} (from {} candidates)",
            len(result), ticker, len(strategies),
        )
        return result

    async def parallel_evaluate(
        self,
        ticker: str,
        strategies: list[StrategyProtocol],
        features: dict[str, float],
        market_condition: MarketCondition,
    ) -> dict[str, float]:
        """
        Evaluate all strategies in parallel using asyncio.

        Returns dict mapping strategy_name -> score.
        """
        loop = asyncio.get_event_loop()

        async def _score_one(strat: StrategyProtocol) -> tuple[str, float]:
            # Run in executor since scoring is CPU-bound
            score = await loop.run_in_executor(
                None,
                self.score_strategy,
                strat,
                ticker,
                features,
                market_condition,
            )
            return strat.name, score

        tasks = [_score_one(s) for s in strategies if s.config.is_active]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scores: dict[str, float] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.warning("ScoreEngine: parallel eval error: {}", result)
                continue
            name, score = result
            scores[name] = score

        logger.info(
            "ScoreEngine: parallel evaluated {} strategies for {}", len(scores), ticker,
        )
        return scores

    # ------------------------------------------------------------------
    # Scoring components
    # ------------------------------------------------------------------

    def _score_historical_win_rate(self, strategy_name: str) -> float:
        """
        Score based on historical win rate.

        Returns 0.5 (neutral) if not enough trades for statistics.
        """
        perf = self._performances.get(strategy_name)

        if perf is None or perf.total_trades < MIN_TRADES_FOR_STATS:
            # Not enough data -- return neutral score
            return 0.5

        # Win rate directly maps to score (already 0-1)
        # Apply a slight bonus for profit factor > 1
        base_score = perf.win_rate

        if perf.profit_factor > 2.0:
            base_score = min(1.0, base_score + 0.1)
        elif perf.profit_factor > 1.5:
            base_score = min(1.0, base_score + 0.05)
        elif perf.profit_factor < 0.8:
            base_score = max(0.0, base_score - 0.1)

        # Penalize high drawdown
        if perf.max_drawdown > 0.1:  # > 10%
            base_score = max(0.0, base_score - 0.15)
        elif perf.max_drawdown > 0.05:  # > 5%
            base_score = max(0.0, base_score - 0.05)

        return base_score

    def _score_condition_match(
        self,
        config: StrategyConfig,
        market_condition: MarketCondition,
    ) -> float:
        """
        Score how well the current market conditions match the strategy's
        expected conditions.
        """
        expected = config.expected_market_condition
        if not expected:
            # Strategy works in any condition -- neutral score
            return 0.6

        current_regime = market_condition.market_regime

        # Direct regime match
        if expected == current_regime:
            score = 1.0
        elif expected in ("bull", "bear") and current_regime == "range":
            score = 0.5  # Range is partially acceptable
        elif expected == "volatile" and current_regime == "volatile":
            score = 1.0
        elif expected == "range" and current_regime in ("bull", "bear"):
            score = 0.4
        else:
            score = 0.2  # Mismatch

        # Bonus/penalty based on VIX-equivalent
        if market_condition.vix > 30 and expected == "volatile":
            score = min(1.0, score + 0.1)
        elif market_condition.vix > 30 and expected != "volatile":
            score = max(0.0, score - 0.1)

        # Check performance by condition from historical data
        perf = self._performances.get(config.strategy_name)
        if perf and perf.performance_by_condition:
            condition_perf = perf.performance_by_condition.get(current_regime, {})
            hist_win_rate = condition_perf.get("win_rate", 0.5)
            # Blend historical condition-specific win rate
            score = score * 0.6 + hist_win_rate * 0.4

        return max(0.0, min(1.0, score))

    def _score_feature_alignment(
        self,
        config: StrategyConfig,
        features: dict[str, float],
    ) -> float:
        """
        Score how well the ticker's current features align with the
        strategy's requirements.
        """
        required_features = config.feature_requirements
        if not required_features:
            return 0.6  # No specific requirements -- neutral

        entry_conditions = config.entry_conditions
        if not entry_conditions:
            # Check if required features are present at all
            present = sum(1 for f in required_features if f in features)
            return present / len(required_features) if required_features else 0.6

        # Score each condition
        condition_scores: list[float] = []

        for feature_name in required_features:
            if feature_name not in features:
                condition_scores.append(0.0)
                continue

            value = features[feature_name]
            condition = entry_conditions.get(feature_name, {})

            if not condition:
                # Feature exists but no specific condition -- partial credit
                condition_scores.append(0.5)
                continue

            score = self._evaluate_condition(value, condition)
            condition_scores.append(score)

        if not condition_scores:
            return 0.6

        return sum(condition_scores) / len(condition_scores)

    def _score_recency(self, strategy_name: str) -> float:
        """
        Score based on recent trading performance.

        More weight on recent trades using exponential decay.
        """
        trades = self._trade_history.get(strategy_name, [])
        if not trades:
            return 0.5  # Neutral if no history

        now = datetime.now()
        weighted_wins = 0.0
        total_weight = 0.0

        for trade in trades:
            days_ago = (now - trade.exit_time).total_seconds() / 86400.0
            if days_ago < 0:
                days_ago = 0

            # Exponential decay weight
            weight = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)

            total_weight += weight
            if trade.pnl > 0:
                weighted_wins += weight

        if total_weight == 0:
            return 0.5

        recency_win_rate = weighted_wins / total_weight

        # Also consider recent streak
        recent_trades = sorted(trades, key=lambda t: t.exit_time, reverse=True)[:5]
        if recent_trades:
            recent_wins = sum(1 for t in recent_trades if t.pnl > 0)
            streak_score = recent_wins / len(recent_trades)
            # Blend recency win rate with streak
            return recency_win_rate * 0.7 + streak_score * 0.3

        return recency_win_rate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_condition(value: float, condition: dict[str, Any]) -> float:
        """
        Evaluate a single feature value against a condition spec.

        Condition dict can have:
        - "min": minimum value
        - "max": maximum value
        - "ideal": ideal value (score decays with distance)
        - "above": value must be above this
        - "below": value must be below this
        """
        score = 1.0

        if "min" in condition and value < condition["min"]:
            # How far below minimum -- linear decay
            deficit = condition["min"] - value
            range_size = abs(condition["min"]) if condition["min"] != 0 else 1.0
            score *= max(0.0, 1.0 - deficit / range_size)

        if "max" in condition and value > condition["max"]:
            excess = value - condition["max"]
            range_size = abs(condition["max"]) if condition["max"] != 0 else 1.0
            score *= max(0.0, 1.0 - excess / range_size)

        if "above" in condition:
            if value <= condition["above"]:
                score *= 0.1
            else:
                score *= 1.0

        if "below" in condition:
            if value >= condition["below"]:
                score *= 0.1
            else:
                score *= 1.0

        if "ideal" in condition:
            distance = abs(value - condition["ideal"])
            ideal_range = abs(condition["ideal"]) * 0.5 if condition["ideal"] != 0 else 1.0
            ideal_score = max(0.0, 1.0 - distance / ideal_range)
            score *= ideal_score

        return max(0.0, min(1.0, score))
