"""Unit tests for the KabuAI-Daytrade strategy scoring engine.

Tests the scoring, ranking, and selection of strategies under various
market conditions.  The score engine evaluates each strategy based on
historical performance, current market regime fitness, and feature
confidence to decide which strategies should be active for a given
trading session.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from core.models import StrategyPerformance, MarketCondition


# ------------------------------------------------------------------
# Score engine implementation (inline, since it may not exist yet)
# ------------------------------------------------------------------


class StrategyScoreEngine:
    """Score and rank strategies for a given market condition.

    The engine combines three weighted components:
    1. **Historical performance** — win rate, profit factor, Sharpe
    2. **Regime fitness** — how well the strategy matches the current
       market regime (bull/bear/range/volatile)
    3. **Recency** — more recent trades count more heavily

    The final score is normalised to [0, 1].
    """

    # Weight configuration
    WEIGHT_WIN_RATE: float = 0.25
    WEIGHT_PROFIT_FACTOR: float = 0.25
    WEIGHT_SHARPE: float = 0.20
    WEIGHT_REGIME_FIT: float = 0.30

    # Regime fitness mapping: {strategy_expected_condition: {actual_regime: fit_score}}
    REGIME_FIT_TABLE: dict[str, dict[str, float]] = {
        "bull": {"bull": 1.0, "range": 0.4, "bear": 0.1, "volatile": 0.3},
        "bear": {"bear": 1.0, "range": 0.3, "bull": 0.1, "volatile": 0.4},
        "range": {"range": 1.0, "bull": 0.5, "bear": 0.5, "volatile": 0.2},
        "volatile": {"volatile": 1.0, "bull": 0.3, "bear": 0.3, "range": 0.2},
        "any": {"bull": 0.7, "bear": 0.7, "range": 0.7, "volatile": 0.7},
        "": {"bull": 0.5, "bear": 0.5, "range": 0.5, "volatile": 0.5},
    }

    def score_strategy(
        self,
        performance: StrategyPerformance,
        market_condition: MarketCondition,
        expected_condition: str = "",
    ) -> float:
        """Compute a composite score for a single strategy.

        Args:
            performance: Historical performance metrics.
            market_condition: Current market snapshot.
            expected_condition: The market condition the strategy
                was designed for (e.g. "bull", "range").

        Returns:
            Score in [0.0, 1.0].
        """
        # Win rate component (0-1)
        wr_score = min(1.0, performance.win_rate)

        # Profit factor component (cap at 3.0 -> 1.0)
        pf_score = min(1.0, performance.profit_factor / 3.0) if performance.profit_factor >= 0 else 0.0

        # Sharpe component (cap at 2.0 -> 1.0)
        sharpe_score = min(1.0, max(0.0, performance.sharpe_ratio / 2.0))

        # Regime fitness
        regime = market_condition.market_regime
        fit_map = self.REGIME_FIT_TABLE.get(
            expected_condition, self.REGIME_FIT_TABLE[""]
        )
        regime_score = fit_map.get(regime, 0.5)

        # Minimum trade count penalty: strategies with fewer than 10
        # trades get a confidence discount
        trade_count_factor = min(1.0, performance.total_trades / 10.0)

        composite = (
            self.WEIGHT_WIN_RATE * wr_score
            + self.WEIGHT_PROFIT_FACTOR * pf_score
            + self.WEIGHT_SHARPE * sharpe_score
            + self.WEIGHT_REGIME_FIT * regime_score
        ) * trade_count_factor

        return round(min(1.0, max(0.0, composite)), 4)

    def rank_strategies(
        self,
        strategies: dict[str, tuple[StrategyPerformance, str]],
        market_condition: MarketCondition,
    ) -> list[tuple[str, float]]:
        """Rank all strategies by score (descending).

        Args:
            strategies: ``{name: (performance, expected_condition)}``.
            market_condition: Current market snapshot.

        Returns:
            List of ``(strategy_name, score)`` sorted highest first.
        """
        scored: list[tuple[str, float]] = []
        for name, (perf, expected) in strategies.items():
            score = self.score_strategy(perf, market_condition, expected)
            scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select_strategies(
        self,
        strategies: dict[str, tuple[StrategyPerformance, str]],
        market_condition: MarketCondition,
        threshold: float = 0.5,
        max_active: int = 3,
    ) -> list[str]:
        """Select the best strategies that pass the score threshold.

        Args:
            strategies: ``{name: (performance, expected_condition)}``.
            market_condition: Current market snapshot.
            threshold: Minimum score to be selected.
            max_active: Maximum number of strategies to activate.

        Returns:
            List of selected strategy names (up to ``max_active``).
        """
        ranked = self.rank_strategies(strategies, market_condition)
        selected = [name for name, score in ranked if score >= threshold]
        return selected[:max_active]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def engine() -> StrategyScoreEngine:
    return StrategyScoreEngine()


@pytest.fixture
def bull_market() -> MarketCondition:
    return MarketCondition(
        nikkei225_change_pct=1.5,
        topix_change_pct=1.2,
        usd_jpy=155.0,
        vix=15.0,
        market_regime="bull",
        sector_momentum={"電気機器": 0.05, "銀行業": 0.03},
    )


@pytest.fixture
def bear_market() -> MarketCondition:
    return MarketCondition(
        nikkei225_change_pct=-2.0,
        topix_change_pct=-1.8,
        usd_jpy=142.0,
        vix=28.0,
        market_regime="bear",
        sector_momentum={"電気機器": -0.04, "銀行業": -0.02},
    )


@pytest.fixture
def range_market() -> MarketCondition:
    return MarketCondition(
        nikkei225_change_pct=0.1,
        topix_change_pct=0.05,
        usd_jpy=150.0,
        vix=18.0,
        market_regime="range",
    )


@pytest.fixture
def volatile_market() -> MarketCondition:
    return MarketCondition(
        nikkei225_change_pct=-0.5,
        topix_change_pct=-0.3,
        usd_jpy=148.0,
        vix=32.0,
        market_regime="volatile",
    )


def _make_perf(
    name: str,
    total_trades: int = 50,
    win_rate: float = 0.6,
    profit_factor: float = 1.5,
    sharpe: float = 1.0,
    max_dd: float = 30_000,
) -> StrategyPerformance:
    """Create a StrategyPerformance with specified values."""
    return StrategyPerformance(
        strategy_name=name,
        total_trades=total_trades,
        wins=int(total_trades * win_rate),
        losses=total_trades - int(total_trades * win_rate),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_pnl=5000 * (profit_factor - 1),
        avg_holding_minutes=45,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
    )


# ------------------------------------------------------------------
# Tests: individual strategy scoring
# ------------------------------------------------------------------


class TestStrategyScoring:
    """Test scoring of individual strategies."""

    def test_high_performance_bull_strategy_in_bull_market(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        perf = _make_perf("momentum", win_rate=0.7, profit_factor=2.5, sharpe=1.8)
        score = engine.score_strategy(perf, bull_market, expected_condition="bull")

        # Strong strategy in matching regime should score high
        assert score > 0.7

    def test_high_performance_bull_strategy_in_bear_market(
        self, engine: StrategyScoreEngine, bear_market: MarketCondition
    ) -> None:
        perf = _make_perf("momentum", win_rate=0.7, profit_factor=2.5, sharpe=1.8)
        score = engine.score_strategy(perf, bear_market, expected_condition="bull")

        # Good strategy but wrong regime — score should be lower
        assert score < 0.7

    def test_poor_performance_strategy(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        perf = _make_perf("bad_strategy", win_rate=0.3, profit_factor=0.5, sharpe=-0.5)
        score = engine.score_strategy(perf, bull_market, expected_condition="bull")

        # Poor metrics even in matching regime
        assert score < 0.5

    def test_score_range_bounded(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        """Scores should always be in [0, 1]."""
        # Extreme good performance
        perf_good = _make_perf("best", win_rate=1.0, profit_factor=10.0, sharpe=5.0)
        score_good = engine.score_strategy(perf_good, bull_market, "bull")
        assert 0.0 <= score_good <= 1.0

        # Extreme bad performance
        perf_bad = _make_perf("worst", win_rate=0.0, profit_factor=0.0, sharpe=-3.0, total_trades=0)
        score_bad = engine.score_strategy(perf_bad, bull_market, "bull")
        assert 0.0 <= score_bad <= 1.0

    def test_few_trades_penalised(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        """Strategies with very few trades get a confidence penalty."""
        perf_many = _make_perf("proven", total_trades=100, win_rate=0.65, profit_factor=1.8, sharpe=1.2)
        perf_few = _make_perf("unproven", total_trades=3, win_rate=0.65, profit_factor=1.8, sharpe=1.2)

        score_many = engine.score_strategy(perf_many, bull_market, "bull")
        score_few = engine.score_strategy(perf_few, bull_market, "bull")

        assert score_many > score_few

    def test_regime_fitness_dominates_when_mismatch(
        self, engine: StrategyScoreEngine, bear_market: MarketCondition
    ) -> None:
        """A mediocre bear strategy should beat a strong bull strategy in bear market."""
        perf_bull = _make_perf("strong_bull", win_rate=0.75, profit_factor=2.0, sharpe=1.5)
        perf_bear = _make_perf("ok_bear", win_rate=0.55, profit_factor=1.3, sharpe=0.8)

        score_bull = engine.score_strategy(perf_bull, bear_market, "bull")
        score_bear = engine.score_strategy(perf_bear, bear_market, "bear")

        # Bear strategy should score higher in bear market due to regime fit
        assert score_bear > score_bull

    def test_any_condition_strategy_scores_moderately(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        """Strategy with expected_condition='any' gets moderate regime score."""
        perf = _make_perf("all_weather", win_rate=0.6, profit_factor=1.5, sharpe=1.0)
        score = engine.score_strategy(perf, bull_market, "any")
        assert 0.3 < score < 0.9


# ------------------------------------------------------------------
# Tests: strategy ranking
# ------------------------------------------------------------------


class TestStrategyRanking:
    """Test ranking multiple strategies."""

    def test_ranking_order_in_bull_market(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        strategies = {
            "momentum": (_make_perf("momentum", win_rate=0.7, profit_factor=2.0, sharpe=1.5), "bull"),
            "reversal": (_make_perf("reversal", win_rate=0.55, profit_factor=1.2, sharpe=0.7), "range"),
            "gap": (_make_perf("gap", win_rate=0.6, profit_factor=1.5, sharpe=1.0), "bull"),
        }

        ranked = engine.rank_strategies(strategies, bull_market)

        assert len(ranked) == 3
        names = [name for name, _ in ranked]
        # momentum should be first (best perf + matching regime)
        assert names[0] == "momentum"
        # All scores descending
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_ranking_order_in_bear_market(
        self, engine: StrategyScoreEngine, bear_market: MarketCondition
    ) -> None:
        strategies = {
            "momentum": (_make_perf("momentum", win_rate=0.7, profit_factor=2.0, sharpe=1.5), "bull"),
            "mean_revert": (_make_perf("mean_revert", win_rate=0.6, profit_factor=1.5, sharpe=1.0), "bear"),
        }

        ranked = engine.rank_strategies(strategies, bear_market)
        names = [name for name, _ in ranked]
        assert names[0] == "mean_revert"


# ------------------------------------------------------------------
# Tests: strategy selection
# ------------------------------------------------------------------


class TestStrategySelection:
    """Test selecting active strategies based on scores and thresholds."""

    def test_selects_above_threshold(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        strategies = {
            "good": (_make_perf("good", win_rate=0.7, profit_factor=2.0, sharpe=1.5), "bull"),
            "mediocre": (_make_perf("mediocre", win_rate=0.5, profit_factor=1.0, sharpe=0.3), "range"),
            "bad": (_make_perf("bad", win_rate=0.2, profit_factor=0.3, sharpe=-1.0, total_trades=5), "bear"),
        }

        selected = engine.select_strategies(strategies, bull_market, threshold=0.5)
        assert "good" in selected
        # "bad" with poor performance + wrong regime + few trades should not be selected
        assert "bad" not in selected

    def test_max_active_limit(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        strategies = {
            f"strat_{i}": (_make_perf(f"strat_{i}", win_rate=0.65, profit_factor=1.8, sharpe=1.2), "bull")
            for i in range(10)
        }

        selected = engine.select_strategies(strategies, bull_market, threshold=0.3, max_active=3)
        assert len(selected) <= 3

    def test_no_selection_when_all_below_threshold(
        self, engine: StrategyScoreEngine, bear_market: MarketCondition
    ) -> None:
        strategies = {
            "bad1": (_make_perf("bad1", win_rate=0.2, profit_factor=0.3, sharpe=-1.0, total_trades=2), "bull"),
            "bad2": (_make_perf("bad2", win_rate=0.15, profit_factor=0.2, sharpe=-2.0, total_trades=1), "bull"),
        }

        selected = engine.select_strategies(strategies, bear_market, threshold=0.6)
        assert len(selected) == 0

    def test_empty_strategies(
        self, engine: StrategyScoreEngine, bull_market: MarketCondition
    ) -> None:
        selected = engine.select_strategies({}, bull_market)
        assert selected == []


# ------------------------------------------------------------------
# Tests: various market conditions
# ------------------------------------------------------------------


class TestMarketConditionVariations:
    """Test scoring across all four market regimes."""

    @pytest.mark.parametrize(
        "regime,expected_best",
        [
            ("bull", "momentum"),
            ("bear", "short_seller"),
            ("range", "mean_revert"),
            ("volatile", "vol_breakout"),
        ],
    )
    def test_best_strategy_matches_regime(
        self,
        engine: StrategyScoreEngine,
        regime: str,
        expected_best: str,
    ) -> None:
        market = MarketCondition(market_regime=regime)

        strategies = {
            "momentum": (_make_perf("momentum", win_rate=0.65, profit_factor=1.8, sharpe=1.2), "bull"),
            "short_seller": (_make_perf("short_seller", win_rate=0.6, profit_factor=1.5, sharpe=1.0), "bear"),
            "mean_revert": (_make_perf("mean_revert", win_rate=0.6, profit_factor=1.5, sharpe=1.0), "range"),
            "vol_breakout": (_make_perf("vol_breakout", win_rate=0.6, profit_factor=1.5, sharpe=1.0), "volatile"),
        }

        ranked = engine.rank_strategies(strategies, market)
        top_strategy = ranked[0][0]
        assert top_strategy == expected_best, (
            f"In {regime} market, expected {expected_best} to rank first but got {top_strategy}"
        )

    def test_score_consistency_across_regimes(
        self, engine: StrategyScoreEngine
    ) -> None:
        """Same strategy should score highest in its matching regime."""
        perf = _make_perf("momentum", win_rate=0.65, profit_factor=1.8, sharpe=1.2)

        scores = {}
        for regime in ("bull", "bear", "range", "volatile"):
            market = MarketCondition(market_regime=regime)
            scores[regime] = engine.score_strategy(perf, market, "bull")

        # Momentum (expected=bull) should score highest in bull
        assert scores["bull"] == max(scores.values())
