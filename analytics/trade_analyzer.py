"""
Trade analysis module for KabuAI day trading.

Provides detailed analysis of individual trades, strategy-level
aggregated performance, and comparative strategy ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from core.models import StrategyPerformance, TradeResult


class TradeAnalyzer:
    """
    Analyzes individual trades and strategy-level performance.

    Computes:
    - Per-trade metrics (P&L, holding time, entry efficiency)
    - Strategy-level aggregated metrics (win rate, profit factor,
      Sharpe, Sortino, max drawdown, expectancy)
    - Cross-strategy comparisons and rankings

    Usage::

        analyzer = TradeAnalyzer()
        metrics = analyzer.analyze_trade(trade)
        perf = analyzer.analyze_strategy("momentum_breakout", trades, period_days=30)
        ranking = analyzer.compare_strategies([perf1, perf2, perf3])
    """

    # ------------------------------------------------------------------
    # Per-trade analysis
    # ------------------------------------------------------------------

    def analyze_trade(self, trade: TradeResult) -> dict[str, Any]:
        """
        Analyze a single completed trade and return detailed metrics.

        Returns dict with:
        - pnl, pnl_pct
        - holding_minutes
        - is_winner
        - entry_efficiency: how close entry was to the best price
        - risk_reward_achieved: actual risk/reward ratio
        - market_condition at entry
        """
        # Basic P&L
        if trade.direction == "long":
            pnl = trade.exit_price - trade.entry_price
            pnl_pct = pnl / trade.entry_price * 100 if trade.entry_price else 0
        else:
            pnl = trade.entry_price - trade.exit_price
            pnl_pct = pnl / trade.entry_price * 100 if trade.entry_price else 0

        # Holding time
        if trade.entry_time and trade.exit_time:
            holding = trade.exit_time - trade.entry_time
            holding_minutes = int(holding.total_seconds() / 60)
        else:
            holding_minutes = trade.holding_minutes

        # Per-share values (100 share lot)
        pnl_per_lot = pnl * 100  # 100株単位

        is_winner = pnl > 0

        # Entry/exit efficiency from features
        features_entry = trade.features_at_entry
        features_exit = trade.features_at_exit

        entry_efficiency = self._calculate_entry_efficiency(trade, features_entry)

        metrics = {
            "trade_id": trade.id,
            "ticker": trade.ticker,
            "strategy_name": trade.strategy_name,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "pnl_per_lot": pnl_per_lot,
            "holding_minutes": holding_minutes,
            "is_winner": is_winner,
            "entry_efficiency": entry_efficiency,
            "entry_reason": trade.entry_reason,
            "exit_reason": trade.exit_reason,
            "market_condition": trade.market_condition,
            "features_at_entry": features_entry,
            "features_at_exit": features_exit,
        }

        logger.debug(
            "TradeAnalyzer: trade {} {} {} P&L={:.0f}円 ({:.2f}%) in {}min",
            trade.id, trade.ticker, "WIN" if is_winner else "LOSS",
            pnl_per_lot, pnl_pct, holding_minutes,
        )

        return metrics

    # ------------------------------------------------------------------
    # Strategy-level analysis
    # ------------------------------------------------------------------

    def analyze_strategy(
        self,
        strategy_name: str,
        trades: list[TradeResult],
        period_days: int = 30,
    ) -> StrategyPerformance:
        """
        Analyze aggregated performance for a strategy over a given period.

        Args:
            strategy_name: Name of the strategy to analyze.
            trades: List of completed trades for this strategy.
            period_days: Only consider trades within this many days.

        Returns:
            StrategyPerformance with all computed metrics.
        """
        # Filter to period
        cutoff = datetime.now() - timedelta(days=period_days)
        filtered = [
            t for t in trades
            if t.strategy_name == strategy_name and t.exit_time >= cutoff
        ]

        if not filtered:
            logger.info(
                "TradeAnalyzer: no trades for {} in last {} days",
                strategy_name, period_days,
            )
            return StrategyPerformance(strategy_name=strategy_name)

        metrics = self.calculate_metrics(filtered)

        # Performance by market condition
        perf_by_condition = self._breakdown_by_condition(filtered)

        # Performance by event type
        perf_by_event = self._breakdown_by_event(filtered)

        performance = StrategyPerformance(
            strategy_name=strategy_name,
            total_trades=metrics["total_trades"],
            wins=metrics["wins"],
            losses=metrics["losses"],
            win_rate=metrics["win_rate"],
            profit_factor=metrics["profit_factor"],
            avg_pnl=metrics["avg_pnl"],
            avg_holding_minutes=metrics["avg_holding_minutes"],
            sharpe_ratio=metrics["sharpe_ratio"],
            max_drawdown=metrics["max_drawdown"],
            performance_by_condition=perf_by_condition,
            performance_by_event=perf_by_event,
        )

        logger.info(
            "TradeAnalyzer: {} - {} trades, {:.1%} win rate, PF={:.2f}, "
            "Sharpe={:.2f}, MaxDD={:.2%}",
            strategy_name, performance.total_trades, performance.win_rate,
            performance.profit_factor, performance.sharpe_ratio,
            performance.max_drawdown,
        )

        return performance

    def compare_strategies(
        self,
        performances: list[StrategyPerformance],
    ) -> list[dict[str, Any]]:
        """
        Compare multiple strategies and return a ranked list.

        Ranking is based on a composite score:
        - win_rate * 0.25
        - profit_factor * 0.25 (normalized)
        - sharpe_ratio * 0.25 (normalized)
        - (1 - max_drawdown) * 0.25

        Returns list of dicts sorted by composite score (descending).
        """
        if not performances:
            return []

        ranked: list[dict[str, Any]] = []

        # Normalize profit factor and sharpe for scoring
        max_pf = max((p.profit_factor for p in performances), default=1.0) or 1.0
        max_sharpe = max((abs(p.sharpe_ratio) for p in performances), default=1.0) or 1.0

        for perf in performances:
            norm_pf = min(perf.profit_factor / max_pf, 1.0) if max_pf > 0 else 0
            norm_sharpe = (
                min(max(perf.sharpe_ratio, 0) / max_sharpe, 1.0) if max_sharpe > 0 else 0
            )
            dd_score = max(0.0, 1.0 - perf.max_drawdown)

            composite = (
                perf.win_rate * 0.25
                + norm_pf * 0.25
                + norm_sharpe * 0.25
                + dd_score * 0.25
            )

            ranked.append({
                "strategy_name": perf.strategy_name,
                "composite_score": composite,
                "total_trades": perf.total_trades,
                "win_rate": perf.win_rate,
                "profit_factor": perf.profit_factor,
                "sharpe_ratio": perf.sharpe_ratio,
                "max_drawdown": perf.max_drawdown,
                "avg_pnl": perf.avg_pnl,
                "avg_holding_minutes": perf.avg_holding_minutes,
            })

        ranked.sort(key=lambda r: r["composite_score"], reverse=True)

        for i, r in enumerate(ranked, 1):
            logger.info(
                "TradeAnalyzer: Rank {} - {} (score={:.3f}, WR={:.1%}, PF={:.2f})",
                i, r["strategy_name"], r["composite_score"],
                r["win_rate"], r["profit_factor"],
            )

        return ranked

    # ------------------------------------------------------------------
    # Core metrics calculation
    # ------------------------------------------------------------------

    def calculate_metrics(self, trades: list[TradeResult]) -> dict[str, Any]:
        """
        Calculate comprehensive trading metrics from a list of trades.

        Returns dict with:
        - total_trades, wins, losses
        - win_rate, profit_factor
        - sharpe_ratio, sortino_ratio, max_drawdown
        - avg_winner, avg_loser, expectancy
        - avg_holding_time (minutes)
        - best_trade, worst_trade (P&L)
        """
        if not trades:
            return self._empty_metrics()

        # Classify wins and losses
        pnls = []
        winning_pnls = []
        losing_pnls = []

        for trade in trades:
            if trade.direction == "long":
                pnl = trade.exit_price - trade.entry_price
            else:
                pnl = trade.entry_price - trade.exit_price
            pnls.append(pnl)

            if pnl > 0:
                winning_pnls.append(pnl)
            elif pnl < 0:
                losing_pnls.append(pnl)

        total_trades = len(trades)
        wins = len(winning_pnls)
        losses = len(losing_pnls)
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Averages
        avg_winner = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0.0
        avg_loser = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0.0

        # Profit factor
        gross_profit = sum(winning_pnls)
        gross_loss = abs(sum(losing_pnls))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Expectancy per trade
        expectancy = sum(pnls) / total_trades if total_trades > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~245 trading days)
        avg_pnl = sum(pnls) / len(pnls)
        if len(pnls) > 1:
            std_pnl = math.sqrt(sum((p - avg_pnl) ** 2 for p in pnls) / (len(pnls) - 1))
        else:
            std_pnl = 0.0
        sharpe_ratio = (avg_pnl / std_pnl * math.sqrt(245)) if std_pnl > 0 else 0.0

        # Sortino ratio (only downside deviation)
        negative_pnls = [p for p in pnls if p < 0]
        if negative_pnls:
            downside_dev = math.sqrt(
                sum(p ** 2 for p in negative_pnls) / len(negative_pnls),
            )
        else:
            downside_dev = 0.0
        sortino_ratio = (avg_pnl / downside_dev * math.sqrt(245)) if downside_dev > 0 else 0.0

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown(pnls)

        # Holding time
        holding_times = [t.holding_minutes for t in trades if t.holding_minutes > 0]
        avg_holding = sum(holding_times) / len(holding_times) if holding_times else 0.0

        # Best / worst
        best_trade = max(pnls)
        worst_trade = min(pnls)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_pnl": avg_pnl,
            "avg_winner": avg_winner,
            "avg_loser": avg_loser,
            "expectancy": expectancy,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "avg_holding_minutes": avg_holding,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_max_drawdown(pnls: list[float]) -> float:
        """Calculate maximum drawdown from a sequence of P&Ls."""
        if not pnls:
            return 0.0

        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            drawdown = (peak - cumulative) / abs(peak) if peak != 0 else 0.0
            max_dd = max(max_dd, drawdown)

        return max_dd

    @staticmethod
    def _calculate_entry_efficiency(
        trade: TradeResult,
        features: dict[str, Any],
    ) -> float:
        """
        Calculate entry efficiency: how close was entry to optimal price.

        Returns value between 0 (worst) and 1 (best).
        For long: optimal entry = lowest price in the bar.
        For short: optimal entry = highest price in the bar.
        """
        high = features.get("entry_bar_high", trade.entry_price)
        low = features.get("entry_bar_low", trade.entry_price)

        if high == low:
            return 0.5  # Can't evaluate

        bar_range = high - low
        if bar_range <= 0:
            return 0.5

        if trade.direction == "long":
            # Best entry is at low; how close were we?
            distance_from_best = trade.entry_price - low
            return max(0.0, 1.0 - distance_from_best / bar_range)
        else:
            # Best entry is at high
            distance_from_best = high - trade.entry_price
            return max(0.0, 1.0 - distance_from_best / bar_range)

    @staticmethod
    def _breakdown_by_condition(
        trades: list[TradeResult],
    ) -> dict[str, dict[str, Any]]:
        """Break down performance by market condition."""
        by_condition: dict[str, list[float]] = defaultdict(list)

        for trade in trades:
            condition = trade.market_condition or "unknown"
            if trade.direction == "long":
                pnl = trade.exit_price - trade.entry_price
            else:
                pnl = trade.entry_price - trade.exit_price
            by_condition[condition].append(pnl)

        result: dict[str, dict[str, Any]] = {}
        for condition, pnls in by_condition.items():
            wins = sum(1 for p in pnls if p > 0)
            result[condition] = {
                "trades": len(pnls),
                "wins": wins,
                "win_rate": wins / len(pnls) if pnls else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
                "total_pnl": sum(pnls),
            }

        return result

    @staticmethod
    def _breakdown_by_event(
        trades: list[TradeResult],
    ) -> dict[str, dict[str, Any]]:
        """Break down performance by event type from entry features."""
        by_event: dict[str, list[float]] = defaultdict(list)

        for trade in trades:
            event = trade.features_at_entry.get("event_type", "none")
            if isinstance(event, float):
                event = str(event)
            if trade.direction == "long":
                pnl = trade.exit_price - trade.entry_price
            else:
                pnl = trade.entry_price - trade.exit_price
            by_event[event].append(pnl)

        result: dict[str, dict[str, Any]] = {}
        for event, pnls in by_event.items():
            wins = sum(1 for p in pnls if p > 0)
            result[event] = {
                "trades": len(pnls),
                "wins": wins,
                "win_rate": wins / len(pnls) if pnls else 0.0,
                "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            }

        return result

    @staticmethod
    def _empty_metrics() -> dict[str, Any]:
        """Return empty metrics dict."""
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_pnl": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "expectancy": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_holding_minutes": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
        }
