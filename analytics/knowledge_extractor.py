"""
Knowledge extraction engine for KabuAI day trading (超重要).

Extracts actionable patterns from trading history:
- Winning patterns (勝ちパターン)
- Losing patterns (負けパターン)
- Strategy-specific insights
- Improvement candidates (parameter/condition refinements)

All improvements are proposed as CandidateUpdate with status="pending".
NEVER auto-applied.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Optional

from loguru import logger

from core.models import (
    CandidateUpdate,
    KnowledgeEntry,
    StrategyPerformance,
    TradeResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_CLUSTER_SIZE = 3            # Minimum trades to form a pattern
WIN_RATE_THRESHOLD = 0.65       # 65%以上で「勝ちパターン」認定
LOSS_RATE_THRESHOLD = 0.65      # 65%以上で「負けパターン」認定
FEATURE_KEYS = [
    "rsi", "volume_ratio", "gap_pct", "atr_pct", "macd_histogram",
    "bb_position", "vwap_deviation", "price_change_pct",
    "spread_pct", "momentum_5", "momentum_10",
]


class KnowledgeExtractor:
    """
    Extracts trading knowledge from completed trades.

    The core insight loop:
    1. Cluster winning/losing trades by features
    2. Identify common feature ranges and conditions
    3. Output human-readable knowledge entries (日本語)
    4. Generate improvement candidates (pending, never auto-applied)

    Usage::

        extractor = KnowledgeExtractor()
        win_patterns = extractor.extract_win_patterns(winning_trades)
        loss_patterns = extractor.extract_loss_patterns(losing_trades)
        candidates = extractor.generate_improvement_candidates(insights)
    """

    # ------------------------------------------------------------------
    # Win patterns
    # ------------------------------------------------------------------

    def extract_win_patterns(
        self, winning_trades: list[TradeResult],
    ) -> list[KnowledgeEntry]:
        """
        Extract winning patterns from a set of winning trades.

        Clusters trades by strategy and market condition, then finds
        common feature ranges that appear in winning trades.

        Output example:
            "勝ちパターン: momentum_breakout は RSI < 30, volume > 2x,
             bull market で勝率75% (12/16 trades)"
        """
        if len(winning_trades) < MIN_CLUSTER_SIZE:
            logger.info(
                "KnowledgeExtractor: not enough winning trades ({}) for pattern extraction",
                len(winning_trades),
            )
            return []

        logger.info(
            "KnowledgeExtractor: extracting win patterns from {} trades",
            len(winning_trades),
        )

        entries: list[KnowledgeEntry] = []

        # Group by strategy
        by_strategy: dict[str, list[TradeResult]] = defaultdict(list)
        for trade in winning_trades:
            by_strategy[trade.strategy_name].append(trade)

        for strategy_name, trades in by_strategy.items():
            if len(trades) < MIN_CLUSTER_SIZE:
                continue

            # Find common feature ranges
            feature_ranges = self._find_common_feature_ranges(trades)
            condition_counts = self._count_market_conditions(trades)

            # Best market condition
            best_condition = max(condition_counts, key=condition_counts.get) if condition_counts else "unknown"
            best_condition_count = condition_counts.get(best_condition, 0)

            # Build description
            feature_desc_parts: list[str] = []
            for feat, (low, high, median) in feature_ranges.items():
                feature_desc_parts.append(f"{feat}: {low:.2f}~{high:.2f} (中央値: {median:.2f})")

            feature_desc = ", ".join(feature_desc_parts[:5])  # Top 5 features

            content = (
                f"勝ちパターン: {strategy_name} は {feature_desc}, "
                f"{best_condition} market で勝率100% "
                f"({len(trades)} trades, condition={best_condition} {best_condition_count}回)"
            )

            trade_ids = [t.id for t in trades]
            confidence = min(1.0, len(trades) / 20.0)  # More trades = higher confidence

            entry = KnowledgeEntry(
                category="win_pattern",
                content=content,
                supporting_trades=trade_ids,
                confidence=confidence,
                auto_generated=True,
            )
            entries.append(entry)

            logger.info("KnowledgeExtractor: win pattern - {}", content[:100])

        # Cross-strategy patterns
        cross_patterns = self._find_cross_strategy_patterns(winning_trades, "win")
        entries.extend(cross_patterns)

        return entries

    # ------------------------------------------------------------------
    # Loss patterns
    # ------------------------------------------------------------------

    def extract_loss_patterns(
        self, losing_trades: list[TradeResult],
    ) -> list[KnowledgeEntry]:
        """
        Extract losing patterns from a set of losing trades.

        Identifies common failure modes:
        - Feature ranges that lead to losses
        - Market conditions that are unfavorable
        - Timing patterns (holding too long, etc.)
        """
        if len(losing_trades) < MIN_CLUSTER_SIZE:
            logger.info(
                "KnowledgeExtractor: not enough losing trades ({}) for pattern extraction",
                len(losing_trades),
            )
            return []

        logger.info(
            "KnowledgeExtractor: extracting loss patterns from {} trades",
            len(losing_trades),
        )

        entries: list[KnowledgeEntry] = []

        # Group by strategy
        by_strategy: dict[str, list[TradeResult]] = defaultdict(list)
        for trade in losing_trades:
            by_strategy[trade.strategy_name].append(trade)

        for strategy_name, trades in by_strategy.items():
            if len(trades) < MIN_CLUSTER_SIZE:
                continue

            feature_ranges = self._find_common_feature_ranges(trades)
            condition_counts = self._count_market_conditions(trades)

            worst_condition = max(condition_counts, key=condition_counts.get) if condition_counts else "unknown"

            # Analyze holding time patterns
            holding_times = [t.holding_minutes for t in trades if t.holding_minutes > 0]
            avg_holding = statistics.mean(holding_times) if holding_times else 0
            median_holding = statistics.median(holding_times) if holding_times else 0

            # Average loss
            losses = []
            for t in trades:
                if t.direction == "long":
                    losses.append(t.entry_price - t.exit_price)
                else:
                    losses.append(t.exit_price - t.entry_price)
            avg_loss = statistics.mean(losses) if losses else 0

            feature_desc_parts: list[str] = []
            for feat, (low, high, median) in feature_ranges.items():
                feature_desc_parts.append(f"{feat}: {low:.2f}~{high:.2f}")

            feature_desc = ", ".join(feature_desc_parts[:5])

            content = (
                f"負けパターン: {strategy_name} は {feature_desc}, "
                f"{worst_condition} market で負け "
                f"(平均損失: {avg_loss:.0f}円, 平均保有: {avg_holding:.0f}分, "
                f"{len(trades)} trades)"
            )

            entry = KnowledgeEntry(
                category="loss_pattern",
                content=content,
                supporting_trades=[t.id for t in trades],
                confidence=min(1.0, len(trades) / 20.0),
                auto_generated=True,
            )
            entries.append(entry)

            logger.info("KnowledgeExtractor: loss pattern - {}", content[:100])

        # Cross-strategy failure modes
        cross_patterns = self._find_cross_strategy_patterns(losing_trades, "loss")
        entries.extend(cross_patterns)

        return entries

    # ------------------------------------------------------------------
    # Strategy insights
    # ------------------------------------------------------------------

    def extract_strategy_insights(
        self,
        strategy_name: str,
        trades: list[TradeResult],
    ) -> list[KnowledgeEntry]:
        """
        Extract insights specific to a single strategy.

        Analyzes:
        - Optimal market conditions
        - Optimal feature ranges (where win rate is highest)
        - Parameter sensitivity observations
        - Best/worst time-of-day patterns
        """
        strategy_trades = [t for t in trades if t.strategy_name == strategy_name]

        if len(strategy_trades) < MIN_CLUSTER_SIZE:
            return []

        logger.info(
            "KnowledgeExtractor: extracting insights for {} ({} trades)",
            strategy_name, len(strategy_trades),
        )

        entries: list[KnowledgeEntry] = []
        winners = [t for t in strategy_trades if self._is_winner(t)]
        losers = [t for t in strategy_trades if not self._is_winner(t)]

        total = len(strategy_trades)
        win_rate = len(winners) / total if total > 0 else 0

        # Insight 1: Optimal market conditions
        condition_insights = self._analyze_conditions(strategy_name, strategy_trades)
        entries.extend(condition_insights)

        # Insight 2: Optimal feature ranges
        if len(winners) >= MIN_CLUSTER_SIZE:
            win_ranges = self._find_common_feature_ranges(winners)
            loss_ranges = self._find_common_feature_ranges(losers) if losers else {}

            for feat in win_ranges:
                w_low, w_high, w_med = win_ranges[feat]
                if feat in loss_ranges:
                    l_low, l_high, l_med = loss_ranges[feat]

                    # Check if win range and loss range are meaningfully different
                    if abs(w_med - l_med) > 0.1 * max(abs(w_med), abs(l_med), 1.0):
                        content = (
                            f"戦略インサイト: {strategy_name} - {feat} "
                            f"勝ち中央値={w_med:.2f} vs 負け中央値={l_med:.2f} "
                            f"(勝ちレンジ: {w_low:.2f}~{w_high:.2f})"
                        )
                        entries.append(KnowledgeEntry(
                            category="strategy_insight",
                            content=content,
                            supporting_trades=[t.id for t in strategy_trades[:10]],
                            confidence=min(0.8, len(strategy_trades) / 30.0),
                            auto_generated=True,
                        ))

        # Insight 3: Time-of-day analysis
        time_insight = self._analyze_time_of_day(strategy_name, strategy_trades)
        if time_insight:
            entries.append(time_insight)

        # Insight 4: Holding time analysis
        holding_insight = self._analyze_holding_time(strategy_name, winners, losers)
        if holding_insight:
            entries.append(holding_insight)

        logger.info(
            "KnowledgeExtractor: extracted {} insights for {}",
            len(entries), strategy_name,
        )
        return entries

    # ------------------------------------------------------------------
    # Improvement candidates
    # ------------------------------------------------------------------

    def generate_improvement_candidates(
        self, insights: list[KnowledgeEntry],
    ) -> list[CandidateUpdate]:
        """
        Based on extracted knowledge, suggest concrete parameter changes.

        All candidates are created with status="pending" -- NEVER auto-applied.

        Types of suggestions:
        - Condition refinements (add/tighten entry conditions)
        - Parameter adjustments (RSI thresholds, ATR multipliers)
        - Strategy activation/deactivation recommendations
        """
        candidates: list[CandidateUpdate] = []

        for insight in insights:
            if insight.confidence < 0.4:
                continue  # Not confident enough

            proposed = self._insight_to_candidate(insight)
            if proposed:
                candidates.append(proposed)

        logger.info(
            "KnowledgeExtractor: generated {} improvement candidates from {} insights",
            len(candidates), len(insights),
        )
        return candidates

    def daily_knowledge_update(
        self,
        trades: list[TradeResult],
        target_date: Optional[date] = None,
    ) -> dict[str, Any]:
        """
        Run the full knowledge extraction pipeline for a day's trades.

        1. Separate winners and losers
        2. Extract win/loss patterns
        3. Extract per-strategy insights
        4. Generate improvement candidates
        5. Return summary

        Args:
            trades: All trades for the day.
            target_date: The date to process (defaults to today).

        Returns:
            Summary dict with all extracted knowledge and candidates.
        """
        d = target_date or date.today()
        logger.info(
            "KnowledgeExtractor: running daily knowledge update for {} ({} trades)",
            d, len(trades),
        )

        if not trades:
            logger.info("KnowledgeExtractor: no trades for {}", d)
            return {
                "date": d.isoformat(),
                "win_patterns": [],
                "loss_patterns": [],
                "strategy_insights": [],
                "improvement_candidates": [],
                "summary": "取引なし",
            }

        # Separate winners and losers
        winners = [t for t in trades if self._is_winner(t)]
        losers = [t for t in trades if not self._is_winner(t)]

        # Extract patterns
        win_patterns = self.extract_win_patterns(winners)
        loss_patterns = self.extract_loss_patterns(losers)

        # Per-strategy insights
        strategy_names = set(t.strategy_name for t in trades)
        all_insights: list[KnowledgeEntry] = []
        for name in strategy_names:
            insights = self.extract_strategy_insights(name, trades)
            all_insights.extend(insights)

        # Generate improvement candidates
        all_knowledge = win_patterns + loss_patterns + all_insights
        candidates = self.generate_improvement_candidates(all_knowledge)

        # Summary
        win_rate = len(winners) / len(trades) if trades else 0
        total_pnl = sum(
            (t.exit_price - t.entry_price if t.direction == "long"
             else t.entry_price - t.exit_price) * 100
            for t in trades
        )

        summary = (
            f"{d}: {len(trades)}取引, 勝率{win_rate:.1%}, "
            f"損益{total_pnl:+,.0f}円, "
            f"勝ちパターン{len(win_patterns)}件, "
            f"負けパターン{len(loss_patterns)}件, "
            f"改善候補{len(candidates)}件"
        )

        logger.info("KnowledgeExtractor: {}", summary)

        return {
            "date": d.isoformat(),
            "win_patterns": [e.model_dump() for e in win_patterns],
            "loss_patterns": [e.model_dump() for e in loss_patterns],
            "strategy_insights": [e.model_dump() for e in all_insights],
            "improvement_candidates": [c.model_dump() for c in candidates],
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_winner(trade: TradeResult) -> bool:
        """Check if a trade is a winner."""
        if trade.direction == "long":
            return trade.exit_price > trade.entry_price
        else:
            return trade.exit_price < trade.entry_price

    @staticmethod
    def _find_common_feature_ranges(
        trades: list[TradeResult],
    ) -> dict[str, tuple[float, float, float]]:
        """
        Find common feature value ranges across trades.

        Returns dict of feature_name -> (low, high, median) for the
        interquartile range (25th-75th percentile).
        """
        feature_values: dict[str, list[float]] = defaultdict(list)

        for trade in trades:
            features = trade.features_at_entry
            for key in FEATURE_KEYS:
                if key in features:
                    val = features[key]
                    if isinstance(val, (int, float)) and not isinstance(val, bool):
                        feature_values[key].append(float(val))

        ranges: dict[str, tuple[float, float, float]] = {}
        for feat, values in feature_values.items():
            if len(values) < MIN_CLUSTER_SIZE:
                continue

            sorted_vals = sorted(values)
            n = len(sorted_vals)
            q1_idx = n // 4
            q3_idx = (3 * n) // 4
            median_idx = n // 2

            ranges[feat] = (
                sorted_vals[q1_idx],        # Q1 (low)
                sorted_vals[q3_idx],        # Q3 (high)
                sorted_vals[median_idx],    # Median
            )

        return ranges

    @staticmethod
    def _count_market_conditions(
        trades: list[TradeResult],
    ) -> dict[str, int]:
        """Count market condition occurrences across trades."""
        counts: dict[str, int] = defaultdict(int)
        for trade in trades:
            condition = trade.market_condition or "unknown"
            counts[condition] += 1
        return dict(counts)

    def _find_cross_strategy_patterns(
        self,
        trades: list[TradeResult],
        pattern_type: str,
    ) -> list[KnowledgeEntry]:
        """Find patterns that span across strategies."""
        entries: list[KnowledgeEntry] = []

        if len(trades) < MIN_CLUSTER_SIZE:
            return entries

        # Check if certain market conditions dominate
        condition_counts = self._count_market_conditions(trades)
        total = len(trades)

        for condition, count in condition_counts.items():
            ratio = count / total
            if ratio >= 0.6 and count >= MIN_CLUSTER_SIZE:
                if pattern_type == "win":
                    content = (
                        f"マーケットインサイト: {condition} market では "
                        f"全体勝率が高い ({count}/{total} trades, {ratio:.0%})"
                    )
                    category = "market_insight"
                else:
                    content = (
                        f"マーケットインサイト: {condition} market では "
                        f"全体負け率が高い ({count}/{total} trades, {ratio:.0%})"
                    )
                    category = "market_insight"

                entries.append(KnowledgeEntry(
                    category=category,
                    content=content,
                    supporting_trades=[t.id for t in trades[:10]],
                    confidence=min(0.7, ratio),
                    auto_generated=True,
                ))

        return entries

    def _analyze_conditions(
        self,
        strategy_name: str,
        trades: list[TradeResult],
    ) -> list[KnowledgeEntry]:
        """Analyze which market conditions produce best/worst results."""
        entries: list[KnowledgeEntry] = []

        by_condition: dict[str, list[TradeResult]] = defaultdict(list)
        for trade in trades:
            condition = trade.market_condition or "unknown"
            by_condition[condition].append(trade)

        for condition, cond_trades in by_condition.items():
            if len(cond_trades) < MIN_CLUSTER_SIZE:
                continue

            wins = sum(1 for t in cond_trades if self._is_winner(t))
            wr = wins / len(cond_trades)

            if wr >= WIN_RATE_THRESHOLD:
                content = (
                    f"戦略インサイト: {strategy_name} は {condition} market で "
                    f"勝率{wr:.0%} ({wins}/{len(cond_trades)} trades) -- 好条件"
                )
                entries.append(KnowledgeEntry(
                    category="strategy_insight",
                    content=content,
                    supporting_trades=[t.id for t in cond_trades[:10]],
                    confidence=min(0.9, wr),
                    auto_generated=True,
                ))
            elif (1 - wr) >= LOSS_RATE_THRESHOLD:
                content = (
                    f"戦略インサイト: {strategy_name} は {condition} market で "
                    f"勝率{wr:.0%} ({wins}/{len(cond_trades)} trades) -- 要注意"
                )
                entries.append(KnowledgeEntry(
                    category="strategy_insight",
                    content=content,
                    supporting_trades=[t.id for t in cond_trades[:10]],
                    confidence=min(0.9, 1 - wr),
                    auto_generated=True,
                ))

        return entries

    def _analyze_time_of_day(
        self,
        strategy_name: str,
        trades: list[TradeResult],
    ) -> Optional[KnowledgeEntry]:
        """Analyze if entry time affects win rate."""
        # Bucket entries by hour
        by_hour: dict[int, list[TradeResult]] = defaultdict(list)
        for trade in trades:
            if trade.entry_time:
                hour = trade.entry_time.hour
                by_hour[hour].append(trade)

        best_hour = None
        best_wr = 0.0
        best_count = 0

        for hour, hour_trades in by_hour.items():
            if len(hour_trades) < MIN_CLUSTER_SIZE:
                continue
            wins = sum(1 for t in hour_trades if self._is_winner(t))
            wr = wins / len(hour_trades)
            if wr > best_wr:
                best_wr = wr
                best_hour = hour
                best_count = len(hour_trades)

        if best_hour is not None and best_wr >= WIN_RATE_THRESHOLD:
            content = (
                f"戦略インサイト: {strategy_name} は {best_hour}時台エントリーで "
                f"勝率{best_wr:.0%} ({best_count} trades)"
            )
            return KnowledgeEntry(
                category="strategy_insight",
                content=content,
                supporting_trades=[],
                confidence=min(0.7, best_wr),
                auto_generated=True,
            )
        return None

    @staticmethod
    def _analyze_holding_time(
        strategy_name: str,
        winners: list[TradeResult],
        losers: list[TradeResult],
    ) -> Optional[KnowledgeEntry]:
        """Compare holding times between winners and losers."""
        win_times = [t.holding_minutes for t in winners if t.holding_minutes > 0]
        loss_times = [t.holding_minutes for t in losers if t.holding_minutes > 0]

        if not win_times or not loss_times:
            return None

        avg_win_time = statistics.mean(win_times)
        avg_loss_time = statistics.mean(loss_times)

        if abs(avg_win_time - avg_loss_time) > 30:  # 30min difference is meaningful
            content = (
                f"戦略インサイト: {strategy_name} - "
                f"勝ちトレード平均保有{avg_win_time:.0f}分 vs "
                f"負けトレード平均保有{avg_loss_time:.0f}分"
            )

            if avg_loss_time > avg_win_time * 1.5:
                content += " -> 負けトレードの損切りを早めることを検討"

            return KnowledgeEntry(
                category="strategy_insight",
                content=content,
                supporting_trades=[],
                confidence=0.6,
                auto_generated=True,
            )
        return None

    @staticmethod
    def _insight_to_candidate(insight: KnowledgeEntry) -> Optional[CandidateUpdate]:
        """
        Convert a knowledge insight into a concrete CandidateUpdate.

        Returns None if no actionable change can be derived.
        """
        content = insight.content

        # Parse strategy name from content
        strategy_name = "unknown"
        if "戦略インサイト: " in content:
            parts = content.split("戦略インサイト: ", 1)
            if len(parts) > 1:
                name_part = parts[1].split(" は ", 1)[0]
                strategy_name = name_part.strip()
        elif "勝ちパターン: " in content:
            parts = content.split("勝ちパターン: ", 1)
            if len(parts) > 1:
                name_part = parts[1].split(" は ", 1)[0]
                strategy_name = name_part.strip()
        elif "負けパターン: " in content:
            parts = content.split("負けパターン: ", 1)
            if len(parts) > 1:
                name_part = parts[1].split(" は ", 1)[0]
                strategy_name = name_part.strip()

        if strategy_name == "unknown":
            return None

        # Determine proposed changes based on insight category
        proposed_changes: dict[str, Any] = {}
        reason = content
        expected_improvement = ""

        if insight.category == "win_pattern":
            proposed_changes = {
                "action": "add_preferred_conditions",
                "description": "勝ちパターンに基づく条件追加",
                "source_insight": content,
            }
            expected_improvement = "勝ちパターン条件でのエントリーを優先"

        elif insight.category == "loss_pattern":
            proposed_changes = {
                "action": "add_avoid_conditions",
                "description": "負けパターンに基づく回避条件追加",
                "source_insight": content,
            }
            expected_improvement = "負けパターン条件でのエントリーを回避"

        elif insight.category == "strategy_insight":
            if "損切りを早める" in content:
                proposed_changes = {
                    "action": "adjust_stop_loss",
                    "description": "損切り時間の短縮",
                    "source_insight": content,
                }
                expected_improvement = "負けトレードの損失削減"
            elif "好条件" in content:
                proposed_changes = {
                    "action": "boost_confidence_for_condition",
                    "description": "好条件でのconfidenceブースト",
                    "source_insight": content,
                }
                expected_improvement = "好条件での取引頻度向上"
            elif "要注意" in content:
                proposed_changes = {
                    "action": "reduce_confidence_for_condition",
                    "description": "不利条件でのconfidence低下",
                    "source_insight": content,
                }
                expected_improvement = "不利条件での損失回避"
            else:
                proposed_changes = {
                    "action": "review_parameters",
                    "description": "パラメータ見直し",
                    "source_insight": content,
                }
                expected_improvement = "戦略パラメータの最適化"

        else:
            return None

        return CandidateUpdate(
            strategy_name=strategy_name,
            proposed_changes=proposed_changes,
            reason=reason,
            expected_improvement=expected_improvement,
            status="pending",  # NEVER auto-apply
        )
