"""
Learning loop orchestrator for KabuAI day trading.

Coordinates the full end-of-day learning pipeline:
1. Fetch and analyze all trades
2. Update strategy performances
3. Extract knowledge patterns
4. Generate improvement candidates (always pending)
5. Generate daily report
6. NEVER auto-apply changes

Human review is required for all CandidateUpdate items.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional, Protocol

from loguru import logger

from core.config import settings
from core.models import (
    CandidateUpdate,
    DailyReport,
    KnowledgeEntry,
    StrategyConfig,
    StrategyPerformance,
    TradeResult,
)
from analytics.daily_report import DailyReportGenerator
from analytics.knowledge_extractor import KnowledgeExtractor
from analytics.trade_analyzer import TradeAnalyzer


class DatabaseProtocol(Protocol):
    """Minimal interface for the database manager."""

    async def get_trades_for_date(self, d: date) -> list[TradeResult]: ...
    async def get_all_trades(self, strategy_name: str, limit: int) -> list[TradeResult]: ...
    async def save_strategy_performance(self, perf: StrategyPerformance) -> None: ...
    async def save_knowledge_entry(self, entry: KnowledgeEntry) -> None: ...
    async def save_candidate_update(self, candidate: CandidateUpdate) -> None: ...
    async def save_daily_report(self, report: DailyReport) -> None: ...
    async def get_candidate_update(self, candidate_id: str) -> Optional[CandidateUpdate]: ...
    async def update_candidate_status(
        self, candidate_id: str, status: str, reviewed_at: datetime,
    ) -> None: ...
    async def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]: ...
    async def update_strategy_config(self, config: StrategyConfig) -> None: ...


class LearningLoop:
    """
    Orchestrates the daily learning and improvement cycle.

    The loop runs at end of day and:
    1. Fetches all trades for the day from the database
    2. Analyzes each trade for metrics
    3. Updates aggregated strategy performance records
    4. Extracts knowledge (win/loss patterns, strategy insights)
    5. Generates improvement candidates (CandidateUpdate, status="pending")
    6. Saves candidate_strategy_updates for human review
    7. Generates and saves the daily report
    8. Logs a summary

    CRITICAL: All improvements go to CandidateUpdate with status="pending".
    The system NEVER auto-applies changes. Human review via review_candidate()
    is required.

    Usage::

        loop = LearningLoop(db)
        summary = await loop.run_daily_loop(date.today())

        # Later, human reviews a candidate:
        await loop.review_candidate("abc123", approved=True)
    """

    def __init__(
        self,
        db: DatabaseProtocol,
        analyzer: Optional[TradeAnalyzer] = None,
        extractor: Optional[KnowledgeExtractor] = None,
        report_generator: Optional[DailyReportGenerator] = None,
    ) -> None:
        self.db = db
        self.analyzer = analyzer or TradeAnalyzer()
        self.extractor = extractor or KnowledgeExtractor()
        self.report_generator = report_generator or DailyReportGenerator(
            db=db, analyzer=self.analyzer, extractor=self.extractor,
        )

    # ------------------------------------------------------------------
    # Main daily loop
    # ------------------------------------------------------------------

    async def run_daily_loop(
        self, target_date: Optional[date] = None,
    ) -> dict[str, Any]:
        """
        Run the full daily learning loop.

        Steps:
        1. Fetch all trades for the day
        2. Analyze each trade
        3. Update strategy performances
        4. Extract knowledge (win/loss patterns)
        5. Generate improvement candidates
        6. Save candidate_strategy_updates
        7. Generate daily report
        8. Log summary

        Returns:
            Summary dict with counts and key metrics.
        """
        d = target_date or date.today()
        logger.info("=" * 60)
        logger.info("LearningLoop: starting daily loop for {}", d)
        logger.info("=" * 60)

        # Step 1: Fetch trades
        trades = await self.db.get_trades_for_date(d)
        logger.info("LearningLoop: fetched {} trades for {}", len(trades), d)

        if not trades:
            logger.info("LearningLoop: no trades for {}, generating empty report", d)
            report = await self.report_generator.generate(d, trades=[])
            return {
                "date": d.isoformat(),
                "total_trades": 0,
                "summary": "取引なし",
            }

        # Step 2: Analyze each trade
        trade_analyses: list[dict[str, Any]] = []
        for trade in trades:
            analysis = self.analyzer.analyze_trade(trade)
            trade_analyses.append(analysis)
        logger.info("LearningLoop: analyzed {} trades", len(trade_analyses))

        # Step 3: Update strategy performances
        strategy_names = set(t.strategy_name for t in trades)
        performances: dict[str, StrategyPerformance] = {}

        for strategy_name in strategy_names:
            # Fetch all historical trades for this strategy (up to 200)
            all_strategy_trades = await self.db.get_all_trades(strategy_name, limit=200)
            performance = self.analyzer.analyze_strategy(
                strategy_name, all_strategy_trades, period_days=30,
            )
            performances[strategy_name] = performance

            try:
                await self.db.save_strategy_performance(performance)
                logger.info(
                    "LearningLoop: updated performance for {} "
                    "(WR={:.1%}, PF={:.2f}, {} trades)",
                    strategy_name, performance.win_rate,
                    performance.profit_factor, performance.total_trades,
                )
            except Exception as e:
                logger.error(
                    "LearningLoop: failed to save performance for {}: {}",
                    strategy_name, e,
                )

        # Step 4: Extract knowledge
        knowledge_data = self.extractor.daily_knowledge_update(trades, d)

        win_patterns = knowledge_data.get("win_patterns", [])
        loss_patterns = knowledge_data.get("loss_patterns", [])
        strategy_insights = knowledge_data.get("strategy_insights", [])
        all_knowledge = win_patterns + loss_patterns + strategy_insights

        logger.info(
            "LearningLoop: extracted {} knowledge entries "
            "(win={}, loss={}, insight={})",
            len(all_knowledge), len(win_patterns),
            len(loss_patterns), len(strategy_insights),
        )

        # Step 5: Save knowledge entries to database
        saved_knowledge_count = 0
        for entry_data in all_knowledge:
            try:
                if isinstance(entry_data, dict):
                    entry = KnowledgeEntry(**entry_data)
                else:
                    entry = entry_data
                await self.db.save_knowledge_entry(entry)
                saved_knowledge_count += 1
            except Exception as e:
                logger.warning("LearningLoop: failed to save knowledge entry: {}", e)

        # Step 6: Save improvement candidates (status="pending", NEVER auto-applied)
        candidates_data = knowledge_data.get("improvement_candidates", [])
        saved_candidates: list[CandidateUpdate] = []

        for cand_data in candidates_data:
            try:
                if isinstance(cand_data, dict):
                    candidate = CandidateUpdate(**cand_data)
                else:
                    candidate = cand_data

                # Ensure status is always pending
                candidate.status = "pending"

                await self.db.save_candidate_update(candidate)
                saved_candidates.append(candidate)
                logger.info(
                    "LearningLoop: saved candidate update for {} (id={}): {}",
                    candidate.strategy_name, candidate.id,
                    candidate.reason[:80],
                )
            except Exception as e:
                logger.warning("LearningLoop: failed to save candidate: {}", e)

        # Step 7: Generate daily report
        report = await self.report_generator.generate(d, trades=trades)

        # Step 8: Log summary
        total_pnl = report.total_pnl
        win_rate = report.win_rate

        summary = (
            f"{d}: {len(trades)}取引完了, "
            f"損益{total_pnl:+,.0f}円, 勝率{win_rate:.1%}, "
            f"知見{saved_knowledge_count}件, "
            f"改善候補{len(saved_candidates)}件(全てpending)"
        )

        logger.info("=" * 60)
        logger.info("LearningLoop: {}", summary)
        logger.info("=" * 60)

        # Strategy ranking
        if len(performances) > 1:
            ranking = self.analyzer.compare_strategies(list(performances.values()))
            for i, r in enumerate(ranking, 1):
                logger.info(
                    "  Rank {}: {} (score={:.3f})",
                    i, r["strategy_name"], r["composite_score"],
                )

        return {
            "date": d.isoformat(),
            "total_trades": len(trades),
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "knowledge_entries": saved_knowledge_count,
            "improvement_candidates": len(saved_candidates),
            "strategy_performances": {
                name: {
                    "win_rate": p.win_rate,
                    "profit_factor": p.profit_factor,
                    "total_trades": p.total_trades,
                }
                for name, p in performances.items()
            },
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Candidate review
    # ------------------------------------------------------------------

    async def review_candidate(
        self,
        candidate_id: str,
        approved: bool,
        reason: str = "",
    ) -> dict[str, Any]:
        """
        Review a pending CandidateUpdate.

        If approved:
        - Apply the proposed changes to the strategy config
        - Mark status as "applied"

        If rejected:
        - Mark status as "rejected" with reason

        Args:
            candidate_id: The ID of the CandidateUpdate to review.
            approved: True to approve, False to reject.
            reason: Optional reason for rejection.

        Returns:
            Dict with review result.
        """
        logger.info(
            "LearningLoop: reviewing candidate {} (approved={})",
            candidate_id, approved,
        )

        # Fetch the candidate
        candidate = await self.db.get_candidate_update(candidate_id)
        if candidate is None:
            logger.error("LearningLoop: candidate {} not found", candidate_id)
            return {"error": f"Candidate {candidate_id} not found"}

        if candidate.status != "pending":
            logger.warning(
                "LearningLoop: candidate {} is not pending (status={})",
                candidate_id, candidate.status,
            )
            return {
                "error": f"Candidate {candidate_id} is not pending "
                         f"(current status: {candidate.status})",
            }

        now = datetime.now()

        if approved:
            # Apply changes to strategy config
            try:
                await self._apply_candidate(candidate)
                await self.db.update_candidate_status(
                    candidate_id, "applied", now,
                )
                logger.info(
                    "LearningLoop: candidate {} approved and applied for {}",
                    candidate_id, candidate.strategy_name,
                )
                return {
                    "candidate_id": candidate_id,
                    "status": "applied",
                    "strategy_name": candidate.strategy_name,
                    "changes": candidate.proposed_changes,
                }

            except Exception as e:
                logger.error(
                    "LearningLoop: failed to apply candidate {}: {}",
                    candidate_id, e,
                )
                return {"error": f"Failed to apply: {e}"}

        else:
            # Reject
            await self.db.update_candidate_status(
                candidate_id, "rejected", now,
            )
            logger.info(
                "LearningLoop: candidate {} rejected (reason: {})",
                candidate_id, reason or "no reason given",
            )
            return {
                "candidate_id": candidate_id,
                "status": "rejected",
                "reason": reason,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _apply_candidate(self, candidate: CandidateUpdate) -> None:
        """
        Apply a CandidateUpdate to the strategy's configuration.

        This modifies the strategy config based on the proposed changes.
        """
        config = await self.db.get_strategy_config(candidate.strategy_name)
        if config is None:
            raise ValueError(
                f"Strategy config not found for {candidate.strategy_name}",
            )

        changes = candidate.proposed_changes
        action = changes.get("action", "")

        if action == "add_preferred_conditions":
            # Add conditions that favor this strategy
            if "preferred_conditions" not in config.entry_conditions:
                config.entry_conditions["preferred_conditions"] = []
            config.entry_conditions["preferred_conditions"].append(
                changes.get("source_insight", ""),
            )

        elif action == "add_avoid_conditions":
            # Add conditions to avoid
            if "avoid_conditions" not in config.entry_conditions:
                config.entry_conditions["avoid_conditions"] = []
            config.entry_conditions["avoid_conditions"].append(
                changes.get("source_insight", ""),
            )

        elif action == "adjust_stop_loss":
            # Tighten stop loss
            current_mult = config.parameter_set.get("stop_loss_atr_multiplier", 2.0)
            config.parameter_set["stop_loss_atr_multiplier"] = max(1.0, current_mult * 0.85)

        elif action == "boost_confidence_for_condition":
            # Add condition-based confidence boost
            if "confidence_boosts" not in config.parameter_set:
                config.parameter_set["confidence_boosts"] = {}
            insight = changes.get("source_insight", "")
            # Extract condition from insight
            config.parameter_set["confidence_boosts"][insight[:50]] = 0.1

        elif action == "reduce_confidence_for_condition":
            # Add condition-based confidence penalty
            if "confidence_penalties" not in config.parameter_set:
                config.parameter_set["confidence_penalties"] = {}
            insight = changes.get("source_insight", "")
            config.parameter_set["confidence_penalties"][insight[:50]] = -0.1

        elif action == "review_parameters":
            # Mark config for review (add a note)
            if "review_notes" not in config.parameter_set:
                config.parameter_set["review_notes"] = []
            config.parameter_set["review_notes"].append({
                "date": datetime.now().isoformat(),
                "note": changes.get("source_insight", ""),
            })

        else:
            logger.warning(
                "LearningLoop: unknown action '{}' for candidate {}",
                action, candidate.id,
            )
            return

        # Bump version
        parts = config.version.split(".")
        if len(parts) == 3:
            parts[2] = str(int(parts[2]) + 1)
            config.version = ".".join(parts)

        await self.db.update_strategy_config(config)
        logger.info(
            "LearningLoop: applied {} to {} (v{})",
            action, candidate.strategy_name, config.version,
        )
