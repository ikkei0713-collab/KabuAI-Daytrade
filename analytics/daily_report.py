"""
Daily report generator for KabuAI day trading.

Aggregates all trades, calculates daily P&L, strategy breakdowns,
and bundles knowledge entries and improvement candidates into a
single report saved to both the database and a JSON file.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Protocol

from loguru import logger

from core.config import settings
from core.models import (
    CandidateUpdate,
    DailyReport,
    KnowledgeEntry,
    TradeResult,
)
from analytics.knowledge_extractor import KnowledgeExtractor
from analytics.trade_analyzer import TradeAnalyzer


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

KNOWLEDGE_TRADES_DIR = settings.BASE_DIR / "knowledge" / "trades"


class DatabaseProtocol(Protocol):
    """Minimal interface for the database manager."""

    async def save_daily_report(self, report: DailyReport) -> None: ...
    async def save_knowledge_entry(self, entry: KnowledgeEntry) -> None: ...
    async def save_candidate_update(self, candidate: CandidateUpdate) -> None: ...
    async def get_trades_for_date(self, d: date) -> list[TradeResult]: ...


class DailyReportGenerator:
    """
    Generates comprehensive end-of-day reports.

    Responsibilities:
    - Aggregate all trades for the day
    - Calculate daily P&L and per-strategy breakdown
    - Collect knowledge entries extracted during the day
    - Bundle improvement candidates
    - Save to database and knowledge/trades/{date}.json

    Usage::

        generator = DailyReportGenerator(db, analyzer, extractor)
        report = await generator.generate(date.today())
    """

    def __init__(
        self,
        db: DatabaseProtocol,
        analyzer: Optional[TradeAnalyzer] = None,
        extractor: Optional[KnowledgeExtractor] = None,
    ) -> None:
        self.db = db
        self.analyzer = analyzer or TradeAnalyzer()
        self.extractor = extractor or KnowledgeExtractor()

    # ------------------------------------------------------------------
    # Main report generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        target_date: Optional[date] = None,
        trades: Optional[list[TradeResult]] = None,
    ) -> DailyReport:
        """
        Generate a full daily report.

        Args:
            target_date: Date to generate report for (default: today).
            trades: Pre-fetched trades. If None, fetched from database.

        Returns:
            DailyReport with all aggregated data.
        """
        d = target_date or date.today()
        logger.info("DailyReportGenerator: generating report for {}", d)

        # Fetch trades if not provided
        if trades is None:
            trades = await self.db.get_trades_for_date(d)

        if not trades:
            logger.info("DailyReportGenerator: no trades for {}", d)
            report = DailyReport(date=d)
            await self._save_report(report, d)
            return report

        # Calculate P&L
        total_pnl = 0.0
        for trade in trades:
            if trade.direction == "long":
                pnl = (trade.exit_price - trade.entry_price) * 100  # 100株単位
            else:
                pnl = (trade.entry_price - trade.exit_price) * 100
            total_pnl += pnl

        # Win rate
        winners = [t for t in trades if self._is_winner(t)]
        win_rate = len(winners) / len(trades) if trades else 0.0

        # Best / worst trade
        trade_pnls: list[tuple[str, float]] = []
        for trade in trades:
            if trade.direction == "long":
                pnl = (trade.exit_price - trade.entry_price) * 100
            else:
                pnl = (trade.entry_price - trade.exit_price) * 100
            trade_pnls.append((trade.id, pnl))

        best_trade_id = max(trade_pnls, key=lambda x: x[1])[0] if trade_pnls else None
        worst_trade_id = min(trade_pnls, key=lambda x: x[1])[0] if trade_pnls else None

        # Strategy breakdown
        strategy_summary = self._build_strategy_summary(trades)

        # Knowledge extraction
        knowledge_data = self.extractor.daily_knowledge_update(trades, d)

        # Collect knowledge entries
        knowledge_entries = (
            knowledge_data.get("win_patterns", [])
            + knowledge_data.get("loss_patterns", [])
            + knowledge_data.get("strategy_insights", [])
        )

        # Improvement candidates
        improvement_candidates = knowledge_data.get("improvement_candidates", [])

        # Build report
        report = DailyReport(
            date=d,
            total_pnl=total_pnl,
            total_trades=len(trades),
            win_rate=win_rate,
            best_trade=best_trade_id,
            worst_trade=worst_trade_id,
            strategy_summary=strategy_summary,
            knowledge_entries=knowledge_entries,
            improvement_candidates=improvement_candidates,
        )

        # Save to database and file
        await self._save_report(report, d)

        # Save knowledge entries to database
        for entry_data in knowledge_entries:
            try:
                entry = KnowledgeEntry(**entry_data) if isinstance(entry_data, dict) else entry_data
                await self.db.save_knowledge_entry(entry)
            except Exception as e:
                logger.warning("Failed to save knowledge entry: {}", e)

        # Save candidates to database
        for cand_data in improvement_candidates:
            try:
                cand = CandidateUpdate(**cand_data) if isinstance(cand_data, dict) else cand_data
                await self.db.save_candidate_update(cand)
            except Exception as e:
                logger.warning("Failed to save candidate update: {}", e)

        logger.info(
            "DailyReportGenerator: report for {} - "
            "P&L={:+,.0f}円, trades={}, WR={:.1%}, "
            "knowledge={}, candidates={}",
            d, total_pnl, len(trades), win_rate,
            len(knowledge_entries), len(improvement_candidates),
        )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _save_report(self, report: DailyReport, d: date) -> None:
        """Save report to database and JSON file."""
        # Save to database
        try:
            await self.db.save_daily_report(report)
        except Exception as e:
            logger.error("Failed to save daily report to DB: {}", e)

        # Save to JSON file
        try:
            KNOWLEDGE_TRADES_DIR.mkdir(parents=True, exist_ok=True)
            output_path = KNOWLEDGE_TRADES_DIR / f"{d.isoformat()}.json"

            report_data = report.model_dump(mode="json")
            # Ensure datetime serialization
            report_json = json.dumps(
                report_data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

            output_path.write_text(report_json, encoding="utf-8")
            logger.info("DailyReportGenerator: saved report to {}", output_path)

        except Exception as e:
            logger.error("Failed to save daily report to file: {}", e)

    def _build_strategy_summary(
        self, trades: list[TradeResult],
    ) -> dict[str, Any]:
        """Build per-strategy summary."""
        summary: dict[str, Any] = {}

        for trade in trades:
            name = trade.strategy_name
            if name not in summary:
                summary[name] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0,
                    "best_pnl": float("-inf"),
                    "worst_pnl": float("inf"),
                    "avg_holding_minutes": 0.0,
                    "tickers": [],
                }

            s = summary[name]
            s["trades"] += 1

            if trade.direction == "long":
                pnl = (trade.exit_price - trade.entry_price) * 100
            else:
                pnl = (trade.entry_price - trade.exit_price) * 100

            s["total_pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1
            s["best_pnl"] = max(s["best_pnl"], pnl)
            s["worst_pnl"] = min(s["worst_pnl"], pnl)
            s["avg_holding_minutes"] += trade.holding_minutes

            if trade.ticker not in s["tickers"]:
                s["tickers"].append(trade.ticker)

        # Finalize averages
        for s in summary.values():
            if s["trades"] > 0:
                s["win_rate"] = s["wins"] / s["trades"]
                s["avg_pnl"] = s["total_pnl"] / s["trades"]
                s["avg_holding_minutes"] /= s["trades"]
            else:
                s["win_rate"] = 0.0
                s["avg_pnl"] = 0.0

            # Clean up inf values
            if s["best_pnl"] == float("-inf"):
                s["best_pnl"] = 0.0
            if s["worst_pnl"] == float("inf"):
                s["worst_pnl"] = 0.0

        return summary

    @staticmethod
    def _is_winner(trade: TradeResult) -> bool:
        """Check if a trade is a winner."""
        if trade.direction == "long":
            return trade.exit_price > trade.entry_price
        else:
            return trade.exit_price < trade.entry_price
