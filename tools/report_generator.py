"""Report generation for KabuAI-Daytrade.

Produces formatted markdown daily and weekly reports, exports trade
logs to CSV, and generates per-strategy performance summaries.
"""

from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

from core.models import DailyReport, TradeResult, StrategyPerformance


class ReportGenerator:
    """Generate human-readable and machine-parseable trading reports."""

    # ------------------------------------------------------------------
    # Daily text report
    # ------------------------------------------------------------------

    def generate_daily_text_report(self, daily_report: DailyReport) -> str:
        """Produce a full markdown report for a single trading day.

        Args:
            daily_report: A populated ``DailyReport`` model.

        Returns:
            Markdown-formatted string.
        """
        d = daily_report
        pnl_emoji = "+" if d.total_pnl >= 0 else ""

        lines: list[str] = [
            f"# Daily Report - {d.date.isoformat()}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total P&L | {pnl_emoji}¥{d.total_pnl:,.0f} |",
            f"| Total Trades | {d.total_trades} |",
            f"| Win Rate | {d.win_rate:.1%} |",
            f"| Best Trade | {d.best_trade or 'N/A'} |",
            f"| Worst Trade | {d.worst_trade or 'N/A'} |",
            "",
        ]

        # Strategy breakdown
        if d.strategy_summary:
            lines.append("## Strategy Breakdown")
            lines.append("")
            lines.append("| Strategy | Trades | P&L | Win Rate |")
            lines.append("|----------|--------|-----|----------|")
            for name, stats in d.strategy_summary.items():
                trades = stats.get("trades", 0)
                pnl = stats.get("pnl", 0.0)
                wr = stats.get("win_rate", 0.0)
                sign = "+" if pnl >= 0 else ""
                lines.append(f"| {name} | {trades} | {sign}¥{pnl:,.0f} | {wr:.1%} |")
            lines.append("")

        # Knowledge entries
        if d.knowledge_entries:
            lines.append("## Learnings")
            lines.append("")
            for entry in d.knowledge_entries:
                if isinstance(entry, dict):
                    cat = entry.get("category", "insight")
                    content = entry.get("content", str(entry))
                    lines.append(f"- **[{cat}]** {content}")
                else:
                    lines.append(f"- {entry}")
            lines.append("")

        # Improvement candidates
        if d.improvement_candidates:
            lines.append("## Proposed Improvements")
            lines.append("")
            for candidate in d.improvement_candidates:
                if isinstance(candidate, dict):
                    strat = candidate.get("strategy_name", "unknown")
                    reason = candidate.get("reason", str(candidate))
                    lines.append(f"- **{strat}**: {reason}")
                else:
                    lines.append(f"- {candidate}")
            lines.append("")

        # Footer
        lines.append("---")
        lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} JST*")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Weekly summary
    # ------------------------------------------------------------------

    def generate_weekly_summary(self, reports: list[DailyReport]) -> str:
        """Aggregate multiple daily reports into a weekly summary.

        Args:
            reports: List of ``DailyReport`` objects for the week.

        Returns:
            Markdown-formatted weekly summary string.
        """
        if not reports:
            return "# Weekly Summary\n\nNo trading data available.\n"

        reports_sorted = sorted(reports, key=lambda r: r.date)
        start = reports_sorted[0].date.isoformat()
        end = reports_sorted[-1].date.isoformat()

        total_pnl = sum(r.total_pnl for r in reports)
        total_trades = sum(r.total_trades for r in reports)
        trading_days = len(reports)
        winning_days = sum(1 for r in reports if r.total_pnl > 0)
        losing_days = sum(1 for r in reports if r.total_pnl < 0)
        flat_days = trading_days - winning_days - losing_days

        # Aggregate win rate: weighted by number of trades
        weighted_wr_num = sum(r.win_rate * r.total_trades for r in reports)
        avg_win_rate = weighted_wr_num / total_trades if total_trades > 0 else 0.0

        # Best and worst days
        best_day = max(reports, key=lambda r: r.total_pnl)
        worst_day = min(reports, key=lambda r: r.total_pnl)

        # Strategy aggregation
        strategy_totals: dict[str, dict[str, float]] = {}
        for r in reports:
            for name, stats in r.strategy_summary.items():
                if name not in strategy_totals:
                    strategy_totals[name] = {"trades": 0, "pnl": 0.0, "wins": 0}
                strategy_totals[name]["trades"] += stats.get("trades", 0)
                strategy_totals[name]["pnl"] += stats.get("pnl", 0.0)
                st_trades = stats.get("trades", 0)
                st_wr = stats.get("win_rate", 0.0)
                strategy_totals[name]["wins"] += st_wr * st_trades

        pnl_sign = "+" if total_pnl >= 0 else ""

        lines: list[str] = [
            f"# Weekly Summary: {start} ~ {end}",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Period | {start} ~ {end} |",
            f"| Trading Days | {trading_days} |",
            f"| Total Trades | {total_trades} |",
            f"| Total P&L | {pnl_sign}¥{total_pnl:,.0f} |",
            f"| Avg P&L / Day | {pnl_sign}¥{total_pnl / trading_days:,.0f} |" if trading_days > 0 else "| Avg P&L / Day | N/A |",
            f"| Win Rate | {avg_win_rate:.1%} |",
            f"| Winning Days | {winning_days} |",
            f"| Losing Days | {losing_days} |",
            f"| Flat Days | {flat_days} |",
            f"| Best Day | {best_day.date.isoformat()} ({'+' if best_day.total_pnl >= 0 else ''}¥{best_day.total_pnl:,.0f}) |",
            f"| Worst Day | {worst_day.date.isoformat()} ({'+' if worst_day.total_pnl >= 0 else ''}¥{worst_day.total_pnl:,.0f}) |",
            "",
        ]

        # Per-day table
        lines.append("## Daily Breakdown")
        lines.append("")
        lines.append("| Date | Trades | P&L | Win Rate |")
        lines.append("|------|--------|-----|----------|")
        for r in reports_sorted:
            sign = "+" if r.total_pnl >= 0 else ""
            lines.append(
                f"| {r.date.isoformat()} | {r.total_trades} | {sign}¥{r.total_pnl:,.0f} | {r.win_rate:.1%} |"
            )
        lines.append("")

        # Strategy summary
        if strategy_totals:
            lines.append("## Strategy Summary")
            lines.append("")
            lines.append("| Strategy | Trades | P&L | Win Rate |")
            lines.append("|----------|--------|-----|----------|")
            for name, st in sorted(strategy_totals.items(), key=lambda x: x[1]["pnl"], reverse=True):
                st_wr = st["wins"] / st["trades"] if st["trades"] > 0 else 0.0
                sign = "+" if st["pnl"] >= 0 else ""
                lines.append(
                    f"| {name} | {int(st['trades'])} | {sign}¥{st['pnl']:,.0f} | {st_wr:.1%} |"
                )
            lines.append("")

        # Accumulated learnings
        all_knowledge: list[str] = []
        for r in reports:
            for entry in r.knowledge_entries:
                if isinstance(entry, dict):
                    all_knowledge.append(entry.get("content", str(entry)))
                else:
                    all_knowledge.append(str(entry))

        if all_knowledge:
            lines.append("## Key Learnings This Week")
            lines.append("")
            for k in all_knowledge:
                lines.append(f"- {k}")
            lines.append("")

        lines.append("---")
        lines.append(f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} JST*")
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_trades_csv(
        self,
        trades: Sequence[TradeResult | dict[str, Any]],
        filepath: str | Path,
    ) -> Path:
        """Export a list of trades to a CSV file.

        Args:
            trades: Sequence of ``TradeResult`` models or plain dicts.
            filepath: Destination file path.

        Returns:
            The ``Path`` object of the written file.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "id",
            "ticker",
            "strategy_name",
            "direction",
            "entry_price",
            "exit_price",
            "entry_time",
            "exit_time",
            "pnl",
            "pnl_pct",
            "holding_minutes",
            "entry_reason",
            "exit_reason",
            "market_condition",
        ]

        rows: list[dict[str, Any]] = []
        for t in trades:
            if isinstance(t, dict):
                rows.append(t)
            else:
                rows.append(t.model_dump())

        with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Exported {} trades to {}", len(rows), filepath)
        return filepath

    # ------------------------------------------------------------------
    # Strategy report
    # ------------------------------------------------------------------

    def generate_strategy_report(
        self,
        strategy_name: str,
        performance: StrategyPerformance | dict[str, Any],
    ) -> str:
        """Generate a markdown report for a single strategy's performance.

        Args:
            strategy_name: Name of the strategy.
            performance: ``StrategyPerformance`` model or equivalent dict.

        Returns:
            Markdown-formatted string.
        """
        if isinstance(performance, dict):
            p = performance
        else:
            p = performance.model_dump()

        total = p.get("total_trades", 0)
        wins = p.get("wins", 0)
        losses = p.get("losses", 0)
        wr = p.get("win_rate", 0.0)
        pf = p.get("profit_factor", 0.0)
        avg_pnl = p.get("avg_pnl", 0.0)
        avg_hold = p.get("avg_holding_minutes", 0.0)
        sharpe = p.get("sharpe_ratio", 0.0)
        mdd = p.get("max_drawdown", 0.0)

        sign = "+" if avg_pnl >= 0 else ""

        lines: list[str] = [
            f"# Strategy Report: {strategy_name}",
            "",
            "## Performance Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {total} |",
            f"| Wins / Losses | {wins} / {losses} |",
            f"| Win Rate | {wr:.1%} |",
            f"| Profit Factor | {pf:.2f} |",
            f"| Avg P&L / Trade | {sign}¥{avg_pnl:,.0f} |",
            f"| Avg Holding Time | {avg_hold:.0f} min |",
            f"| Sharpe Ratio | {sharpe:.2f} |",
            f"| Max Drawdown | ¥{mdd:,.0f} |",
            "",
        ]

        # Performance by market condition
        by_condition = p.get("performance_by_condition", {})
        if by_condition:
            lines.append("## Performance by Market Condition")
            lines.append("")
            lines.append("| Condition | Trades | Win Rate | Avg P&L |")
            lines.append("|-----------|--------|----------|---------|")
            for cond, stats in by_condition.items():
                if isinstance(stats, dict):
                    ct = stats.get("trades", 0)
                    cwr = stats.get("win_rate", 0.0)
                    cpnl = stats.get("avg_pnl", 0.0)
                    cs = "+" if cpnl >= 0 else ""
                    lines.append(f"| {cond} | {ct} | {cwr:.1%} | {cs}¥{cpnl:,.0f} |")
            lines.append("")

        # Performance by event type
        by_event = p.get("performance_by_event", {})
        if by_event:
            lines.append("## Performance by Event Type")
            lines.append("")
            lines.append("| Event | Trades | Win Rate | Avg P&L |")
            lines.append("|-------|--------|----------|---------|")
            for evt, stats in by_event.items():
                if isinstance(stats, dict):
                    et = stats.get("trades", 0)
                    ewr = stats.get("win_rate", 0.0)
                    epnl = stats.get("avg_pnl", 0.0)
                    es = "+" if epnl >= 0 else ""
                    lines.append(f"| {evt} | {et} | {ewr:.1%} | {es}¥{epnl:,.0f} |")
            lines.append("")

        lines.append("---")
        lines.append(f"*Last updated: {p.get('last_updated', datetime.now())}*")
        lines.append("")

        return "\n".join(lines)
