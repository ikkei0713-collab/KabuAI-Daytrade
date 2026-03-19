"""SQLite database manager using aiosqlite.

Provides ``DatabaseManager`` — the single gateway for all persistence
operations in the system.  Tables are created lazily on first call to
``init_db()``.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from core.config import settings
from core.models import (
    CandidateUpdate,
    DailyReport,
    KnowledgeEntry,
    Order,
    Position,
    StrategyPerformance,
    TradeResult,
    TradeSignal,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# SQL DDL
# ──────────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id              TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    strategy_name   TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_price     REAL NOT NULL,
    exit_price      REAL NOT NULL,
    entry_time      TEXT NOT NULL,
    exit_time       TEXT NOT NULL,
    pnl             REAL NOT NULL DEFAULT 0,
    pnl_pct         REAL NOT NULL DEFAULT 0,
    holding_minutes INTEGER NOT NULL DEFAULT 0,
    entry_reason    TEXT NOT NULL DEFAULT '',
    exit_reason     TEXT NOT NULL DEFAULT '',
    features_at_entry TEXT NOT NULL DEFAULT '{}',
    features_at_exit  TEXT NOT NULL DEFAULT '{}',
    market_condition  TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

CREATE TABLE IF NOT EXISTS positions (
    id              TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    strategy_name   TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    entry_price     REAL NOT NULL,
    entry_time      TEXT NOT NULL,
    current_price   REAL NOT NULL DEFAULT 0,
    unrealized_pnl  REAL NOT NULL DEFAULT 0,
    holding_minutes INTEGER NOT NULL DEFAULT 0,
    stop_loss       REAL NOT NULL DEFAULT 0,
    take_profit     REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_positions_ticker ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_name);

CREATE TABLE IF NOT EXISTS orders (
    id              TEXT PRIMARY KEY,
    ticker          TEXT NOT NULL,
    direction       TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    order_type      TEXT NOT NULL DEFAULT 'market',
    price           REAL NOT NULL DEFAULT 0,
    quantity        INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'pending',
    strategy_name   TEXT NOT NULL DEFAULT '',
    timestamp       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_ticker ON orders(ticker);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_strategy ON orders(strategy_name);

CREATE TABLE IF NOT EXISTS strategy_performance (
    strategy_name           TEXT PRIMARY KEY,
    total_trades            INTEGER NOT NULL DEFAULT 0,
    wins                    INTEGER NOT NULL DEFAULT 0,
    losses                  INTEGER NOT NULL DEFAULT 0,
    win_rate                REAL NOT NULL DEFAULT 0,
    profit_factor           REAL NOT NULL DEFAULT 0,
    avg_pnl                 REAL NOT NULL DEFAULT 0,
    avg_holding_minutes     REAL NOT NULL DEFAULT 0,
    sharpe_ratio            REAL NOT NULL DEFAULT 0,
    max_drawdown            REAL NOT NULL DEFAULT 0,
    performance_by_condition TEXT NOT NULL DEFAULT '{}',
    performance_by_event     TEXT NOT NULL DEFAULT '{}',
    last_updated            TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge (
    id              TEXT PRIMARY KEY,
    date            TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN (
                        'win_pattern', 'loss_pattern',
                        'strategy_insight', 'market_insight'
                    )),
    content         TEXT NOT NULL,
    supporting_trades TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0.5,
    auto_generated  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_knowledge_date ON knowledge(date);
CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge(category);

CREATE TABLE IF NOT EXISTS candidate_updates (
    id                   TEXT PRIMARY KEY,
    strategy_name        TEXT NOT NULL,
    proposed_changes     TEXT NOT NULL DEFAULT '{}',
    reason               TEXT NOT NULL DEFAULT '',
    expected_improvement TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending','approved','rejected','applied')),
    created_at           TEXT NOT NULL,
    reviewed_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_candidate_status ON candidate_updates(status);
CREATE INDEX IF NOT EXISTS idx_candidate_strategy ON candidate_updates(strategy_name);

CREATE TABLE IF NOT EXISTS feature_snapshots (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,
    features    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_features_ticker_date ON feature_snapshots(ticker, date);

CREATE TABLE IF NOT EXISTS daily_reports (
    date                   TEXT PRIMARY KEY,
    total_pnl              REAL NOT NULL DEFAULT 0,
    total_trades           INTEGER NOT NULL DEFAULT 0,
    win_rate               REAL NOT NULL DEFAULT 0,
    best_trade             TEXT,
    worst_trade            TEXT,
    strategy_summary       TEXT NOT NULL DEFAULT '{}',
    knowledge_entries      TEXT NOT NULL DEFAULT '[]',
    improvement_candidates TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS skipped_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_name   TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    confidence      REAL NOT NULL,
    timestamp       TEXT NOT NULL,
    reason          TEXT NOT NULL,
    features_snapshot TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_skipped_strategy ON skipped_signals(strategy_name);
CREATE INDEX IF NOT EXISTS idx_skipped_timestamp ON skipped_signals(timestamp);

CREATE TABLE IF NOT EXISTS ticker_affinity (
    strategy_name   TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    trades          INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    avg_pnl         REAL NOT NULL DEFAULT 0,
    last_updated    TEXT NOT NULL,
    PRIMARY KEY (strategy_name, ticker)
);

CREATE INDEX IF NOT EXISTS idx_affinity_strategy ON ticker_affinity(strategy_name);
CREATE INDEX IF NOT EXISTS idx_affinity_ticker ON ticker_affinity(ticker);
"""


def _dt_to_str(dt: datetime | date) -> str:
    """Serialise a datetime/date to ISO-8601 string."""
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class DatabaseManager:
    """Async SQLite gateway for KabuAI-Daytrade."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path or settings.DB_PATH)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init_db(self) -> None:
        """Create all tables and indexes if they do not exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(_SCHEMA_SQL)
            await db.commit()
        logger.info("Database initialised at %s", self.db_path)

    async def _connect(self) -> aiosqlite.Connection:
        return await aiosqlite.connect(self.db_path)

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------

    async def save_trade(self, trade: TradeResult) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO trades
                    (id, ticker, strategy_name, direction, entry_price,
                     exit_price, entry_time, exit_time, pnl, pnl_pct,
                     holding_minutes, entry_reason, exit_reason,
                     features_at_entry, features_at_exit, market_condition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.id,
                    trade.ticker,
                    trade.strategy_name,
                    trade.direction,
                    trade.entry_price,
                    trade.exit_price,
                    _dt_to_str(trade.entry_time),
                    _dt_to_str(trade.exit_time),
                    trade.pnl,
                    trade.pnl_pct,
                    trade.holding_minutes,
                    trade.entry_reason,
                    trade.exit_reason,
                    json.dumps(trade.features_at_entry, ensure_ascii=False),
                    json.dumps(trade.features_at_exit, ensure_ascii=False),
                    trade.market_condition,
                ),
            )
            await db.commit()

    async def get_trades(
        self,
        strategy_name: str | None = None,
        ticker: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        limit: int = 500,
    ) -> list[TradeResult]:
        clauses: list[str] = []
        params: list[Any] = []

        if strategy_name is not None:
            clauses.append("strategy_name = ?")
            params.append(strategy_name)
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if start_date is not None:
            clauses.append("entry_time >= ?")
            params.append(_dt_to_str(datetime.combine(start_date, datetime.min.time())))
        if end_date is not None:
            clauses.append("entry_time <= ?")
            params.append(_dt_to_str(datetime.combine(end_date, datetime.max.time())))

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM trades{where} ORDER BY entry_time DESC LIMIT ?"
        params.append(limit)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        return [
            TradeResult(
                id=row["id"],
                ticker=row["ticker"],
                strategy_name=row["strategy_name"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                exit_price=row["exit_price"],
                entry_time=_str_to_dt(row["entry_time"]),
                exit_time=_str_to_dt(row["exit_time"]),
                pnl=row["pnl"],
                pnl_pct=row["pnl_pct"],
                holding_minutes=row["holding_minutes"],
                entry_reason=row["entry_reason"],
                exit_reason=row["exit_reason"],
                features_at_entry=json.loads(row["features_at_entry"]),
                features_at_exit=json.loads(row["features_at_exit"]),
                market_condition=row["market_condition"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    async def save_position(self, pos: Position) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO positions
                    (id, ticker, strategy_name, direction, entry_price,
                     entry_time, current_price, unrealized_pnl,
                     holding_minutes, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pos.id,
                    pos.ticker,
                    pos.strategy_name,
                    pos.direction,
                    pos.entry_price,
                    _dt_to_str(pos.entry_time),
                    pos.current_price,
                    pos.unrealized_pnl,
                    pos.holding_minutes,
                    pos.stop_loss,
                    pos.take_profit,
                ),
            )
            await db.commit()

    async def delete_position(self, position_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM positions WHERE id = ?", (position_id,))
            await db.commit()

    async def get_open_positions(self) -> list[Position]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM positions") as cursor:
                rows = await cursor.fetchall()

        return [
            Position(
                id=row["id"],
                ticker=row["ticker"],
                strategy_name=row["strategy_name"],
                direction=row["direction"],
                entry_price=row["entry_price"],
                entry_time=_str_to_dt(row["entry_time"]),
                current_price=row["current_price"],
                unrealized_pnl=row["unrealized_pnl"],
                holding_minutes=row["holding_minutes"],
                stop_loss=row["stop_loss"],
                take_profit=row["take_profit"],
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    async def save_order(self, order: Order) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO orders
                    (id, ticker, direction, order_type, price, quantity,
                     status, strategy_name, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.id,
                    order.ticker,
                    order.direction,
                    order.order_type,
                    order.price,
                    order.quantity,
                    order.status,
                    order.strategy_name,
                    _dt_to_str(order.timestamp),
                ),
            )
            await db.commit()

    async def get_pending_orders(self) -> list[Order]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM orders WHERE status IN ('pending', 'submitted', 'partial')"
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            Order(
                id=row["id"],
                ticker=row["ticker"],
                direction=row["direction"],
                order_type=row["order_type"],
                price=row["price"],
                quantity=row["quantity"],
                status=row["status"],
                strategy_name=row["strategy_name"],
                timestamp=_str_to_dt(row["timestamp"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Skipped signals
    # ------------------------------------------------------------------

    async def save_signal_skipped(self, signal: TradeSignal, reason: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO skipped_signals
                    (strategy_name, ticker, direction, entry_price,
                     confidence, timestamp, reason, features_snapshot)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.strategy_name,
                    signal.ticker,
                    signal.direction,
                    signal.entry_price,
                    signal.confidence,
                    _dt_to_str(signal.timestamp),
                    reason,
                    json.dumps(signal.features_snapshot, ensure_ascii=False),
                ),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Strategy performance
    # ------------------------------------------------------------------

    async def update_strategy_performance(
        self, strategy_name: str, perf: StrategyPerformance
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO strategy_performance
                    (strategy_name, total_trades, wins, losses, win_rate,
                     profit_factor, avg_pnl, avg_holding_minutes, sharpe_ratio,
                     max_drawdown, performance_by_condition, performance_by_event,
                     last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    perf.total_trades,
                    perf.wins,
                    perf.losses,
                    perf.win_rate,
                    perf.profit_factor,
                    perf.avg_pnl,
                    perf.avg_holding_minutes,
                    perf.sharpe_ratio,
                    perf.max_drawdown,
                    json.dumps(perf.performance_by_condition, ensure_ascii=False),
                    json.dumps(perf.performance_by_event, ensure_ascii=False),
                    _dt_to_str(perf.last_updated),
                ),
            )
            await db.commit()

    async def get_strategy_performance(
        self, strategy_name: str, days: int = 30
    ) -> StrategyPerformance | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM strategy_performance WHERE strategy_name = ?",
                (strategy_name,),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        return StrategyPerformance(
            strategy_name=row["strategy_name"],
            total_trades=row["total_trades"],
            wins=row["wins"],
            losses=row["losses"],
            win_rate=row["win_rate"],
            profit_factor=row["profit_factor"],
            avg_pnl=row["avg_pnl"],
            avg_holding_minutes=row["avg_holding_minutes"],
            sharpe_ratio=row["sharpe_ratio"],
            max_drawdown=row["max_drawdown"],
            performance_by_condition=json.loads(row["performance_by_condition"]),
            performance_by_event=json.loads(row["performance_by_event"]),
            last_updated=_str_to_dt(row["last_updated"]),
        )

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    async def save_knowledge(self, entry: KnowledgeEntry) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO knowledge
                    (id, date, category, content, supporting_trades,
                     confidence, auto_generated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    _dt_to_str(entry.date),
                    entry.category,
                    entry.content,
                    json.dumps(entry.supporting_trades, ensure_ascii=False),
                    entry.confidence,
                    1 if entry.auto_generated else 0,
                ),
            )
            await db.commit()

    async def get_knowledge(
        self,
        category: str | None = None,
        limit: int = 100,
    ) -> list[KnowledgeEntry]:
        if category is not None:
            sql = "SELECT * FROM knowledge WHERE category = ? ORDER BY date DESC LIMIT ?"
            params: tuple = (category, limit)
        else:
            sql = "SELECT * FROM knowledge ORDER BY date DESC LIMIT ?"
            params = (limit,)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()

        return [
            KnowledgeEntry(
                id=row["id"],
                date=date.fromisoformat(row["date"]),
                category=row["category"],
                content=row["content"],
                supporting_trades=json.loads(row["supporting_trades"]),
                confidence=row["confidence"],
                auto_generated=bool(row["auto_generated"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Candidate updates
    # ------------------------------------------------------------------

    async def save_candidate_update(self, cu: CandidateUpdate) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO candidate_updates
                    (id, strategy_name, proposed_changes, reason,
                     expected_improvement, status, created_at, reviewed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cu.id,
                    cu.strategy_name,
                    json.dumps(cu.proposed_changes, ensure_ascii=False),
                    cu.reason,
                    cu.expected_improvement,
                    cu.status,
                    _dt_to_str(cu.created_at),
                    _dt_to_str(cu.reviewed_at) if cu.reviewed_at else None,
                ),
            )
            await db.commit()

    async def get_pending_updates(self) -> list[CandidateUpdate]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM candidate_updates WHERE status = 'pending' ORDER BY created_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()

        return [
            CandidateUpdate(
                id=row["id"],
                strategy_name=row["strategy_name"],
                proposed_changes=json.loads(row["proposed_changes"]),
                reason=row["reason"],
                expected_improvement=row["expected_improvement"],
                status=row["status"],
                created_at=_str_to_dt(row["created_at"]),
                reviewed_at=_str_to_dt(row["reviewed_at"]) if row["reviewed_at"] else None,
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Feature snapshots
    # ------------------------------------------------------------------

    async def save_feature_snapshot(
        self, ticker: str, snapshot_date: date, features: dict
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO feature_snapshots (ticker, date, features, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    ticker,
                    _dt_to_str(snapshot_date),
                    json.dumps(features, ensure_ascii=False),
                    _dt_to_str(datetime.now()),
                ),
            )
            await db.commit()

    async def get_feature_snapshots(
        self, ticker: str, snapshot_date: date
    ) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT features FROM feature_snapshots WHERE ticker = ? AND date = ? ORDER BY created_at",
                (ticker, _dt_to_str(snapshot_date)),
            ) as cursor:
                rows = await cursor.fetchall()

        return [json.loads(row["features"]) for row in rows]

    # ------------------------------------------------------------------
    # Daily reports
    # ------------------------------------------------------------------

    async def save_daily_report(self, report: DailyReport) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO daily_reports
                    (date, total_pnl, total_trades, win_rate,
                     best_trade, worst_trade, strategy_summary,
                     knowledge_entries, improvement_candidates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _dt_to_str(report.date),
                    report.total_pnl,
                    report.total_trades,
                    report.win_rate,
                    report.best_trade,
                    report.worst_trade,
                    json.dumps(report.strategy_summary, ensure_ascii=False),
                    json.dumps(report.knowledge_entries, ensure_ascii=False),
                    json.dumps(report.improvement_candidates, ensure_ascii=False),
                ),
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Ticker affinity
    # ------------------------------------------------------------------

    async def update_ticker_affinity(
        self, strategy_name: str, ticker: str, pnl: float, is_win: bool,
    ) -> None:
        """Update win/loss stats for a strategy-ticker pair."""
        async with aiosqlite.connect(self.db_path) as db:
            # Upsert
            await db.execute(
                """
                INSERT INTO ticker_affinity (strategy_name, ticker, trades, wins, avg_pnl, last_updated)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(strategy_name, ticker) DO UPDATE SET
                    trades = trades + 1,
                    wins = wins + (CASE WHEN ? THEN 1 ELSE 0 END),
                    avg_pnl = (avg_pnl * trades + ?) / (trades + 1),
                    last_updated = ?
                """,
                (
                    strategy_name, ticker, 1 if is_win else 0, pnl,
                    _dt_to_str(datetime.now()),
                    is_win, pnl, _dt_to_str(datetime.now()),
                ),
            )
            await db.commit()

    async def get_ticker_affinity(
        self, strategy_name: str, ticker: str,
    ) -> dict | None:
        """Get affinity stats for a strategy-ticker pair."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM ticker_affinity WHERE strategy_name = ? AND ticker = ?",
                (strategy_name, ticker),
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "trades": row["trades"],
            "wins": row["wins"],
            "win_rate": row["wins"] / row["trades"] if row["trades"] > 0 else 0,
            "avg_pnl": row["avg_pnl"],
        }

    # ------------------------------------------------------------------
    # Daily reports
    # ------------------------------------------------------------------

    async def get_daily_report(self, report_date: date) -> DailyReport | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM daily_reports WHERE date = ?",
                (_dt_to_str(report_date),),
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        return DailyReport(
            date=date.fromisoformat(row["date"]),
            total_pnl=row["total_pnl"],
            total_trades=row["total_trades"],
            win_rate=row["win_rate"],
            best_trade=row["best_trade"],
            worst_trade=row["worst_trade"],
            strategy_summary=json.loads(row["strategy_summary"]),
            knowledge_entries=json.loads(row["knowledge_entries"]),
            improvement_candidates=json.loads(row["improvement_candidates"]),
        )
