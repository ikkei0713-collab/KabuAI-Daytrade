"""
Main execution engine for KabuAI day trading.

Orchestrates the full trading loop: scanning, strategy evaluation,
signal generation, order execution, position monitoring, and end-of-day.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, time, timedelta
from typing import Any, Optional, Protocol

from loguru import logger

from core.config import settings
from core.models import (
    DailyReport,
    MarketCondition,
    Order,
    Position,
    TradeResult,
    TradeSignal,
)
from execution.risk_manager import RiskManager
from scanners.score_engine import ScoreEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOT_SIZE = 100              # 単元株数
LOOP_INTERVAL_SEC = 30      # Main loop interval
SIGNAL_DEDUP_WINDOW_SEC = 300  # 5 minutes dedup window for idempotent orders


class BrokerProtocol(Protocol):
    """Minimal interface for the broker (PaperBroker or live)."""

    async def place_order(self, order: Order) -> Order: ...
    async def get_positions(self) -> list[Position]: ...
    async def close_position(self, position_id: str, price: float) -> TradeResult: ...
    async def get_current_price(self, ticker: str) -> float: ...
    async def cancel_order(self, order_id: str) -> bool: ...


class DatabaseProtocol(Protocol):
    """Minimal interface for the database manager."""

    async def save_trade(self, trade: TradeResult) -> None: ...
    async def save_order(self, order: Order) -> None: ...
    async def save_signal(self, signal: TradeSignal) -> None: ...
    async def save_daily_report(self, report: DailyReport) -> None: ...
    async def get_trades_for_date(self, d: date) -> list[TradeResult]: ...
    async def get_orders_for_date(self, d: date) -> list[Order]: ...
    async def get_strategy_performances(self) -> dict: ...


class StrategyProtocol(Protocol):
    """Minimal interface for a trading strategy."""

    @property
    def name(self) -> str: ...

    async def generate_signal(
        self, ticker: str, features: dict[str, float], market_condition: MarketCondition,
    ) -> Optional[TradeSignal]: ...

    async def should_exit(
        self, position: Position, features: dict[str, float], market_condition: MarketCondition,
    ) -> tuple[bool, str]: ...

    async def calculate_features(
        self, ticker: str, prices: list[dict[str, Any]],
    ) -> dict[str, float]: ...


class ExecutionEngine:
    """
    Orchestrates the full day-trading cycle for Japanese stocks.

    Responsibilities:
    - Run the main trading loop during TSE hours (09:00-15:25 JST)
    - Evaluate strategies and generate trade signals
    - Execute orders through the broker with risk checks
    - Monitor open positions for exit conditions
    - Force close all positions at FORCE_CLOSE_TIME (15:20 JST)
    - Generate end-of-day reports and trigger analytics

    Usage::

        engine = ExecutionEngine(broker, db, strategies, score_engine)
        await engine.run_trading_loop()
    """

    def __init__(
        self,
        broker: BrokerProtocol,
        db: DatabaseProtocol,
        strategies: list[StrategyProtocol],
        score_engine: ScoreEngine,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.broker = broker
        self.db = db
        self.strategies = strategies
        self.score_engine = score_engine
        self.risk_manager = risk_manager or RiskManager()

        self._running = False
        self._daily_pnl = 0.0
        self._today_orders: dict[str, Order] = {}  # dedup key -> order
        self._today_trades: list[TradeResult] = []

    # ------------------------------------------------------------------
    # Main trading loop
    # ------------------------------------------------------------------

    async def run_trading_loop(
        self,
        watchlist: list[str],
        market_condition: MarketCondition,
        price_fetcher: Any = None,
    ) -> None:
        """
        Run the main trading loop until market close or stop signal.

        Args:
            watchlist: List of ticker codes to trade today.
            market_condition: Current market condition snapshot.
            price_fetcher: Callable to fetch current prices/features for a ticker.
        """
        self._running = True
        self._daily_pnl = 0.0
        self._today_orders.clear()
        self._today_trades.clear()

        market_open = self._parse_time(settings.MARKET_OPEN)
        market_close = self._parse_time(settings.MARKET_CLOSE)
        force_close = self._parse_time(settings.FORCE_CLOSE_TIME)

        logger.info(
            "ExecutionEngine: starting trading loop "
            "(watchlist={} tickers, market_open={}, force_close={})",
            len(watchlist), market_open, force_close,
        )

        while self._running:
            now = datetime.now().time()

            # Pre-market: don't trade yet
            if now < market_open:
                logger.debug("ExecutionEngine: waiting for market open")
                await asyncio.sleep(LOOP_INTERVAL_SEC)
                continue

            # After market close: stop loop
            if now >= market_close:
                logger.info("ExecutionEngine: market closed, ending loop")
                break

            # Safety: check daily loss limit
            if not self._check_daily_loss_limit():
                logger.warning(
                    "ExecutionEngine: daily loss limit reached ({:.0f}円), stopping",
                    self._daily_pnl,
                )
                break

            # Force close time: close everything
            if now >= force_close:
                logger.info("ExecutionEngine: force close time reached")
                await self._force_close_all_positions(market_condition)
                break

            # --- Main loop body ---
            try:
                # Monitor existing positions first
                await self.monitor_positions(market_condition)

                # Process watchlist for new signals
                for ticker in watchlist:
                    await self._process_ticker(ticker, market_condition, price_fetcher)

            except Exception as e:
                logger.error("ExecutionEngine: loop iteration error: {}", e)

            await asyncio.sleep(LOOP_INTERVAL_SEC)

        # End of day
        await self.end_of_day()
        logger.info("ExecutionEngine: trading loop ended")

    async def stop(self) -> None:
        """Signal the trading loop to stop."""
        logger.info("ExecutionEngine: stop signal received")
        self._running = False

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------

    async def process_signal(self, signal: TradeSignal) -> Optional[Order]:
        """
        Process a trade signal through safety checks, sizing, and execution.

        Returns the placed Order if successful, None otherwise.
        """
        logger.info(
            "ExecutionEngine: processing signal {} {} {} @ {:.0f} (confidence={:.2f})",
            signal.strategy_name, signal.direction, signal.ticker,
            signal.entry_price, signal.confidence,
        )

        # Idempotency check
        dedup_key = self._signal_dedup_key(signal)
        if dedup_key in self._today_orders:
            logger.info(
                "ExecutionEngine: duplicate signal ignored ({})", dedup_key,
            )
            return None

        # Confidence threshold
        if signal.confidence < settings.MIN_CONFIDENCE:
            logger.info(
                "ExecutionEngine: signal confidence {:.2f} below threshold {:.2f}",
                signal.confidence, settings.MIN_CONFIDENCE,
            )
            return None

        # Risk checks
        positions = await self.broker.get_positions()
        portfolio = {
            "positions": positions,
            "daily_pnl": self._daily_pnl,
            "capital": settings.TOTAL_CAPITAL,
        }

        if not self.risk_manager.check_position_risk(signal, portfolio):
            logger.warning(
                "ExecutionEngine: signal rejected by risk manager: {} {}",
                signal.ticker, signal.strategy_name,
            )
            return None

        # Position sizing (round to lot size)
        quantity = self._calculate_quantity(signal.entry_price)
        if quantity <= 0:
            logger.warning("ExecutionEngine: calculated quantity is 0, skipping")
            return None

        # Create and place order
        order = Order(
            ticker=signal.ticker,
            direction=signal.direction,
            order_type="market",
            price=signal.entry_price,
            quantity=quantity,
            status="pending",
            strategy_name=signal.strategy_name,
        )

        try:
            placed = await self.broker.place_order(order)
            self._today_orders[dedup_key] = placed

            # Log
            await self.db.save_order(placed)
            await self.db.save_signal(signal)

            logger.info(
                "ExecutionEngine: order placed {} {} {} x{} @ {:.0f}",
                placed.id, placed.direction, placed.ticker,
                placed.quantity, placed.price,
            )
            return placed

        except Exception as e:
            logger.error(
                "ExecutionEngine: order placement failed for {} {}: {}",
                signal.ticker, signal.strategy_name, e,
            )
            return None

    # ------------------------------------------------------------------
    # Position monitoring
    # ------------------------------------------------------------------

    async def monitor_positions(
        self, market_condition: MarketCondition,
    ) -> None:
        """
        Monitor all open positions for exit signals.

        For each position:
        1. Update current price and unrealized P&L
        2. Check each strategy's should_exit()
        3. Check stop loss and take profit
        4. Track holding time
        """
        positions = await self.broker.get_positions()

        if not positions:
            return

        logger.debug("ExecutionEngine: monitoring {} positions", len(positions))

        for position in positions:
            try:
                # Update current price
                current_price = await self.broker.get_current_price(position.ticker)
                position.current_price = current_price

                # Calculate unrealized P&L
                if position.direction == "long":
                    position.unrealized_pnl = (
                        (current_price - position.entry_price) * position.current_price
                    )
                else:
                    position.unrealized_pnl = (
                        (position.entry_price - current_price) * position.current_price
                    )

                # Update holding time
                elapsed = datetime.now() - position.entry_time
                position.holding_minutes = int(elapsed.total_seconds() / 60)

                # Check stop loss
                if self._check_stop_loss(position, current_price):
                    logger.info(
                        "ExecutionEngine: stop loss hit for {} ({:.0f})",
                        position.ticker, current_price,
                    )
                    await self._close_position(position, current_price, "stop_loss")
                    continue

                # Check take profit
                if self._check_take_profit(position, current_price):
                    logger.info(
                        "ExecutionEngine: take profit hit for {} ({:.0f})",
                        position.ticker, current_price,
                    )
                    await self._close_position(position, current_price, "take_profit")
                    continue

                # Check max holding time
                if position.holding_minutes >= settings.MAX_HOLDING_MINUTES:
                    logger.info(
                        "ExecutionEngine: max holding time for {} ({}min)",
                        position.ticker, position.holding_minutes,
                    )
                    await self._close_position(position, current_price, "max_holding_time")
                    continue

                # Ask strategy if we should exit
                exit_triggered = False
                for strategy in self.strategies:
                    if strategy.name != position.strategy_name:
                        continue

                    # Get fresh features for exit decision
                    features: dict[str, float] = {}
                    should_exit, exit_reason = await strategy.should_exit(
                        position, features, market_condition,
                    )
                    if should_exit:
                        logger.info(
                            "ExecutionEngine: strategy exit for {} ({})",
                            position.ticker, exit_reason,
                        )
                        await self._close_position(
                            position, current_price, f"strategy: {exit_reason}",
                        )
                        exit_triggered = True
                        break

                if not exit_triggered:
                    logger.debug(
                        "ExecutionEngine: holding {} ({:.0f}円 P&L, {}min)",
                        position.ticker, position.unrealized_pnl,
                        position.holding_minutes,
                    )

            except Exception as e:
                logger.error(
                    "ExecutionEngine: error monitoring position {}: {}",
                    position.id, e,
                )

    # ------------------------------------------------------------------
    # End of day
    # ------------------------------------------------------------------

    async def end_of_day(self) -> None:
        """
        End-of-day processing:
        1. Force close any remaining positions
        2. Generate daily report
        3. Trigger analytics pipeline
        """
        logger.info("ExecutionEngine: running end-of-day processing")

        # Close any remaining positions
        positions = await self.broker.get_positions()
        for position in positions:
            try:
                current_price = await self.broker.get_current_price(position.ticker)
                await self._close_position(position, current_price, "end_of_day")
            except Exception as e:
                logger.error(
                    "ExecutionEngine: error closing position {} at EOD: {}",
                    position.id, e,
                )

        # Generate daily report
        report = DailyReport(
            date=date.today(),
            total_pnl=self._daily_pnl,
            total_trades=len(self._today_trades),
            win_rate=self._calculate_win_rate(),
            best_trade=self._get_best_trade_id(),
            worst_trade=self._get_worst_trade_id(),
            strategy_summary=self._build_strategy_summary(),
        )

        try:
            await self.db.save_daily_report(report)
            logger.info(
                "ExecutionEngine: daily report saved "
                "(P&L={:.0f}円, trades={}, win_rate={:.1%})",
                report.total_pnl, report.total_trades, report.win_rate,
            )
        except Exception as e:
            logger.error("ExecutionEngine: failed to save daily report: {}", e)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _process_ticker(
        self,
        ticker: str,
        market_condition: MarketCondition,
        price_fetcher: Any,
    ) -> None:
        """Evaluate strategies and generate signals for a single ticker."""
        # Get current features
        features: dict[str, float] = {}
        if price_fetcher:
            try:
                features = await price_fetcher(ticker)
            except Exception as e:
                logger.warning("Failed to fetch features for {}: {}", ticker, e)
                return

        # Score and select best strategies
        best_strategies = self.score_engine.select_best_strategies(
            ticker, features, market_condition, self.strategies,
        )

        if not best_strategies:
            return

        # Generate signals from top strategies
        for strategy, score in best_strategies:
            try:
                signal = await strategy.generate_signal(
                    ticker, features, market_condition,
                )
            except Exception as e:
                logger.warning(
                    "Strategy {} signal generation failed for {}: {}",
                    strategy.name, ticker, e,
                )
                continue

            if signal is None:
                continue

            # Boost/reduce confidence based on strategy score
            signal.confidence = signal.confidence * 0.7 + score * 0.3

            # Process the signal
            await self.process_signal(signal)

            # Only process one signal per ticker per loop
            break

    async def _force_close_all_positions(
        self, market_condition: MarketCondition,
    ) -> None:
        """Force close all open positions at FORCE_CLOSE_TIME."""
        positions = await self.broker.get_positions()
        logger.info(
            "ExecutionEngine: force closing {} positions", len(positions),
        )

        for position in positions:
            try:
                current_price = await self.broker.get_current_price(position.ticker)
                await self._close_position(position, current_price, "force_close_15:20")
            except Exception as e:
                logger.error(
                    "ExecutionEngine: error force closing {}: {}", position.id, e,
                )

    async def _close_position(
        self, position: Position, price: float, reason: str,
    ) -> Optional[TradeResult]:
        """Close a position and record the trade result."""
        try:
            trade = await self.broker.close_position(position.id, price)
            trade.exit_reason = reason
            trade.holding_minutes = int(
                (datetime.now() - position.entry_time).total_seconds() / 60,
            )

            self._daily_pnl += trade.pnl
            self._today_trades.append(trade)

            await self.db.save_trade(trade)
            logger.info(
                "ExecutionEngine: closed {} {} P&L={:.0f}円 ({})",
                position.ticker, position.direction, trade.pnl, reason,
            )
            return trade

        except Exception as e:
            logger.error(
                "ExecutionEngine: failed to close position {}: {}", position.id, e,
            )
            return None

    def _calculate_quantity(self, price: float) -> int:
        """
        Calculate order quantity respecting position size limits and lot size.

        Japanese stocks trade in units of LOT_SIZE (100 shares).
        """
        if price <= 0:
            return 0

        max_value = min(settings.MAX_POSITION_SIZE, settings.TOTAL_CAPITAL * 0.2)
        raw_shares = max_value / price
        lots = int(raw_shares // LOT_SIZE)
        return lots * LOT_SIZE

    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Check if stop loss has been hit."""
        if position.stop_loss <= 0:
            return False

        if position.direction == "long":
            return current_price <= position.stop_loss
        else:
            return current_price >= position.stop_loss

    def _check_take_profit(self, position: Position, current_price: float) -> bool:
        """Check if take profit has been hit."""
        if position.take_profit <= 0:
            return False

        if position.direction == "long":
            return current_price >= position.take_profit
        else:
            return current_price <= position.take_profit

    def _check_daily_loss_limit(self) -> bool:
        """Return True if we can continue trading (loss limit not reached)."""
        return self._daily_pnl > settings.MAX_LOSS_PER_DAY

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from today's trades."""
        if not self._today_trades:
            return 0.0
        wins = sum(1 for t in self._today_trades if t.pnl > 0)
        return wins / len(self._today_trades)

    def _get_best_trade_id(self) -> Optional[str]:
        """Get the ID of the best trade today."""
        if not self._today_trades:
            return None
        best = max(self._today_trades, key=lambda t: t.pnl)
        return best.id

    def _get_worst_trade_id(self) -> Optional[str]:
        """Get the ID of the worst trade today."""
        if not self._today_trades:
            return None
        worst = min(self._today_trades, key=lambda t: t.pnl)
        return worst.id

    def _build_strategy_summary(self) -> dict[str, Any]:
        """Build a per-strategy summary of today's trades."""
        summary: dict[str, Any] = {}

        for trade in self._today_trades:
            name = trade.strategy_name
            if name not in summary:
                summary[name] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0,
                    "best_pnl": 0.0,
                    "worst_pnl": 0.0,
                }

            s = summary[name]
            s["trades"] += 1
            s["total_pnl"] += trade.pnl
            if trade.pnl > 0:
                s["wins"] += 1
            s["best_pnl"] = max(s["best_pnl"], trade.pnl)
            s["worst_pnl"] = min(s["worst_pnl"], trade.pnl)

        # Add win rates
        for s in summary.values():
            s["win_rate"] = s["wins"] / s["trades"] if s["trades"] > 0 else 0.0

        return summary

    @staticmethod
    def _signal_dedup_key(signal: TradeSignal) -> str:
        """
        Generate a deduplication key for a signal.

        Two signals are considered duplicates if they have the same
        strategy, ticker, and direction within SIGNAL_DEDUP_WINDOW_SEC.
        """
        # Round timestamp to dedup window
        window = signal.timestamp.timestamp() // SIGNAL_DEDUP_WINDOW_SEC
        return f"{signal.strategy_name}:{signal.ticker}:{signal.direction}:{int(window)}"

    @staticmethod
    def _parse_time(time_str: str) -> time:
        """Parse a HH:MM time string."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))
