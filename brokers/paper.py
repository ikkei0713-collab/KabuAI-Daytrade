"""
Paper trading broker implementation.

Simulates order execution with realistic slippage for backtesting
and development. All state is maintained in-memory with optional
persistence to the database.

SAFETY: Checks ALLOW_LIVE_TRADING and refuses to operate if live
trading is enabled -- this broker is for simulation only.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from loguru import logger

from brokers.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

# Default configuration
_DEFAULT_INITIAL_BALANCE = 30_000.0  # 3万円
_DEFAULT_SLIPPAGE_PCT = 0.001  # 0.1%
_TRADE_LOG_DIR = Path("data/processed")


class PaperBroker(BaseBroker):
    """
    Paper trading broker that simulates order execution.

    Features:
    - In-memory position and order tracking
    - Simulated fills with configurable slippage (default 0.1%)
    - P&L tracking per position and overall
    - Daily reset capability
    - Thread-safe with asyncio.Lock
    - Activity logging to data/processed/

    Usage:
        broker = PaperBroker(initial_balance=10_000_000)
        order = Order(ticker="7203", side=OrderSide.BUY, quantity=100)
        filled = await broker.place_order(order)
    """

    def __init__(
        self,
        initial_balance: float = _DEFAULT_INITIAL_BALANCE,
        slippage_pct: float = _DEFAULT_SLIPPAGE_PCT,
        price_provider=None,
        allow_live_trading: bool = False,
    ) -> None:
        if allow_live_trading:
            raise RuntimeError("LIVE TRADING IS DISABLED")

        self._initial_balance = initial_balance
        self._cash_balance = initial_balance
        self._slippage_pct = slippage_pct
        self._price_provider = price_provider  # MarketDataProvider instance

        # State
        self._positions: dict[str, Position] = {}  # ticker -> Position
        self._orders: list[Order] = []
        self._trade_history: list[dict] = []
        self._realized_pnl: float = 0.0
        self._trading_day: str = date.today().isoformat()

        # Thread safety
        self._lock = asyncio.Lock()

        # Ensure log directory exists
        _TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            "PaperBroker initialized: balance={:,.0f} JPY, slippage={:.2%}",
            initial_balance, slippage_pct,
        )

    # ------------------------------------------------------------------
    # BaseBroker implementation
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> Order:
        """
        Place and immediately simulate execution of an order.

        Market orders are filled instantly with slippage applied.
        Limit orders are filled if the simulated price meets the limit.

        Args:
            order: The Order to execute.

        Returns:
            Updated Order with fill information.
        """
        async with self._lock:
            self._check_safety()
            self._check_trading_day()

            logger.info(
                "Paper order: {} {} x{} {} (type={}, limit={})",
                order.side.value,
                order.ticker,
                order.quantity,
                order.strategy_name or "manual",
                order.order_type.value,
                order.limit_price,
            )

            order.status = OrderStatus.SUBMITTED
            order.updated_at = datetime.now()
            self._orders.append(order)

            # Determine execution price
            base_price = await self._get_price(order.ticker)

            if base_price <= 0:
                order.status = OrderStatus.REJECTED
                order.notes = "Could not determine price"
                order.updated_at = datetime.now()
                logger.warning("Order rejected for {}: no price available", order.ticker)
                return order

            fill_price = self._apply_slippage(base_price, order.side)

            # For limit orders, check if the price meets the limit
            if order.order_type == OrderType.LIMIT and order.limit_price is not None:
                if order.side == OrderSide.BUY and fill_price > order.limit_price:
                    order.status = OrderStatus.PENDING
                    order.notes = f"Limit not reached: market={fill_price:.0f} > limit={order.limit_price:.0f}"
                    logger.info("Buy limit order pending: {} (market {} > limit {})",
                                order.ticker, fill_price, order.limit_price)
                    return order
                elif order.side == OrderSide.SELL and fill_price < order.limit_price:
                    order.status = OrderStatus.PENDING
                    order.notes = f"Limit not reached: market={fill_price:.0f} < limit={order.limit_price:.0f}"
                    logger.info("Sell limit order pending: {} (market {} < limit {})",
                                order.ticker, fill_price, order.limit_price)
                    return order

            # Check buying power
            total_cost = fill_price * order.quantity
            if order.side == OrderSide.BUY and total_cost > self._cash_balance:
                order.status = OrderStatus.REJECTED
                order.notes = f"Insufficient funds: need {total_cost:,.0f}, have {self._cash_balance:,.0f}"
                order.updated_at = datetime.now()
                logger.warning("Order rejected: insufficient funds for {}", order.ticker)
                return order

            # Check position for sells
            if order.side == OrderSide.SELL:
                pos = self._positions.get(order.ticker)
                if not pos or pos.quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    held = pos.quantity if pos else 0
                    order.notes = f"Insufficient shares: need {order.quantity}, have {held}"
                    order.updated_at = datetime.now()
                    logger.warning("Order rejected: insufficient shares for {}", order.ticker)
                    return order

            # Execute the fill
            order.filled_price = fill_price
            order.filled_quantity = order.quantity
            order.status = OrderStatus.FILLED
            order.updated_at = datetime.now()

            # Update positions and balance
            self._apply_fill(order)

            # Log the trade
            trade_record = {
                "order_id": order.order_id,
                "ticker": order.ticker,
                "side": order.side.value,
                "quantity": order.quantity,
                "price": fill_price,
                "total": total_cost if order.side == OrderSide.BUY else fill_price * order.quantity,
                "strategy": order.strategy_name,
                "timestamp": order.updated_at.isoformat(),
            }
            self._trade_history.append(trade_record)
            self._save_trade(trade_record)

            logger.info(
                "Paper fill: {} {} x{} @ {:.0f} (total: {:,.0f}). Balance: {:,.0f}",
                order.side.value, order.ticker, order.quantity,
                fill_price, fill_price * order.quantity, self._cash_balance,
            )

            return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID."""
        async with self._lock:
            for order in self._orders:
                if order.order_id == order_id and order.status in (
                    OrderStatus.PENDING,
                    OrderStatus.SUBMITTED,
                ):
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()
                    logger.info("Cancelled order {}", order_id)
                    return True

            logger.warning("Order {} not found or not cancellable", order_id)
            return False

    async def get_positions(self) -> list[Position]:
        """Get all open positions with updated P&L."""
        async with self._lock:
            positions = []
            for ticker, pos in self._positions.items():
                if pos.quantity > 0:
                    current_price = await self._get_price(ticker)
                    pos.current_price = current_price
                    pos.unrealized_pnl = (current_price - pos.average_price) * pos.quantity
                    positions.append(pos)
            return positions

    async def get_orders(self) -> list[Order]:
        """Get all orders for the current trading day."""
        async with self._lock:
            return list(self._orders)

    async def get_balance(self) -> float:
        """Get current cash balance."""
        async with self._lock:
            return self._cash_balance

    # ------------------------------------------------------------------
    # Paper-broker specific methods
    # ------------------------------------------------------------------

    async def get_total_pnl(self) -> dict[str, float]:
        """Get total P&L breakdown."""
        async with self._lock:
            unrealized = 0.0
            for ticker, pos in self._positions.items():
                if pos.quantity > 0:
                    current_price = await self._get_price(ticker)
                    unrealized += (current_price - pos.average_price) * pos.quantity

            return {
                "realized_pnl": self._realized_pnl,
                "unrealized_pnl": unrealized,
                "total_pnl": self._realized_pnl + unrealized,
                "cash_balance": self._cash_balance,
                "initial_balance": self._initial_balance,
                "return_pct": (
                    (self._cash_balance + unrealized - self._initial_balance)
                    / self._initial_balance * 100
                ),
            }

    async def daily_reset(self) -> None:
        """
        Reset for a new trading day.

        Clears pending orders and resets the trading day marker.
        Positions carry over. Trade history is preserved.
        """
        async with self._lock:
            # Cancel any pending orders
            for order in self._orders:
                if order.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                    order.status = OrderStatus.CANCELLED
                    order.updated_at = datetime.now()

            old_day = self._trading_day
            self._trading_day = date.today().isoformat()
            self._orders = []

            logger.info(
                "Daily reset: {} -> {}. Positions: {}, Realized P&L: {:,.0f}",
                old_day, self._trading_day, len(self._positions), self._realized_pnl,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_safety() -> None:
        """Ensure we never accidentally run alongside live trading."""
        # This is checked at __init__ time, but double-check on every action
        # by importing settings if available
        try:
            import os
            if os.getenv("ALLOW_LIVE_TRADING", "false").lower() == "true":
                raise RuntimeError("LIVE TRADING IS DISABLED")
        except ImportError:
            pass

    def _check_trading_day(self) -> None:
        """Auto-reset if the trading day has changed."""
        today = date.today().isoformat()
        if today != self._trading_day:
            logger.info("Trading day changed from {} to {}, pending orders will be stale", self._trading_day, today)
            self._trading_day = today

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Apply realistic slippage to a price.

        Buys slip up, sells slip down (adverse to the trader).
        Slippage is randomized around the configured percentage.
        """
        # Randomize slippage between 0 and 2x the configured amount
        actual_slippage = random.uniform(0, self._slippage_pct * 2)

        if side == OrderSide.BUY:
            slipped = price * (1 + actual_slippage)
        else:
            slipped = price * (1 - actual_slippage)

        # Round to valid tick size
        if slipped >= 5000:
            slipped = round(slipped)
        elif slipped >= 3000:
            slipped = round(slipped / 5) * 5
        elif slipped >= 1000:
            slipped = round(slipped)
        else:
            slipped = round(slipped, 1)

        return slipped

    def _apply_fill(self, order: Order) -> None:
        """Update positions and cash balance based on a filled order."""
        ticker = order.ticker
        fill_price = order.filled_price
        quantity = order.filled_quantity

        if order.side == OrderSide.BUY:
            cost = fill_price * quantity
            self._cash_balance -= cost

            if ticker in self._positions:
                pos = self._positions[ticker]
                total_cost = pos.average_price * pos.quantity + cost
                pos.quantity += quantity
                pos.average_price = total_cost / pos.quantity if pos.quantity > 0 else 0
            else:
                self._positions[ticker] = Position(
                    ticker=ticker,
                    quantity=quantity,
                    average_price=fill_price,
                    current_price=fill_price,
                    strategy_name=order.strategy_name,
                )

        elif order.side == OrderSide.SELL:
            proceeds = fill_price * quantity
            self._cash_balance += proceeds

            pos = self._positions[ticker]
            pnl = (fill_price - pos.average_price) * quantity
            self._realized_pnl += pnl
            pos.realized_pnl += pnl
            pos.quantity -= quantity

            if pos.quantity <= 0:
                del self._positions[ticker]

    async def _get_price(self, ticker: str) -> float:
        """Get current price from provider or generate simulated price."""
        if self._price_provider:
            try:
                return await self._price_provider.get_current_price(ticker)
            except Exception as e:
                logger.warning("Price provider error for {}: {}", ticker, e)

        # Fallback: use last fill price from positions
        if ticker in self._positions:
            return self._positions[ticker].current_price

        # Fallback: generate a deterministic fake price
        rng = random.Random(int(ticker) if ticker.isdigit() else hash(ticker))
        return round(rng.uniform(500, 5000))

    def _save_trade(self, trade: dict) -> None:
        """Persist a trade record to disk."""
        try:
            log_file = _TRADE_LOG_DIR / f"paper_trades_{self._trading_day}.jsonl"
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("Failed to save trade record: {}", e)
