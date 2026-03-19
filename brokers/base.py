"""
Abstract broker interface.

Defines the contract that all broker implementations (paper, live) must
follow, along with shared data models for orders and positions.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""

    ticker: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    notes: str = ""


@dataclass
class Position:
    """Represents a current holding."""

    ticker: str
    quantity: int
    average_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.average_price


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.

    All broker integrations (paper trading, Tachibana Securities, etc.)
    must implement this interface to ensure consistency across the system.
    """

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """
        Submit an order for execution.

        Args:
            order: The Order object to submit.

        Returns:
            The updated Order with status and fill information.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: The unique order ID to cancel.

        Returns:
            True if cancellation was successful, False otherwise.
        """
        ...

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """
        Get all current open positions.

        Returns:
            List of Position objects.
        """
        ...

    @abstractmethod
    async def get_orders(self) -> list[Order]:
        """
        Get all orders (pending, filled, cancelled).

        Returns:
            List of Order objects.
        """
        ...

    @abstractmethod
    async def get_balance(self) -> float:
        """
        Get the current account cash balance.

        Returns:
            Available cash balance as a float (in JPY).
        """
        ...
