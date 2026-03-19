"""Trade cost model for realistic backtest simulation.

Accounts for:
- Brokerage commission (SBI Securities standard plan)
- Slippage (estimated from spread and volatility)
- Market impact (for larger orders)
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger


@dataclass
class TradeCost:
    """Breakdown of costs for a single trade (round-trip)."""
    commission: float = 0.0      # 手数料（往復）
    slippage: float = 0.0        # スリッページ
    market_impact: float = 0.0   # マーケットインパクト

    @property
    def total(self) -> float:
        return self.commission + self.slippage + self.market_impact


class CostModel:
    """Realistic cost model for Japanese equities day-trading.

    Default parameters approximate SBI Securities active plan +
    typical slippage for mid/large-cap stocks.
    """

    # SBI Securities commission tiers (one-way)
    COMMISSION_TIERS = [
        (50_000, 55),
        (100_000, 99),
        (200_000, 115),
        (500_000, 275),
        (1_000_000, 535),
        (1_500_000, 640),
        (30_000_000, 1013),
        (float("inf"), 1070),
    ]

    def __init__(
        self,
        slippage_bps: float = 5.0,      # 5 bps default slippage
        impact_bps: float = 2.0,         # 2 bps market impact per 1M notional
        commission_free: bool = False,    # Some plans have 0 commission
    ):
        self.slippage_bps = slippage_bps
        self.impact_bps = impact_bps
        self.commission_free = commission_free

    def _one_way_commission(self, notional: float) -> float:
        """Compute one-way commission based on notional value."""
        if self.commission_free:
            return 0.0
        for threshold, fee in self.COMMISSION_TIERS:
            if notional <= threshold:
                return fee
        return self.COMMISSION_TIERS[-1][1]

    def calculate_trade_cost(
        self,
        entry_price: float,
        exit_price: float,
        quantity: int,
    ) -> TradeCost:
        """Calculate round-trip trade cost.

        Args:
            entry_price: Entry price per share.
            exit_price: Exit price per share.
            quantity: Number of shares.

        Returns:
            TradeCost with full breakdown.
        """
        entry_notional = entry_price * quantity
        exit_notional = exit_price * quantity

        commission = (
            self._one_way_commission(entry_notional)
            + self._one_way_commission(exit_notional)
        )

        avg_notional = (entry_notional + exit_notional) / 2.0
        slippage = avg_notional * (self.slippage_bps / 10_000.0)

        impact = avg_notional * (self.impact_bps / 10_000.0) * (avg_notional / 1_000_000.0)
        impact = min(impact, avg_notional * 0.001)  # cap at 10 bps

        return TradeCost(
            commission=round(commission, 0),
            slippage=round(slippage, 0),
            market_impact=round(impact, 0),
        )

    def adjust_entry_price(self, price: float, direction: str) -> float:
        """Adjust entry price for slippage (worse fill).

        Long entries fill higher; short entries fill lower.
        """
        slip = price * (self.slippage_bps / 10_000.0 / 2.0)
        if direction == "long":
            return price + slip
        else:
            return price - slip

    def adjust_exit_price(self, price: float, direction: str) -> float:
        """Adjust exit price for slippage (worse fill).

        Long exits fill lower; short exits fill higher.
        """
        slip = price * (self.slippage_bps / 10_000.0 / 2.0)
        if direction == "long":
            return price - slip
        else:
            return price + slip
