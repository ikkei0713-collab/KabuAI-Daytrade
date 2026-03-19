"""
Risk management module for KabuAI day trading.

Enforces position limits, daily loss limits, sector concentration
limits, and calculates stop-loss / take-profit levels.
"""

from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from core.config import settings
from core.models import Position, TradeSignal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOT_SIZE = 100                      # 単元株数
MAX_SECTOR_CONCENTRATION = 0.4      # 同一セクターに40%以上集中しない
MAX_CORRELATED_POSITIONS = 3        # 同セクター最大3ポジション
DEFAULT_ATR_STOP_MULTIPLIER = 2.0   # ATR x 2 でストップロス
DEFAULT_RISK_REWARD_RATIO = 2.0     # リスクリワード比


class RiskManager:
    """
    Centralized risk management for the trading system.

    Enforces:
    - Maximum number of simultaneous positions
    - Maximum position size (yen value)
    - Daily loss limit
    - Sector concentration limits
    - Stop-loss and take-profit calculation

    Usage::

        rm = RiskManager()
        can_trade = rm.check_position_risk(signal, portfolio)
        stop = rm.calculate_stop_loss(entry, atr, "long")
        tp = rm.calculate_take_profit(entry, atr, "long")
    """

    def __init__(
        self,
        max_positions: Optional[int] = None,
        max_position_size: Optional[float] = None,
        max_daily_loss: Optional[float] = None,
        total_capital: Optional[float] = None,
    ) -> None:
        self.max_positions = max_positions or settings.MAX_POSITIONS
        self.max_position_size = max_position_size or settings.MAX_POSITION_SIZE
        self.max_daily_loss = max_daily_loss or settings.MAX_LOSS_PER_DAY
        self.total_capital = total_capital or settings.TOTAL_CAPITAL

        # Track sector mapping (ticker -> sector)
        self._sector_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sector_map(self, sector_map: dict[str, str]) -> None:
        """Set the ticker-to-sector mapping for concentration checks."""
        self._sector_map = sector_map

    def check_position_risk(
        self,
        signal: TradeSignal,
        portfolio: dict[str, Any],
    ) -> bool:
        """
        Check if a new position can be opened based on risk constraints.

        Args:
            signal: The trade signal proposing a new position.
            portfolio: Dict with keys:
                - positions: list[Position] of current open positions
                - daily_pnl: float of today's realized P&L
                - capital: float of total capital

        Returns:
            True if the position is allowed, False otherwise.
        """
        positions: list[Position] = portfolio.get("positions", [])
        daily_pnl: float = portfolio.get("daily_pnl", 0.0)
        capital: float = portfolio.get("capital", self.total_capital)

        # Check 1: Max positions
        if len(positions) >= self.max_positions:
            logger.warning(
                "RiskManager: max positions reached ({}/{})",
                len(positions), self.max_positions,
            )
            return False

        # Check 2: Daily loss limit
        if daily_pnl <= self.max_daily_loss:
            logger.warning(
                "RiskManager: daily loss limit reached ({:.0f}円 <= {:.0f}円)",
                daily_pnl, self.max_daily_loss,
            )
            return False

        # Check 3: Position size
        position_value = signal.entry_price * LOT_SIZE
        if position_value > self.max_position_size:
            # Check if a smaller lot is feasible
            max_lots = int(self.max_position_size / signal.entry_price // LOT_SIZE)
            if max_lots <= 0:
                logger.warning(
                    "RiskManager: position size too large "
                    "({:.0f}円 > {:.0f}円 max, no valid lot size)",
                    position_value, self.max_position_size,
                )
                return False

        # Check 4: Total exposure
        total_exposure = sum(
            p.entry_price * LOT_SIZE for p in positions
        )
        new_exposure = total_exposure + signal.entry_price * LOT_SIZE
        max_exposure = capital * 0.8  # 80% of capital max
        if new_exposure > max_exposure:
            logger.warning(
                "RiskManager: total exposure would exceed limit "
                "({:.0f}円 > {:.0f}円)",
                new_exposure, max_exposure,
            )
            return False

        # Check 5: Sector concentration
        if not self._check_sector_concentration(signal.ticker, positions):
            return False

        # Check 6: No duplicate ticker/direction
        for pos in positions:
            if pos.ticker == signal.ticker and pos.direction == signal.direction:
                logger.warning(
                    "RiskManager: duplicate position for {} {}",
                    signal.ticker, signal.direction,
                )
                return False

        logger.debug(
            "RiskManager: position risk check passed for {} {}",
            signal.ticker, signal.direction,
        )
        return True

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        multiplier: float = DEFAULT_ATR_STOP_MULTIPLIER,
    ) -> float:
        """
        Calculate stop-loss price based on ATR.

        For long positions: entry - ATR * multiplier
        For short positions: entry + ATR * multiplier

        The result is rounded to the nearest yen (integer).
        """
        stop_distance = atr * multiplier

        if direction == "long":
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Round to nearest yen
        stop_loss = round(stop_loss)

        # Ensure stop loss is positive
        stop_loss = max(1.0, stop_loss)

        logger.debug(
            "RiskManager: stop_loss for {} @ {:.0f} (ATR={:.1f}, mult={:.1f}) = {:.0f}",
            direction, entry_price, atr, multiplier, stop_loss,
        )
        return stop_loss

    def calculate_take_profit(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        risk_reward: float = DEFAULT_RISK_REWARD_RATIO,
    ) -> float:
        """
        Calculate take-profit price based on ATR and risk/reward ratio.

        take_profit_distance = ATR * stop_multiplier * risk_reward

        Rounded to nearest yen.
        """
        tp_distance = atr * DEFAULT_ATR_STOP_MULTIPLIER * risk_reward

        if direction == "long":
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        # Round to nearest yen
        take_profit = round(take_profit)

        # Ensure take profit is positive
        take_profit = max(1.0, take_profit)

        logger.debug(
            "RiskManager: take_profit for {} @ {:.0f} (ATR={:.1f}, RR={:.1f}) = {:.0f}",
            direction, entry_price, atr, risk_reward, take_profit,
        )
        return take_profit

    def portfolio_risk_assessment(
        self,
        positions: list[Position],
        daily_pnl: float = 0.0,
    ) -> dict[str, Any]:
        """
        Generate a comprehensive risk assessment of the current portfolio.

        Returns dict with:
        - total_exposure: Total yen value of open positions
        - exposure_pct: Percentage of capital deployed
        - position_count: Number of open positions
        - sector_breakdown: Dict of sector -> count and value
        - max_unrealized_loss: Worst unrealized P&L
        - max_unrealized_gain: Best unrealized P&L
        - daily_pnl: Today's realized P&L
        - remaining_loss_budget: How much more we can lose today
        - risk_level: "low", "medium", "high", "critical"
        """
        total_exposure = sum(p.entry_price * LOT_SIZE for p in positions)
        exposure_pct = total_exposure / self.total_capital if self.total_capital > 0 else 0.0

        # Sector breakdown
        sector_breakdown: dict[str, dict[str, Any]] = {}
        for pos in positions:
            sector = self._sector_map.get(pos.ticker, "unknown")
            if sector not in sector_breakdown:
                sector_breakdown[sector] = {"count": 0, "value": 0.0, "tickers": []}
            sector_breakdown[sector]["count"] += 1
            sector_breakdown[sector]["value"] += pos.entry_price * LOT_SIZE
            sector_breakdown[sector]["tickers"].append(pos.ticker)

        # Unrealized P&L extremes
        unrealized_pnls = [p.unrealized_pnl for p in positions] if positions else [0.0]
        max_unrealized_loss = min(unrealized_pnls)
        max_unrealized_gain = max(unrealized_pnls)

        # Remaining loss budget
        remaining_loss = daily_pnl - self.max_daily_loss

        # Risk level
        if daily_pnl <= self.max_daily_loss * 0.5:
            risk_level = "critical"
        elif daily_pnl <= self.max_daily_loss * 0.3:
            risk_level = "high"
        elif exposure_pct > 0.6 or len(positions) >= self.max_positions - 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        assessment = {
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "position_count": len(positions),
            "max_positions": self.max_positions,
            "sector_breakdown": sector_breakdown,
            "max_unrealized_loss": max_unrealized_loss,
            "max_unrealized_gain": max_unrealized_gain,
            "daily_pnl": daily_pnl,
            "remaining_loss_budget": remaining_loss,
            "risk_level": risk_level,
        }

        logger.info(
            "RiskManager: portfolio assessment - "
            "exposure={:.0f}円 ({:.1%}), positions={}/{}, risk={}",
            total_exposure, exposure_pct, len(positions),
            self.max_positions, risk_level,
        )

        return assessment

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_sector_concentration(
        self,
        ticker: str,
        positions: list[Position],
    ) -> bool:
        """Check that adding this ticker doesn't over-concentrate in one sector."""
        sector = self._sector_map.get(ticker, "unknown")

        if sector == "unknown":
            # Can't check concentration without sector info
            return True

        # Count existing positions in the same sector
        same_sector_count = sum(
            1 for p in positions
            if self._sector_map.get(p.ticker, "unknown") == sector
        )

        if same_sector_count >= MAX_CORRELATED_POSITIONS:
            logger.warning(
                "RiskManager: sector concentration limit reached "
                "(sector={}, count={}/{})",
                sector, same_sector_count, MAX_CORRELATED_POSITIONS,
            )
            return False

        # Check value concentration
        if positions:
            same_sector_value = sum(
                p.entry_price * LOT_SIZE
                for p in positions
                if self._sector_map.get(p.ticker, "unknown") == sector
            )
            total_value = sum(p.entry_price * LOT_SIZE for p in positions)
            if total_value > 0:
                concentration = same_sector_value / total_value
                if concentration > MAX_SECTOR_CONCENTRATION:
                    logger.warning(
                        "RiskManager: sector value concentration too high "
                        "(sector={}, {:.1%} > {:.1%})",
                        sector, concentration, MAX_SECTOR_CONCENTRATION,
                    )
                    return False

        return True
