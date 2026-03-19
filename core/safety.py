"""Safety guards that prevent dangerous trading actions.

Every guard raises ``SafetyError`` on violation so the caller can handle
it uniformly.  ``SafetyGuard`` wraps all individual checks into a
convenient facade.
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Sequence

from core.config import settings
from core.models import Order


class SafetyError(Exception):
    """Raised when a safety check fails."""


# ---------------------------------------------------------------------------
# Anomaly detection helpers
# ---------------------------------------------------------------------------


def check_consecutive_losses(recent_trades: list, max_consecutive: int = 4) -> bool:
    """Return True if the last max_consecutive trades were all losses."""
    if len(recent_trades) < max_consecutive:
        return False
    return all(t.pnl <= 0 for t in recent_trades[-max_consecutive:])


def check_drawdown_speed(
    recent_trades: list, window: int = 10, max_loss_pct: float = 3.0, capital: float = 3_000_000,
) -> bool:
    """Return True if drawdown in last N trades exceeds threshold."""
    if len(recent_trades) < 2:
        return False
    window_trades = recent_trades[-window:]
    window_pnl = sum(t.pnl for t in window_trades)
    pct = window_pnl / capital * 100
    return pct < -max_loss_pct


def check_strategy_degradation(
    strategy_name: str, recent_trades: list, min_trades: int = 10, min_pf: float = 0.7,
) -> bool:
    """Return True if a strategy's rolling PF has degraded below threshold."""
    strades = [t for t in recent_trades if t.strategy_name == strategy_name]
    if len(strades) < min_trades:
        return False
    gp = sum(t.pnl for t in strades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in strades if t.pnl <= 0))
    pf = gp / gl if gl > 0 else 99.0
    return pf < min_pf


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_live_trading_blocked() -> None:
    """Raise if live trading is enabled.

    During development the system must NEVER place real orders.  This
    guard ensures ``ALLOW_LIVE_TRADING`` stays ``False``.
    """
    if settings.ALLOW_LIVE_TRADING:
        raise SafetyError(
            "ALLOW_LIVE_TRADING is True — live trading is blocked by safety policy. "
            "Set ALLOW_LIVE_TRADING=False or enable PAPER_TRADING."
        )


def check_daily_loss_limit(current_pnl: float) -> None:
    """Raise if the daily P&L has breached the configured loss limit."""
    if current_pnl <= settings.MAX_LOSS_PER_DAY:
        raise SafetyError(
            f"Daily loss limit reached: current P&L ¥{current_pnl:,.0f} "
            f"<= limit ¥{settings.MAX_LOSS_PER_DAY:,.0f}. "
            "No further trading allowed today."
        )


def check_max_positions(current_count: int) -> None:
    """Raise if opening another position would exceed the limit."""
    if current_count >= settings.MAX_POSITIONS:
        raise SafetyError(
            f"Maximum positions reached: {current_count}/{settings.MAX_POSITIONS}. "
            "Close an existing position before opening a new one."
        )


def check_market_hours(now: datetime | None = None) -> bool:
    """Return ``True`` if the market is currently open (JST).

    Does **not** raise — callers decide how to handle a closed market.
    """
    if now is None:
        now = datetime.now()

    open_h, open_m = map(int, settings.MARKET_OPEN.split(":"))
    close_h, close_m = map(int, settings.MARKET_CLOSE.split(":"))

    market_open = time(open_h, open_m)
    market_close = time(close_h, close_m)

    current_time = now.time()
    return market_open <= current_time <= market_close


def check_duplicate_order(
    ticker: str,
    direction: str,
    existing_orders: Sequence[Order],
) -> None:
    """Raise if an identical pending order already exists."""
    for order in existing_orders:
        if (
            order.ticker == ticker
            and order.direction == direction
            and order.status in ("pending", "submitted", "partial")
        ):
            raise SafetyError(
                f"Duplicate order detected: {direction} {ticker} "
                f"already has a pending order (id={order.id})."
            )


def force_close_check(now: datetime | None = None) -> bool:
    """Return ``True`` if it is at or past the forced-close time.

    All positions should be closed before market close to ensure
    day-trade completion.
    """
    if now is None:
        now = datetime.now()

    fc_h, fc_m = map(int, settings.FORCE_CLOSE_TIME.split(":"))
    force_close_time = time(fc_h, fc_m)

    return now.time() >= force_close_time


# ---------------------------------------------------------------------------
# SafetyGuard facade
# ---------------------------------------------------------------------------


class SafetyGuard:
    """Convenience wrapper that runs all pre-trade safety checks at once.

    Usage::

        guard = SafetyGuard()
        guard.pre_trade_check(
            current_pnl=-12000,
            current_position_count=2,
            ticker="7203",
            direction="long",
            existing_orders=orders,
        )
    """

    def pre_trade_check(
        self,
        current_pnl: float,
        current_position_count: int,
        ticker: str,
        direction: str,
        existing_orders: Sequence[Order] | None = None,
        now: datetime | None = None,
    ) -> None:
        """Run every safety gate.  Raises ``SafetyError`` on the first failure."""
        check_live_trading_blocked()
        check_daily_loss_limit(current_pnl)
        check_max_positions(current_position_count)

        if not check_market_hours(now):
            raise SafetyError(
                "Market is closed. Trading is only allowed during market hours "
                f"({settings.MARKET_OPEN}–{settings.MARKET_CLOSE} JST)."
            )

        if existing_orders is not None:
            check_duplicate_order(ticker, direction, existing_orders)

        if force_close_check(now):
            raise SafetyError(
                f"Past force-close time ({settings.FORCE_CLOSE_TIME}). "
                "New positions are not allowed; close all existing positions."
            )

    def can_open_position(
        self,
        current_pnl: float,
        current_position_count: int,
        ticker: str,
        direction: str,
        existing_orders: Sequence[Order] | None = None,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        """Non-raising variant — returns ``(ok, reason)``."""
        try:
            self.pre_trade_check(
                current_pnl=current_pnl,
                current_position_count=current_position_count,
                ticker=ticker,
                direction=direction,
                existing_orders=existing_orders,
                now=now,
            )
            return True, ""
        except SafetyError as exc:
            return False, str(exc)

    def check_anomalies(
        self,
        recent_trades: list,
        capital: float = 3_000_000,
    ) -> tuple[bool, str]:
        """Run all anomaly checks. Returns (should_halt, reason).

        保守的チューニング (2026-03-19):
        - 3連敗で停止 (4→3)
        - 直近8件で-20K超で停止 (金額ベース)
        """
        if check_consecutive_losses(recent_trades, max_consecutive=3):
            return True, "3連敗検知 — 一時停止"
        if len(recent_trades) >= 2:
            window_trades = recent_trades[-8:]
            window_pnl = sum(t.pnl for t in window_trades)
            if window_pnl < -20000:
                return True, f"急速ドローダウン検知 (直近{len(window_trades)}件で{window_pnl:+,.0f}円) — 一時停止"
        return False, ""

    @staticmethod
    def is_market_open(now: datetime | None = None) -> bool:
        return check_market_hours(now)

    @staticmethod
    def should_force_close(now: datetime | None = None) -> bool:
        return force_close_check(now)
