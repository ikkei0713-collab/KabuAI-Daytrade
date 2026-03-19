"""Unit tests for KabuAI-Daytrade safety guards.

Tests every safety check in ``core.safety``:
- Live trading block
- Daily loss limit
- Max positions
- Market hours check
- Duplicate order prevention
- Force close timing
"""

from __future__ import annotations

from datetime import datetime, time
from unittest.mock import patch

import pytest

from core.models import Order
from core.safety import (
    SafetyError,
    SafetyGuard,
    check_daily_loss_limit,
    check_duplicate_order,
    check_live_trading_blocked,
    check_market_hours,
    check_max_positions,
    force_close_check,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def guard() -> SafetyGuard:
    return SafetyGuard()


@pytest.fixture
def sample_orders() -> list[Order]:
    """Pre-built order list for duplicate checks."""
    return [
        Order(ticker="7203", direction="long", status="pending", strategy_name="momentum"),
        Order(ticker="6758", direction="short", status="submitted", strategy_name="reversal"),
        Order(ticker="9984", direction="long", status="filled", strategy_name="gap"),
    ]


# ------------------------------------------------------------------
# Live trading block
# ------------------------------------------------------------------


class TestLiveTradingBlock:
    """ALLOW_LIVE_TRADING must be False at all times."""

    @patch("core.safety.settings")
    def test_blocks_when_live_trading_enabled(self, mock_settings) -> None:
        mock_settings.ALLOW_LIVE_TRADING = True
        with pytest.raises(SafetyError, match="ALLOW_LIVE_TRADING is True"):
            check_live_trading_blocked()

    @patch("core.safety.settings")
    def test_passes_when_live_trading_disabled(self, mock_settings) -> None:
        mock_settings.ALLOW_LIVE_TRADING = False
        # Should not raise
        check_live_trading_blocked()


# ------------------------------------------------------------------
# Daily loss limit
# ------------------------------------------------------------------


class TestDailyLossLimit:
    """Trading must stop when daily loss limit is reached."""

    @patch("core.safety.settings")
    def test_blocks_at_loss_limit(self, mock_settings) -> None:
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        with pytest.raises(SafetyError, match="Daily loss limit"):
            check_daily_loss_limit(-50_000)

    @patch("core.safety.settings")
    def test_blocks_beyond_loss_limit(self, mock_settings) -> None:
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        with pytest.raises(SafetyError, match="Daily loss limit"):
            check_daily_loss_limit(-80_000)

    @patch("core.safety.settings")
    def test_passes_within_limit(self, mock_settings) -> None:
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        check_daily_loss_limit(-10_000)  # should not raise

    @patch("core.safety.settings")
    def test_passes_with_profit(self, mock_settings) -> None:
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        check_daily_loss_limit(100_000)  # should not raise

    @patch("core.safety.settings")
    def test_passes_at_zero(self, mock_settings) -> None:
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        check_daily_loss_limit(0)  # should not raise


# ------------------------------------------------------------------
# Max positions
# ------------------------------------------------------------------


class TestMaxPositions:
    """Cannot exceed the configured maximum number of positions."""

    @patch("core.safety.settings")
    def test_blocks_at_max(self, mock_settings) -> None:
        mock_settings.MAX_POSITIONS = 5
        with pytest.raises(SafetyError, match="Maximum positions"):
            check_max_positions(5)

    @patch("core.safety.settings")
    def test_blocks_above_max(self, mock_settings) -> None:
        mock_settings.MAX_POSITIONS = 5
        with pytest.raises(SafetyError, match="Maximum positions"):
            check_max_positions(7)

    @patch("core.safety.settings")
    def test_passes_below_max(self, mock_settings) -> None:
        mock_settings.MAX_POSITIONS = 5
        check_max_positions(3)  # should not raise

    @patch("core.safety.settings")
    def test_passes_at_zero(self, mock_settings) -> None:
        mock_settings.MAX_POSITIONS = 5
        check_max_positions(0)  # should not raise


# ------------------------------------------------------------------
# Market hours
# ------------------------------------------------------------------


class TestMarketHours:
    """Trading only allowed during market hours (09:00-15:00 JST)."""

    @patch("core.safety.settings")
    def test_market_open_during_trading_hours(self, mock_settings) -> None:
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        # 10:30 AM — market should be open
        t = datetime(2026, 3, 19, 10, 30)
        assert check_market_hours(t) is True

    @patch("core.safety.settings")
    def test_market_closed_before_open(self, mock_settings) -> None:
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        t = datetime(2026, 3, 19, 8, 30)
        assert check_market_hours(t) is False

    @patch("core.safety.settings")
    def test_market_closed_after_close(self, mock_settings) -> None:
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        t = datetime(2026, 3, 19, 15, 30)
        assert check_market_hours(t) is False

    @patch("core.safety.settings")
    def test_market_open_at_exact_open(self, mock_settings) -> None:
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        t = datetime(2026, 3, 19, 9, 0)
        assert check_market_hours(t) is True

    @patch("core.safety.settings")
    def test_market_open_at_exact_close(self, mock_settings) -> None:
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        t = datetime(2026, 3, 19, 15, 0)
        assert check_market_hours(t) is True

    @patch("core.safety.settings")
    def test_lunch_break_still_counts_as_open(self, mock_settings) -> None:
        """TSE has a lunch break 11:30-12:30 but our simple check
        doesn't model it — just open/close boundaries."""
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        t = datetime(2026, 3, 19, 12, 0)
        assert check_market_hours(t) is True


# ------------------------------------------------------------------
# Duplicate order prevention
# ------------------------------------------------------------------


class TestDuplicateOrder:
    """Prevent submitting an identical pending order."""

    def test_detects_duplicate_pending_order(self, sample_orders: list[Order]) -> None:
        with pytest.raises(SafetyError, match="Duplicate order"):
            check_duplicate_order("7203", "long", sample_orders)

    def test_detects_duplicate_submitted_order(self, sample_orders: list[Order]) -> None:
        with pytest.raises(SafetyError, match="Duplicate order"):
            check_duplicate_order("6758", "short", sample_orders)

    def test_allows_different_ticker(self, sample_orders: list[Order]) -> None:
        # 8306 has no existing order
        check_duplicate_order("8306", "long", sample_orders)  # should not raise

    def test_allows_different_direction(self, sample_orders: list[Order]) -> None:
        # 7203 has a long order, short should be fine
        check_duplicate_order("7203", "short", sample_orders)  # should not raise

    def test_allows_when_existing_order_filled(self, sample_orders: list[Order]) -> None:
        # 9984 has a "filled" order — not pending, so no duplicate
        check_duplicate_order("9984", "long", sample_orders)  # should not raise

    def test_allows_with_empty_orders(self) -> None:
        check_duplicate_order("7203", "long", [])  # should not raise


# ------------------------------------------------------------------
# Force close
# ------------------------------------------------------------------


class TestForceClose:
    """All positions must be closed before FORCE_CLOSE_TIME."""

    @patch("core.safety.settings")
    def test_force_close_at_threshold(self, mock_settings) -> None:
        mock_settings.FORCE_CLOSE_TIME = "14:50"
        t = datetime(2026, 3, 19, 14, 50)
        assert force_close_check(t) is True

    @patch("core.safety.settings")
    def test_force_close_after_threshold(self, mock_settings) -> None:
        mock_settings.FORCE_CLOSE_TIME = "14:50"
        t = datetime(2026, 3, 19, 14, 55)
        assert force_close_check(t) is True

    @patch("core.safety.settings")
    def test_no_force_close_before_threshold(self, mock_settings) -> None:
        mock_settings.FORCE_CLOSE_TIME = "14:50"
        t = datetime(2026, 3, 19, 14, 30)
        assert force_close_check(t) is False

    @patch("core.safety.settings")
    def test_no_force_close_morning(self, mock_settings) -> None:
        mock_settings.FORCE_CLOSE_TIME = "14:50"
        t = datetime(2026, 3, 19, 9, 30)
        assert force_close_check(t) is False


# ------------------------------------------------------------------
# SafetyGuard facade (pre_trade_check)
# ------------------------------------------------------------------


class TestSafetyGuardFacade:
    """Test the combined pre_trade_check and can_open_position methods."""

    @patch("core.safety.settings")
    def test_pre_trade_check_passes_all(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = False
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        mock_settings.MAX_POSITIONS = 5
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"
        mock_settings.FORCE_CLOSE_TIME = "14:50"

        # All conditions met: during trading hours, within limits
        t = datetime(2026, 3, 19, 10, 0)
        guard.pre_trade_check(
            current_pnl=5000,
            current_position_count=2,
            ticker="7203",
            direction="long",
            existing_orders=[],
            now=t,
        )

    @patch("core.safety.settings")
    def test_pre_trade_check_fails_on_live_trading(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = True

        with pytest.raises(SafetyError, match="ALLOW_LIVE_TRADING"):
            guard.pre_trade_check(
                current_pnl=0,
                current_position_count=0,
                ticker="7203",
                direction="long",
                now=datetime(2026, 3, 19, 10, 0),
            )

    @patch("core.safety.settings")
    def test_pre_trade_check_fails_on_loss_limit(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = False
        mock_settings.MAX_LOSS_PER_DAY = -50_000

        with pytest.raises(SafetyError, match="Daily loss limit"):
            guard.pre_trade_check(
                current_pnl=-60_000,
                current_position_count=0,
                ticker="7203",
                direction="long",
                now=datetime(2026, 3, 19, 10, 0),
            )

    @patch("core.safety.settings")
    def test_pre_trade_check_fails_on_closed_market(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = False
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        mock_settings.MAX_POSITIONS = 5
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        with pytest.raises(SafetyError, match="Market is closed"):
            guard.pre_trade_check(
                current_pnl=0,
                current_position_count=0,
                ticker="7203",
                direction="long",
                now=datetime(2026, 3, 19, 8, 0),
            )

    @patch("core.safety.settings")
    def test_pre_trade_check_fails_on_force_close(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = False
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        mock_settings.MAX_POSITIONS = 5
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"
        mock_settings.FORCE_CLOSE_TIME = "14:50"

        with pytest.raises(SafetyError, match="force-close"):
            guard.pre_trade_check(
                current_pnl=0,
                current_position_count=0,
                ticker="7203",
                direction="long",
                now=datetime(2026, 3, 19, 14, 55),
            )

    @patch("core.safety.settings")
    def test_can_open_position_returns_tuple(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = False
        mock_settings.MAX_LOSS_PER_DAY = -50_000
        mock_settings.MAX_POSITIONS = 5
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"
        mock_settings.FORCE_CLOSE_TIME = "14:50"

        t = datetime(2026, 3, 19, 10, 0)
        ok, reason = guard.can_open_position(
            current_pnl=0,
            current_position_count=2,
            ticker="7203",
            direction="long",
            existing_orders=[],
            now=t,
        )
        assert ok is True
        assert reason == ""

    @patch("core.safety.settings")
    def test_can_open_position_returns_failure_reason(self, mock_settings, guard: SafetyGuard) -> None:
        mock_settings.ALLOW_LIVE_TRADING = True

        ok, reason = guard.can_open_position(
            current_pnl=0,
            current_position_count=0,
            ticker="7203",
            direction="long",
        )
        assert ok is False
        assert "ALLOW_LIVE_TRADING" in reason

    @patch("core.safety.settings")
    def test_is_market_open_static(self, mock_settings) -> None:
        mock_settings.MARKET_OPEN = "09:00"
        mock_settings.MARKET_CLOSE = "15:00"

        assert SafetyGuard.is_market_open(datetime(2026, 3, 19, 10, 0)) is True
        assert SafetyGuard.is_market_open(datetime(2026, 3, 19, 8, 0)) is False

    @patch("core.safety.settings")
    def test_should_force_close_static(self, mock_settings) -> None:
        mock_settings.FORCE_CLOSE_TIME = "14:50"

        assert SafetyGuard.should_force_close(datetime(2026, 3, 19, 14, 50)) is True
        assert SafetyGuard.should_force_close(datetime(2026, 3, 19, 14, 30)) is False
