"""Backtesting engine for KabuAI-Daytrade strategies.

Simulates strategy performance against historical OHLCV data sourced
from J-Quants.  Tracks individual trades, calculates performance
metrics, and builds equity curves for single or multi-strategy runs.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from core.models import TradeSignal, Position, TradeResult, StrategyPerformance
from strategies.base import BaseStrategy
from tools.feature_engineering import FeatureEngineer


# ------------------------------------------------------------------
# Result containers
# ------------------------------------------------------------------


@dataclass
class BacktestTrade:
    """Record of a single simulated trade."""

    ticker: str
    strategy_name: str
    direction: str
    entry_price: float
    exit_price: float
    entry_date: str
    exit_date: str
    shares: int
    pnl: float
    pnl_pct: float
    holding_bars: int
    entry_reason: str = ""
    exit_reason: str = ""


@dataclass
class BacktestMetrics:
    """Aggregated performance metrics for a backtest run."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_holding_bars: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0


@dataclass
class BacktestResult:
    """Complete output of a backtest run."""

    strategy_name: str
    start_date: str
    end_date: str
    tickers: list[str]
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: BacktestMetrics = field(default_factory=BacktestMetrics)
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: list[float] = field(default_factory=list)


# ------------------------------------------------------------------
# Backtester
# ------------------------------------------------------------------


class Backtester:
    """Run backtests of KabuAI strategies against historical data.

    Args:
        db: Optional ``DatabaseManager`` instance for loading cached
            price data.  When absent, the caller must supply data
            directly via ``run_backtest()``.
        initial_capital: Starting capital in JPY.
        commission_rate: One-way commission as a fraction of notional
            (default 0 for SBI zero-commission stocks under 1M yen).
    """

    def __init__(
        self,
        db: Any = None,
        initial_capital: float = 3_000_000,
        commission_rate: float = 0.0,
    ) -> None:
        self.db = db
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self._feature_engineer = FeatureEngineer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_backtest(
        self,
        strategy: BaseStrategy,
        tickers: list[str],
        start_date: str,
        end_date: str,
        price_data: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        """Execute a full backtest for a single strategy.

        Args:
            strategy: Strategy instance to test.
            tickers: List of ticker codes (e.g. ``["7203", "6758"]``).
            start_date: Start date ``"YYYY-MM-DD"``.
            end_date: End date ``"YYYY-MM-DD"``.
            price_data: Optional pre-loaded OHLCV DataFrames keyed by
                ticker.  If not provided, data is loaded from J-Quants
                via the database layer.

        Returns:
            A ``BacktestResult`` with all trades, metrics, and the
            equity curve.
        """
        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
        )

        all_data = price_data or {}

        # Load data for tickers not already provided
        if self.db and not price_data:
            for ticker in tickers:
                df = await self._load_data(ticker, start_date, end_date)
                if df is not None and not df.empty:
                    all_data[ticker] = df

        if not all_data:
            logger.warning("No price data available for backtest")
            return result

        # Simulate across all tickers
        capital = self.initial_capital
        equity_curve = [capital]
        open_positions: dict[str, _SimPosition] = {}

        # Merge all dates across tickers for day-by-day simulation
        all_dates = sorted(
            {
                d
                for df in all_data.values()
                for d in (df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)))
            }
        )

        for bar_idx in range(50, len(all_dates)):
            # Process exits first
            tickers_to_close: list[str] = []
            for ticker, pos in open_positions.items():
                if ticker not in all_data:
                    continue
                df = all_data[ticker]
                if bar_idx >= len(df):
                    continue

                window = df.iloc[max(0, bar_idx - 50) : bar_idx + 1]
                features = self._feature_engineer.calculate_all_features(window)

                mock_position = Position(
                    ticker=ticker,
                    strategy_name=strategy.name,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    entry_time=datetime.now(),
                    current_price=float(df.iloc[bar_idx]["close"]) if "close" in df.columns else float(df.iloc[bar_idx].get("Close", 0)),
                    stop_loss=pos.stop_loss,
                    take_profit=pos.take_profit,
                )

                should_exit, exit_reason = strategy.should_exit(
                    mock_position, window, features
                )

                current_close = self._get_close(df, bar_idx)

                # Check stop loss / take profit
                if not should_exit:
                    if pos.direction == "long":
                        if current_close <= pos.stop_loss:
                            should_exit, exit_reason = True, "stop_loss"
                        elif current_close >= pos.take_profit:
                            should_exit, exit_reason = True, "take_profit"
                    else:
                        if current_close >= pos.stop_loss:
                            should_exit, exit_reason = True, "stop_loss"
                        elif current_close <= pos.take_profit:
                            should_exit, exit_reason = True, "take_profit"

                if should_exit:
                    exit_price = current_close
                    if pos.direction == "long":
                        pnl = (exit_price - pos.entry_price) * pos.shares
                    else:
                        pnl = (pos.entry_price - exit_price) * pos.shares

                    # Commission
                    commission = (pos.entry_price * pos.shares + exit_price * pos.shares) * self.commission_rate
                    pnl -= commission

                    pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.entry_price * pos.shares > 0 else 0.0

                    trade = BacktestTrade(
                        ticker=ticker,
                        strategy_name=strategy.name,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        entry_date=pos.entry_date,
                        exit_date=str(bar_idx),
                        shares=pos.shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        holding_bars=bar_idx - pos.entry_bar,
                        entry_reason=pos.entry_reason,
                        exit_reason=exit_reason,
                    )
                    result.trades.append(trade)
                    capital += pnl
                    tickers_to_close.append(ticker)

            for t in tickers_to_close:
                del open_positions[t]

            # Process entries
            for ticker in tickers:
                if ticker in open_positions:
                    continue
                if ticker not in all_data:
                    continue
                df = all_data[ticker]
                if bar_idx >= len(df):
                    continue

                window = df.iloc[max(0, bar_idx - 50) : bar_idx + 1]
                features = self._feature_engineer.calculate_all_features(window)

                signal = await strategy.scan(ticker, window, features)
                if signal is not None and signal.confidence >= 0.5:
                    entry_price = self._get_close(df, bar_idx)
                    atr = features.get("atr_14", entry_price * 0.02) or entry_price * 0.02
                    shares = strategy.calculate_position_size(entry_price, atr, capital)
                    notional = entry_price * shares

                    if notional > capital * 0.3:
                        continue  # skip if too large relative to remaining capital

                    open_positions[ticker] = _SimPosition(
                        ticker=ticker,
                        direction=signal.direction,
                        entry_price=entry_price,
                        entry_date=str(bar_idx),
                        entry_bar=bar_idx,
                        shares=shares,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        entry_reason=signal.entry_reason,
                    )

            equity_curve.append(capital)

        # Force-close any remaining open positions at the last bar
        for ticker, pos in open_positions.items():
            df = all_data.get(ticker)
            if df is None:
                continue
            exit_price = self._get_close(df, len(df) - 1)
            if pos.direction == "long":
                pnl = (exit_price - pos.entry_price) * pos.shares
            else:
                pnl = (pos.entry_price - exit_price) * pos.shares
            commission = (pos.entry_price * pos.shares + exit_price * pos.shares) * self.commission_rate
            pnl -= commission
            pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.entry_price * pos.shares > 0 else 0.0

            result.trades.append(
                BacktestTrade(
                    ticker=ticker,
                    strategy_name=strategy.name,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    entry_date=pos.entry_date,
                    exit_date="end",
                    shares=pos.shares,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    holding_bars=len(df) - 1 - pos.entry_bar,
                    entry_reason=pos.entry_reason,
                    exit_reason="backtest_end",
                )
            )
            capital += pnl

        equity_curve.append(capital)
        result.equity_curve = equity_curve
        result.metrics = self._calculate_metrics(result.trades, equity_curve)

        logger.info(
            "Backtest complete: {} | {} trades | PnL ¥{:,.0f} | WR {:.1%}",
            strategy.name,
            result.metrics.total_trades,
            result.metrics.total_pnl,
            result.metrics.win_rate,
        )

        return result

    async def compare_strategies(
        self,
        strategies: list[BaseStrategy],
        tickers: list[str],
        start_date: str,
        end_date: str,
        price_data: dict[str, pd.DataFrame] | None = None,
    ) -> list[BacktestResult]:
        """Run backtests for multiple strategies and return results for comparison.

        Args:
            strategies: List of strategy instances to compare.
            tickers: Shared ticker list.
            start_date: Start date.
            end_date: End date.
            price_data: Optional shared price data dict.

        Returns:
            List of ``BacktestResult``, one per strategy.
        """
        results: list[BacktestResult] = []
        for strategy in strategies:
            result = await self.run_backtest(
                strategy=strategy,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                price_data=price_data,
            )
            results.append(result)

        # Log comparison summary
        logger.info("=== Strategy Comparison ===")
        for r in sorted(results, key=lambda x: x.metrics.total_pnl, reverse=True):
            logger.info(
                "  {:<25s} PnL=¥{:>10,.0f}  WR={:.1%}  Trades={:>3d}  Sharpe={:.2f}",
                r.strategy_name,
                r.metrics.total_pnl,
                r.metrics.win_rate,
                r.metrics.total_trades,
                r.metrics.sharpe_ratio,
            )

        return results

    async def run_interactive(self) -> None:
        """Interactive CLI backtest session.

        Prompts the user for tickers, date range, and strategy,
        then runs the backtest and prints a summary.
        """
        print("\n=== KabuAI Backtester ===")
        print("Enter tickers (comma-separated, e.g. 7203,6758,9984):")
        tickers_input = input("> ").strip()
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

        if not tickers:
            print("No tickers provided. Exiting.")
            return

        print("Start date (YYYY-MM-DD):")
        start = input("> ").strip()
        print("End date (YYYY-MM-DD):")
        end = input("> ").strip()

        # Import strategies dynamically
        from strategies.registry import StrategyRegistry

        available = StrategyRegistry.list_strategies()
        if not available:
            print("No strategies registered. Run StrategyRegistry.register_all_defaults() first.")
            return

        print(f"\nAvailable strategies: {', '.join(available)}")
        print("Enter strategy name (or 'all' to compare):")
        choice = input("> ").strip()

        if choice.lower() == "all":
            strategies = [StrategyRegistry.get(name) for name in available]
            strategies = [s for s in strategies if s is not None]
            results = await self.compare_strategies(strategies, tickers, start, end)
            for r in results:
                self._print_result(r)
        else:
            strategy = StrategyRegistry.get(choice)
            if strategy is None:
                print(f"Strategy '{choice}' not found.")
                return
            result = await self.run_backtest(strategy, tickers, start, end)
            self._print_result(result)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    async def _load_data(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """Load historical OHLCV from the database / J-Quants."""
        try:
            from data_sources.jquants import JQuantsClient

            async with JQuantsClient() as client:
                raw = await client.get_prices_daily(ticker, start_date, end_date)

            if not raw:
                return None

            df = pd.DataFrame(raw)
            rename_map = {
                "AdjustmentOpen": "open",
                "AdjustmentHigh": "high",
                "AdjustmentLow": "low",
                "AdjustmentClose": "close",
                "AdjustmentVolume": "volume",
            }
            # Fallback to unadjusted if adjusted columns missing
            for adj, target in rename_map.items():
                if adj not in df.columns:
                    rename_map[adj.replace("Adjustment", "")] = target

            df.rename(columns=rename_map, inplace=True)
            for col in ("open", "high", "low", "close", "volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
                df.sort_index(inplace=True)

            return df

        except Exception as e:
            logger.error("Failed to load data for {}: {}", ticker, e)
            return None

    # ------------------------------------------------------------------
    # Metrics calculation
    # ------------------------------------------------------------------

    def _calculate_metrics(
        self, trades: list[BacktestTrade], equity_curve: list[float]
    ) -> BacktestMetrics:
        """Compute aggregate performance metrics from a trade list."""
        m = BacktestMetrics()

        if not trades:
            return m

        pnls = [t.pnl for t in trades]
        m.total_trades = len(trades)
        m.winning_trades = sum(1 for p in pnls if p > 0)
        m.losing_trades = sum(1 for p in pnls if p <= 0)
        m.win_rate = m.winning_trades / m.total_trades if m.total_trades > 0 else 0.0
        m.total_pnl = sum(pnls)
        m.avg_pnl = np.mean(pnls) if pnls else 0.0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        m.avg_win = float(np.mean(wins)) if wins else 0.0
        m.avg_loss = float(np.mean(losses)) if losses else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

        m.best_trade = max(pnls) if pnls else 0.0
        m.worst_trade = min(pnls) if pnls else 0.0

        m.avg_holding_bars = float(np.mean([t.holding_bars for t in trades])) if trades else 0.0

        # Drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0.0
            max_dd_pct = 0.0
            for val in equity_curve:
                if val > peak:
                    peak = val
                dd = peak - val
                dd_pct = dd / peak if peak > 0 else 0.0
                if dd > max_dd:
                    max_dd = dd
                if dd_pct > max_dd_pct:
                    max_dd_pct = dd_pct
            m.max_drawdown = max_dd
            m.max_drawdown_pct = max_dd_pct

        # Sharpe ratio (annualised, assuming ~245 trading days)
        if len(pnls) > 1:
            returns = np.array(pnls)
            m.sharpe_ratio = float(
                np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(245)
            ) if np.std(returns, ddof=1) > 0 else 0.0

        return m

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_close(df: pd.DataFrame, idx: int) -> float:
        """Get the close price at the given integer index."""
        row = df.iloc[idx]
        for col in ("close", "Close", "AdjustmentClose"):
            if col in df.columns:
                return float(row[col])
        return float(row.iloc[3])  # fallback to 4th column

    @staticmethod
    def _print_result(result: BacktestResult) -> None:
        """Pretty-print a backtest result to the console."""
        m = result.metrics
        print(f"\n{'=' * 50}")
        print(f"Strategy: {result.strategy_name}")
        print(f"Period:   {result.start_date} -> {result.end_date}")
        print(f"Tickers:  {', '.join(result.tickers)}")
        print(f"{'=' * 50}")
        print(f"Total Trades:    {m.total_trades}")
        print(f"Win Rate:        {m.win_rate:.1%}")
        print(f"Total PnL:       ¥{m.total_pnl:,.0f}")
        print(f"Avg PnL/Trade:   ¥{m.avg_pnl:,.0f}")
        print(f"Avg Win:         ¥{m.avg_win:,.0f}")
        print(f"Avg Loss:        ¥{m.avg_loss:,.0f}")
        print(f"Profit Factor:   {m.profit_factor:.2f}")
        print(f"Sharpe Ratio:    {m.sharpe_ratio:.2f}")
        print(f"Max Drawdown:    ¥{m.max_drawdown:,.0f} ({m.max_drawdown_pct:.1%})")
        print(f"Avg Holding:     {m.avg_holding_bars:.1f} bars")
        print(f"Best Trade:      ¥{m.best_trade:,.0f}")
        print(f"Worst Trade:     ¥{m.worst_trade:,.0f}")
        print(f"{'=' * 50}\n")


# ------------------------------------------------------------------
# Internal simulation position
# ------------------------------------------------------------------


@dataclass
class _SimPosition:
    """Internal position tracker for the simulator."""

    ticker: str
    direction: str
    entry_price: float
    entry_date: str
    entry_bar: int
    shares: int
    stop_loss: float
    take_profit: float
    entry_reason: str = ""
