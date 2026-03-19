"""Strategy Registry – central catalogue of all available strategies.

Provides class-level methods to register, retrieve, and query
strategies.  ``register_all_defaults`` wires up every built-in
strategy with its default configuration.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from strategies.base import BaseStrategy


class StrategyRegistry:
    """Singleton-style registry for all trading strategies."""

    _strategies: dict[str, BaseStrategy] = {}

    @classmethod
    def register(cls, strategy: BaseStrategy) -> None:
        """Register a strategy instance by its name."""
        cls._strategies[strategy.name] = strategy
        logger.debug(f"[registry] Registered strategy: {strategy.name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a strategy from the registry."""
        if name in cls._strategies:
            del cls._strategies[name]
            logger.debug(f"[registry] Unregistered strategy: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[BaseStrategy]:
        """Get a strategy by name. Returns None if not found."""
        strategy = cls._strategies.get(name)
        if strategy is None:
            logger.warning(f"[registry] Strategy not found: {name}")
        return strategy

    @classmethod
    def get_all(cls) -> list[BaseStrategy]:
        """Return all registered strategies."""
        return list(cls._strategies.values())

    @classmethod
    def get_active(cls) -> list[BaseStrategy]:
        """Return only strategies whose config has is_active=True."""
        return [s for s in cls._strategies.values() if s.config.is_active]

    @classmethod
    def get_names(cls) -> list[str]:
        """Return the names of all registered strategies."""
        return list(cls._strategies.keys())

    @classmethod
    def clear(cls) -> None:
        """Remove all registered strategies (useful for testing)."""
        cls._strategies.clear()
        logger.debug("[registry] All strategies cleared")

    @classmethod
    def register_all_defaults(cls) -> None:
        """Instantiate and register every built-in strategy with its
        default configuration.

        This is the main entry-point called during application startup.
        """
        # --- Gap strategies ---
        from strategies.gap.gap_go import GapGoStrategy
        from strategies.gap.gap_fade import GapFadeStrategy

        # --- Opening strategies ---
        from strategies.opening.open_drive import OpenDriveStrategy
        from strategies.opening.orb import ORBStrategy

        # --- Momentum strategies ---
        from strategies.momentum.vwap_reclaim import VWAPReclaimStrategy
        from strategies.momentum.vwap_bounce import VWAPBounceStrategy
        from strategies.momentum.trend_follow import TrendFollowStrategy

        # --- Reversal strategies ---
        from strategies.reversal.overextension import OverextensionStrategy
        from strategies.reversal.rsi_reversal import RSIReversalStrategy
        from strategies.reversal.crash_rebound import CrashReboundStrategy

        # --- Orderbook strategies ---
        from strategies.orderbook.imbalance import ImbalanceStrategy
        from strategies.orderbook.large_absorption import LargeAbsorptionStrategy
        from strategies.orderbook.spread_entry import SpreadEntryStrategy

        # --- Event strategies ---
        from strategies.event.tdnet_event import TDnetEventStrategy
        from strategies.event.earnings_momentum import EarningsMomentumStrategy
        from strategies.event.catalyst_initial import CatalystInitialStrategy

        default_strategies: list[BaseStrategy] = [
            GapGoStrategy(),
            GapFadeStrategy(),
            OpenDriveStrategy(),
            ORBStrategy(),
            VWAPReclaimStrategy(),
            VWAPBounceStrategy(),
            TrendFollowStrategy(),
            OverextensionStrategy(),
            RSIReversalStrategy(),
            CrashReboundStrategy(),
            ImbalanceStrategy(),
            LargeAbsorptionStrategy(),
            SpreadEntryStrategy(),
            TDnetEventStrategy(),
            EarningsMomentumStrategy(),
            CatalystInitialStrategy(),
        ]

        # バックテスト実績に基づく無効化リスト
        # orderbook系: ダミーデータ依存で信頼性ゼロ
        # open_drive: 勝率35%, PF 0.23
        disabled = {"orderbook_imbalance", "large_absorption", "open_drive"}

        for strategy in default_strategies:
            if strategy.name in disabled:
                strategy.config.is_active = False
            cls.register(strategy)

        active = [s.name for s in default_strategies if s.config.is_active]
        logger.info(
            f"[registry] Registered {len(default_strategies)} strategies "
            f"({len(active)} active): {active}"
        )

    @classmethod
    def auto_toggle(cls, recent_trades: list, min_trades: int = 8) -> list[str]:
        """Auto-disable strategies with poor rolling performance.

        Returns list of strategy names that were toggled off.
        Strategies are re-enabled if they were disabled by auto_toggle
        and their performance has recovered.
        """
        from core.safety import check_strategy_degradation

        toggled: list[str] = []
        for name, strategy in cls._strategies.items():
            if name in {"orderbook_imbalance", "large_absorption", "open_drive"}:
                continue  # permanently disabled

            strades = [t for t in recent_trades if t.strategy_name == name]

            if len(strades) < min_trades:
                continue

            # Calculate rolling PF
            gp = sum(t.pnl for t in strades if t.pnl > 0)
            gl = abs(sum(t.pnl for t in strades if t.pnl <= 0))
            pf = gp / gl if gl > 0 else 99.0
            wr = sum(1 for t in strades if t.pnl > 0) / len(strades)

            was_active = strategy.config.is_active

            # Disable if PF < 0.7 or win rate < 30%
            if pf < 0.7 or wr < 0.30:
                if was_active:
                    strategy.config.is_active = False
                    toggled.append(name)
                    logger.info(
                        f"[auto_toggle] {name} 停止: PF={pf:.2f} 勝率={wr:.0%} "
                        f"({len(strades)}件)"
                    )
            # Re-enable if PF > 1.0 and win rate > 45%
            elif not was_active and pf > 1.0 and wr > 0.45:
                strategy.config.is_active = True
                toggled.append(name)
                logger.info(
                    f"[auto_toggle] {name} 再開: PF={pf:.2f} 勝率={wr:.0%}"
                )

        return toggled
