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

        for strategy in default_strategies:
            cls.register(strategy)

        logger.info(
            f"[registry] Registered {len(default_strategies)} default strategies: "
            f"{[s.name for s in default_strategies]}"
        )
