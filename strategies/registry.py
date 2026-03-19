"""Strategy Registry – central catalogue of all available strategies.

Provides class-level methods to register, retrieve, and query
strategies.  ``register_all_defaults`` wires up every built-in
strategy with its default configuration.

戦略 status 体系:
- active:     単独エントリー可能。主戦略。
- filter:     他戦略の発火条件として使用。単独発火しない。
- supplement: confidence boost 専用。単独発火しない。
- watch:      将来用に残すが現時点では運用しない。
- off:        完全停止。proxy 依存が強い / OOS 信頼性不足。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from strategies.base import BaseStrategy


class StrategyRegistry:
    """Singleton-style registry for all trading strategies."""

    _strategies: dict[str, BaseStrategy] = {}

    # 戦略 status 定義: active/filter/supplement/watch/off
    STRATEGY_STATUS: dict[str, str] = {
        "vwap_reclaim":        "active",       # 唯一の主戦略
        "trend_follow":        "filter",       # is_trending() のみ使用
        "spread_entry":        "supplement",   # get_spread_boost() のみ使用
        "orb":                 "watch",        # intraday proxy 信頼性不足
        "tdnet_event":         "watch",        # イベント時参考
        "gap_go":              "watch",        # 擬似特徴量依存
        "vwap_bounce":         "off",          # データ不足
        "orderbook_imbalance": "off",          # ダミーデータ依存
        "large_absorption":    "off",          # ダミーデータ依存
        "open_drive":          "off",          # PF 0.23
        "gap_fade":            "off",          # 逆張り系リスク高
        "overextension":       "off",          # 擬似 intraday 依存
        "rsi_reversal":        "off",          # 擬似 intraday 依存
        "crash_rebound":       "off",          # 擬似 intraday 依存
        "earnings_momentum":   "off",          # データ不足
        "catalyst_initial":    "off",          # データ不足
    }

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

        # 保守的チューニング (2026-03-19): vwap_reclaim 主軸運用
        # 擬似intraday依存が強い戦略、OOS信頼性が低い戦略を全面停止
        # trend_follow: filter専用 (is_trending() のみ使用、単独発火しない)
        # spread_entry: supplement専用 (get_spread_boost() のみ使用)
        disabled = {
            "orderbook_imbalance",  # ダミーデータ依存
            "large_absorption",     # ダミーデータ依存
            "open_drive",           # PF 0.23
            "orb",                  # watch降格: intraday proxy信頼性不足
            "gap_go",              # watch降格: 擬似特徴量依存
            "gap_fade",            # off: 逆張り系リスク高
            "vwap_bounce",         # off: データ不足 (3件)
            "overextension",       # off: 擬似intraday依存
            "rsi_reversal",        # off: 擬似intraday依存
            "crash_rebound",       # off: 擬似intraday依存
            "trend_follow",        # off (filter専用: is_trending() は static で呼べる)
            "spread_entry",        # off (supplement専用: get_spread_boost() は static)
            "earnings_momentum",   # off: データ不足
            "catalyst_initial",    # off: データ不足
            "tdnet_event",         # watch: 単独発火しない (イベント情報はfeatures経由で供給)
        }

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
    def get_status(cls, name: str) -> str:
        """Return the status of a strategy (active/filter/supplement/watch/off)."""
        return cls.STRATEGY_STATUS.get(name, "off")

    @classmethod
    def get_status_summary(cls) -> dict[str, list[str]]:
        """Return strategies grouped by status."""
        groups: dict[str, list[str]] = {
            "active": [], "filter": [], "supplement": [], "watch": [], "off": [],
        }
        for name, status in cls.STRATEGY_STATUS.items():
            groups.setdefault(status, []).append(name)
        return groups

    @classmethod
    def get_proxy_summary(cls) -> dict[str, dict]:
        """Return proxy_usage_rate and penalty for all strategies."""
        from tools.feature_engineering import FeatureEngineer
        result = {}
        for name in cls.STRATEGY_STATUS:
            rate = FeatureEngineer.get_proxy_usage_rate(name)
            penalty = FeatureEngineer.get_proxy_penalty(name)
            result[name] = {
                "status": cls.STRATEGY_STATUS.get(name, "off"),
                "proxy_usage_rate": rate,
                "proxy_penalty": penalty,
                "proxy_features": FeatureEngineer.STRATEGY_PROXY_DEPS.get(name, []),
            }
        return result

    # 恒久停止リスト（auto_toggleで再開しない）
    _PERMANENTLY_DISABLED = {
        "orderbook_imbalance", "large_absorption", "open_drive",
        "gap_fade", "overextension", "rsi_reversal", "crash_rebound",
        "vwap_bounce", "earnings_momentum", "catalyst_initial",
        "trend_follow", "spread_entry",  # filter/supplement専用
    }

    @classmethod
    def auto_toggle(cls, recent_trades: list, min_trades: int = 12) -> list[str]:
        """Auto-disable strategies with poor rolling performance.

        閾値 (保守的チューニング 2026-03-19):
        - off: PF < 0.90 or WR < 40% or avg_pnl < 0
        - on:  PF > 1.20 and WR > 48% (再開は15件以上)
        """
        toggled: list[str] = []
        for name, strategy in cls._strategies.items():
            if name in cls._PERMANENTLY_DISABLED:
                continue

            strades = [t for t in recent_trades if t.strategy_name == name]

            if len(strades) < min_trades:
                continue

            gp = sum(t.pnl for t in strades if t.pnl > 0)
            gl = abs(sum(t.pnl for t in strades if t.pnl <= 0))
            pf = gp / gl if gl > 0 else 99.0
            wr = sum(1 for t in strades if t.pnl > 0) / len(strades)
            avg_pnl = sum(t.pnl for t in strades) / len(strades)

            was_active = strategy.config.is_active

            # 停止条件: PF < 0.90 or WR < 40% or 平均損益マイナス
            if pf < 0.90 or wr < 0.40 or avg_pnl < 0:
                if was_active:
                    strategy.config.is_active = False
                    toggled.append(name)
                    logger.info(
                        f"[auto_toggle] {name} 停止: PF={pf:.2f} 勝率={wr:.0%} "
                        f"avg={avg_pnl:+,.0f} ({len(strades)}件)"
                    )
            # 再開条件: PF > 1.20 and WR > 48% (厳しめ、15件以上)
            elif not was_active and pf > 1.20 and wr > 0.48 and len(strades) >= 15:
                strategy.config.is_active = True
                toggled.append(name)
                logger.info(
                    f"[auto_toggle] {name} 再開: PF={pf:.2f} 勝率={wr:.0%}"
                )

        return toggled
