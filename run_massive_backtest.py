"""
大規模バックテスト: 全戦略 × パラメータ × レジームフィルタを網羅テスト

テストパターン:
1. 全16戦略を個別にテスト（デフォルトパラメータ）
2. 各戦略をパラメータ変化させてテスト
3. 有望戦略の組み合わせテスト
4. レジームフィルタON/OFFテスト
5. 収束フィルタパラメータ探索

1時間で可能な限り多くのパターンを回す。
"""

import asyncio
import copy
import json
import random
import sys
import time
import itertools
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from core.models import TradeResult
from data_sources.jquants import JQuantsClient
from data_sources.tdnet import TDnetClient
from strategies.registry import StrategyRegistry
from strategies.base import BaseStrategy
from tools.feature_engineering import FeatureEngineer
from tools.cost_model import CostModel
from tools.market_regime import RegimeDetector
from scanners.stock_selector import StockSelector
from core.ticker_map import update_from_jquants
from run_backtest_learn import CANDIDATE_CODES, _clean_features

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/massive_backtest.log", rotation="10 MB", level="DEBUG")

DURATION_MINUTES = 55  # データ取得に5分使うので55分

# 全戦略のパラメータ探索空間
ALL_PARAM_SPACE = {
    "vwap_reclaim": {
        "min_time_below_vwap_min": [5, 8, 10, 15, 20, 25, 30],
        "min_volume_at_reclaim": [0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
        "target_atr_multiple": [0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5],
        "reclaim_buffer_pct": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "max_distance_from_vwap_pct": [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0],
    },
    "gap_go": {
        "min_gap_pct": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "min_volume_ratio": [1.0, 1.2, 1.5, 2.0, 2.5],
        "target_atr_multiple": [1.0, 1.5, 2.0, 2.5, 3.0],
        "max_gap_pct": [5.0, 8.0, 10.0, 15.0],
    },
    "gap_fade": {
        "min_gap_pct": [2.0, 3.0, 4.0, 5.0],
        "max_gap_pct": [8.0, 10.0, 15.0, 20.0],
        "rsi_exhaustion": [70.0, 75.0, 80.0, 85.0],
        "min_volume_ratio": [1.0, 1.2, 1.5, 2.0],
    },
    "orb": {
        "min_volume_ratio": [1.0, 1.2, 1.5, 2.0, 2.5],
        "min_atr": [5.0, 8.0, 10.0, 15.0, 20.0],
        "target_range_multiple": [1.0, 1.2, 1.5, 2.0],
        "atr_stop_multiple": [1.0, 1.5, 2.0],
        "atr_target_multiple": [1.5, 2.0, 2.5, 3.0],
        "min_confidence": [0.35, 0.40, 0.45, 0.50, 0.55],
    },
    "vwap_bounce": {
        "min_vwap_touches": [1, 2, 3],
        "trend_direction_min": [0.2, 0.3, 0.4, 0.5],
        "min_volume_ratio": [1.0, 1.2, 1.5],
        "target_atr_multiple": [0.8, 1.0, 1.2, 1.5],
        "stop_atr_below_vwap": [0.3, 0.5, 0.8, 1.0],
    },
    "trend_follow": {
        "min_trend_strength": [0.1, 0.15, 0.2, 0.3, 0.4],
        "min_volume_trend": [0.8, 1.0, 1.2, 1.5],
        "trailing_atr_multiple": [1.0, 1.2, 1.5, 2.0],
    },
    "overextension": {
        "min_atr_distance": [2.0, 2.5, 3.0, 4.0],
        "rsi_overbought": [75, 80, 85],
        "rsi_oversold": [15, 20, 25],
        "min_volume_climax": [1.5, 2.0, 2.5, 3.0],
    },
    "rsi_reversal": {
        "rsi_oversold": [10, 15, 20, 25],
        "rsi_overbought": [75, 80, 85, 90],
        "min_volume_spike": [1.5, 2.0, 2.5, 3.0],
        "stop_atr_multiple": [0.5, 0.8, 1.0, 1.5],
    },
    "crash_rebound": {
        "min_drop_pct": [3.0, 4.0, 5.0, 7.0],
        "max_drop_pct": [10.0, 15.0, 20.0, 25.0],
        "min_volume_surge": [2.0, 2.5, 3.0, 4.0],
        "retracement_target": [0.30, 0.40, 0.50, 0.60],
    },
    "spread_entry": {
        "spread_percentile_max": [10, 15, 20, 25, 30],
        "min_volume_building": [1.2, 1.5, 1.8, 2.0, 2.5],
        "min_price_compression": [0.3, 0.4, 0.5, 0.6],
        "target_atr_multiple": [1.0, 1.2, 1.5, 2.0],
    },
    "tdnet_event": {
        "min_event_magnitude": [0.1, 0.2, 0.3, 0.5],
        "min_historical_response": [0.0, 0.1, 0.2],
    },
    "open_drive": {
        # ORB系パラメータ
        "min_volume_ratio": [1.0, 1.5, 2.0, 2.5],
    },
    "orderbook_imbalance": {
        "long_ratio_threshold": [1.5, 2.0, 2.5, 3.0],
        "min_depth_imbalance": [0.2, 0.3, 0.4, 0.5],
        "target_atr_multiple": [0.8, 1.0, 1.2, 1.5],
    },
    "large_absorption": {
        "min_volume_price_divergence": [1.5, 2.0, 2.5, 3.0],
        "target_atr_multiple": [0.8, 1.0, 1.2, 1.5],
    },
    "earnings_momentum": {
        "min_earnings_surprise_pct": [3.0, 5.0, 7.0, 10.0],
        "min_volume_ratio": [1.0, 1.5, 2.0],
    },
    "catalyst_initial": {
        "min_news_sentiment": [0.3, 0.4, 0.5, 0.6],
        "min_volume_surge": [1.5, 2.0, 2.5, 3.0],
    },
}

# 収束フィルタパラメータ空間
CONVERGENCE_SPACE = {
    "MAX_MA_SPREAD_PCT_FOR_ENTRY": [0.008, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040],
    "MIN_MA_CONVERGENCE_SCORE": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "MIN_RANGE_COMPRESSION_SCORE": [0.20, 0.30, 0.40, 0.50, 0.55],
    "MIN_VOLATILITY_COMPRESSION_SCORE": [0.25, 0.30, 0.35, 0.40, 0.50],
    "CONVERGENCE_CONFIDENCE_BOOST": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15],
    "EXPANSION_PENALTY_AFTER_CROSS": [0.05, 0.08, 0.10, 0.12, 0.15, 0.20],
}

# レジームフィルタ設定のバリエーション
REGIME_FILTER_CONFIGS = [
    {"name": "no_filter", "blocked_regimes": set()},
    {"name": "block_trend_down", "blocked_regimes": {"trend_down"}},
    {"name": "block_volatile", "blocked_regimes": {"volatile"}},
    {"name": "block_down_vol", "blocked_regimes": {"trend_down", "volatile"}},
    {"name": "only_trend_up", "blocked_regimes": {"trend_down", "volatile", "range", "low_vol"}},
    {"name": "only_trend_up_range", "blocked_regimes": {"trend_down", "volatile", "low_vol"}},
    {"name": "no_low_vol", "blocked_regimes": {"low_vol"}},
    {"name": "block_down_lowvol", "blocked_regimes": {"trend_down", "low_vol"}},
]


def calc_metrics(trades: list[TradeResult]) -> dict:
    if not trades:
        return {"total": 0, "wins": 0, "win_rate": 0, "pf": 0, "pnl": 0,
                "avg": 0, "fit3": 0, "max_win": 0, "max_loss": 0, "sharpe": 0}
    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    pnl = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    wr = wins / total
    pf = gp / gl if gl > 0 else (99.0 if gp > 0 else 0)
    fit3 = -gl + 0.01 * gp
    pnls = [t.pnl for t in trades]
    sharpe = 0.0
    if len(pnls) > 1:
        std = np.std(pnls, ddof=1)
        if std > 0:
            sharpe = float(np.mean(pnls) / std * np.sqrt(245))
    return {
        "total": total, "wins": wins, "win_rate": round(wr, 3),
        "pf": round(pf, 3), "pnl": round(pnl, 0), "avg": round(pnl / total, 0),
        "fit3": round(fit3, 0),
        "max_win": round(max(pnls), 0), "max_loss": round(min(pnls), 0),
        "sharpe": round(sharpe, 2),
    }


class MassiveBacktester:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.cost_model = CostModel(commission_free=True)
        self.selector = StockSelector()
        self.regime_detector = RegimeDetector()
        self.stock_data: dict[str, pd.DataFrame] = {}
        self.tdnet_events: dict[date, dict[str, str]] = {}
        self.sim_dates: list[date] = []
        self.is_dates: set[date] = set()
        self.oos_dates: set[date] = set()
        self.all_results: list[dict] = []
        self.best_results: list[dict] = []  # Top N tracker

    async def load_data(self):
        """データを一回だけ取得"""
        logger.info("=" * 60)
        logger.info("データ取得開始...")

        # TDnet
        async with TDnetClient() as tdnet:
            d = date(2025, 9, 1)
            end = date(2026, 3, 24)
            while d <= end:
                if d.weekday() < 5:
                    try:
                        disclosures = await tdnet.fetch_today_disclosures(d)
                        material = tdnet.filter_material_events(disclosures)
                        for disc in material:
                            self.tdnet_events.setdefault(d, {})[disc.ticker + "0"] = disc.disclosure_type.value
                    except Exception:
                        pass
                d += timedelta(days=1)

        # J-Quants
        async with JQuantsClient() as client:
            try:
                master = await client.get_listed_info()
                update_from_jquants(master)
            except Exception:
                pass

            for code in CANDIDATE_CODES:
                try:
                    raw = await client.get_prices_daily(code, "2025-09-01", "2026-03-24")
                    if not raw:
                        continue
                    df = pd.DataFrame(raw)
                    df = df.rename(columns={
                        "AdjO": "open", "AdjH": "high", "AdjL": "low",
                        "AdjC": "close", "AdjVo": "volume", "Date": "Date",
                    })
                    for c in ["open", "high", "low", "close", "volume"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.sort_values("Date").reset_index(drop=True)
                    df = df[["Date", "open", "high", "low", "close", "volume"]].dropna()
                    if len(df) >= 30:
                        self.stock_data[code] = df
                except Exception:
                    pass

        logger.info(f"データ取得完了: {len(self.stock_data)}銘柄")

        # 日付リスト
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df["Date"].dt.date.tolist())
        self.sim_dates = sorted(all_dates)[30:]

        # IS/OOS 50/50 分割
        split = int(len(self.sim_dates) * 0.5)
        self.is_dates = set(self.sim_dates[:split])
        self.oos_dates = set(self.sim_dates[split:])
        logger.info(f"シミュレーション: {len(self.sim_dates)}日 (IS {len(self.is_dates)} / OOS {len(self.oos_dates)})")

    async def _run_backtest(
        self,
        strategies: list[BaseStrategy],
        blocked_regimes: set[str] | None = None,
        capital: float = 30_000,
    ) -> tuple[list[TradeResult], list[TradeResult]]:
        """IS/OOS分離バックテスト"""
        is_trades = []
        oos_trades = []
        blocked = blocked_regimes or set()

        market_proxy = max(self.stock_data.keys(), key=lambda c: len(self.stock_data[c]))

        for sim_date in self.sim_dates:
            mp_df = self.stock_data[market_proxy]
            mask = mp_df["Date"].dt.date <= sim_date
            mp_slice = mp_df[mask]
            regime = self.regime_detector.detect(mp_slice).regime if len(mp_slice) >= 50 else "range"

            if regime in blocked:
                continue

            bt_clock = datetime.combine(sim_date, datetime.min.time().replace(hour=13, minute=15))
            try:
                bt_clock = bt_clock.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
            except Exception:
                pass

            for code, full_df in self.stock_data.items():
                df = full_df[full_df["Date"].dt.date <= sim_date]
                if len(df) < 20:
                    continue

                next_rows = full_df[full_df["Date"].dt.date > sim_date]
                if next_rows.empty:
                    continue

                features = self.fe.calculate_all_features(df, clock=bt_clock)
                entry_price = float(df["close"].iloc[-1])
                features["current_price"] = entry_price
                atr = features.get("atr", entry_price * 0.02) or entry_price * 0.02

                # TDnet
                if self.tdnet_events and sim_date in self.tdnet_events:
                    evt = self.tdnet_events[sim_date].get(code, "")
                    if evt:
                        features["event_type"] = evt
                        features["event_magnitude"] = 1.0
                        features["historical_event_response"] = 0.5

                for strategy in strategies:
                    if not strategy.config.is_active:
                        continue
                    try:
                        signal = await strategy.scan(code, df, features)
                        if not signal or signal.confidence < 0.3:
                            continue

                        nr = next_rows.iloc[0]
                        nh, nl, nc = float(nr["high"]), float(nr["low"]), float(nr["close"])
                        qty = strategy.calculate_position_size(entry_price, atr, capital)
                        if qty <= 0:
                            continue

                        if signal.direction == "long":
                            if nl <= signal.stop_loss:
                                exit_p = signal.stop_loss
                                exit_reason = "ストップロス"
                            elif nh >= signal.take_profit:
                                exit_p = signal.take_profit
                                exit_reason = "利確"
                            else:
                                exit_p = nc
                                exit_reason = "翌日決済"
                            raw_pnl = (exit_p - entry_price) * qty
                        else:
                            if nh >= signal.stop_loss:
                                exit_p = signal.stop_loss
                                exit_reason = "ストップロス"
                            elif nl <= signal.take_profit:
                                exit_p = signal.take_profit
                                exit_reason = "利確"
                            else:
                                exit_p = nc
                                exit_reason = "翌日決済"
                            raw_pnl = (entry_price - exit_p) * qty

                        adj_e = self.cost_model.adjust_entry_price(entry_price, signal.direction)
                        adj_x = self.cost_model.adjust_exit_price(exit_p, signal.direction)
                        cost = self.cost_model.calculate_trade_cost(entry_price, exit_p, qty)

                        if signal.direction == "long":
                            pnl = (adj_x - adj_e) * qty - cost.total
                        else:
                            pnl = (adj_e - adj_x) * qty - cost.total

                        trade = TradeResult(
                            ticker=code,
                            strategy_name=strategy.name,
                            direction=signal.direction,
                            entry_price=entry_price,
                            exit_price=exit_p,
                            entry_time=datetime.combine(sim_date, datetime.min.time().replace(hour=9)),
                            exit_time=datetime.combine(sim_date + timedelta(days=1), datetime.min.time().replace(hour=15)),
                            pnl=round(pnl, 0),
                            pnl_pct=round(pnl / (entry_price * qty) * 100, 2) if qty > 0 else 0,
                            holding_minutes=360,
                            entry_reason=signal.entry_reason,
                            exit_reason=exit_reason,
                            market_condition=regime,
                        )

                        if sim_date in self.is_dates:
                            is_trades.append(trade)
                        else:
                            oos_trades.append(trade)

                    except Exception:
                        continue

        return is_trades, oos_trades

    def _record_result(self, test_type: str, description: str, strategies_used: list[str],
                       params: dict, is_m: dict, oos_m: dict, regime_filter: str = "none",
                       conv_params: dict | None = None):
        """結果を記録"""
        result = {
            "test_type": test_type,
            "description": description,
            "strategies": strategies_used,
            "params": params,
            "regime_filter": regime_filter,
            "convergence_params": conv_params,
            "is": is_m,
            "oos": oos_m,
        }
        self.all_results.append(result)

        # Top 20 tracker (OOS PF基準、最低3トレード)
        if oos_m["total"] >= 3:
            self.best_results.append(result)
            self.best_results.sort(key=lambda r: r["oos"]["pf"], reverse=True)
            self.best_results = self.best_results[:30]

    async def run(self):
        """大規模バックテスト実行"""
        await self.load_data()

        if not self.stock_data:
            logger.error("データなし。終了。")
            return

        start_time = time.time()
        end_time = start_time + DURATION_MINUTES * 60
        total_tests = 0

        logger.info("=" * 60)
        logger.info(f"大規模バックテスト開始 ({DURATION_MINUTES}分間)")
        logger.info("=" * 60)

        # ============================================================
        # Phase 1: 全16戦略を個別にデフォルトパラメータでテスト
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 1: 全戦略個別テスト（デフォルトパラメータ）")
        logger.info("=" * 60)

        all_strategy_names = list(ALL_PARAM_SPACE.keys())

        for sname in all_strategy_names:
            if time.time() >= end_time:
                break

            StrategyRegistry.clear()
            StrategyRegistry.register_all_defaults()

            # 対象戦略だけ有効化
            for s in StrategyRegistry.get_all():
                s.config.is_active = (s.name == sname)

            strategies = StrategyRegistry.get_active()
            if not strategies:
                continue

            is_trades, oos_trades = await self._run_backtest(strategies)
            is_m = calc_metrics(is_trades)
            oos_m = calc_metrics(oos_trades)

            self._record_result(
                "individual", f"{sname} デフォルト", [sname], {}, is_m, oos_m,
            )
            total_tests += 1

            logger.info(
                f"  [{sname}] IS: {is_m['total']}件 WR={is_m['win_rate']:.0%} PF={is_m['pf']:.2f} ¥{is_m['pnl']:+,.0f} | "
                f"OOS: {oos_m['total']}件 WR={oos_m['win_rate']:.0%} PF={oos_m['pf']:.2f} ¥{oos_m['pnl']:+,.0f}"
            )

        # ============================================================
        # Phase 2: 各戦略 × レジームフィルタ
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: 戦略 × レジームフィルタ")
        logger.info("=" * 60)

        # Phase 1で成績が良かった戦略を優先
        phase1_good = [
            r for r in self.all_results
            if r["test_type"] == "individual" and r["oos"]["total"] >= 3
        ]
        phase1_good.sort(key=lambda r: r["oos"]["pf"], reverse=True)
        priority_strategies = [r["strategies"][0] for r in phase1_good[:8]]

        if not priority_strategies:
            priority_strategies = all_strategy_names[:8]

        for sname in priority_strategies:
            for rf in REGIME_FILTER_CONFIGS:
                if time.time() >= end_time:
                    break
                if rf["name"] == "no_filter":
                    continue  # Phase 1で済み

                StrategyRegistry.clear()
                StrategyRegistry.register_all_defaults()
                for s in StrategyRegistry.get_all():
                    s.config.is_active = (s.name == sname)

                strategies = StrategyRegistry.get_active()
                if not strategies:
                    continue

                is_trades, oos_trades = await self._run_backtest(
                    strategies, blocked_regimes=rf["blocked_regimes"]
                )
                is_m = calc_metrics(is_trades)
                oos_m = calc_metrics(oos_trades)

                self._record_result(
                    "regime_filter", f"{sname} + {rf['name']}", [sname], {},
                    is_m, oos_m, regime_filter=rf["name"],
                )
                total_tests += 1

                if oos_m["total"] >= 3:
                    logger.info(
                        f"  [{sname}+{rf['name']}] "
                        f"OOS: {oos_m['total']}件 WR={oos_m['win_rate']:.0%} PF={oos_m['pf']:.2f} ¥{oos_m['pnl']:+,.0f}"
                    )

        # ============================================================
        # Phase 3: パラメータランダムサーチ（残り時間の50%）
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 3: パラメータランダムサーチ")
        logger.info("=" * 60)

        remaining = end_time - time.time()
        phase3_end = time.time() + remaining * 0.5

        # 有望戦略を中心にパラメータ探索
        search_weights = {}
        for r in self.all_results:
            for s in r["strategies"]:
                if s not in search_weights:
                    search_weights[s] = 0
                if r["oos"]["total"] >= 3 and r["oos"]["pf"] > 0.5:
                    search_weights[s] += r["oos"]["pf"]

        weighted_strategies = sorted(search_weights.items(), key=lambda x: -x[1])
        search_pool = [s for s, _ in weighted_strategies if s in ALL_PARAM_SPACE]
        if not search_pool:
            search_pool = list(ALL_PARAM_SPACE.keys())

        while time.time() < phase3_end:
            # 重み付きランダム選択（上位に偏る）
            idx = min(int(abs(random.gauss(0, len(search_pool) / 3))), len(search_pool) - 1)
            sname = search_pool[idx]

            if sname not in ALL_PARAM_SPACE:
                continue

            # ランダムパラメータ
            param_space = ALL_PARAM_SPACE[sname]
            trial_params = {k: random.choice(v) for k, v in param_space.items()}

            # ランダムでレジームフィルタも適用
            rf = random.choice(REGIME_FILTER_CONFIGS)

            # 収束フィルタも20%の確率で変更
            conv_params = None
            if random.random() < 0.2:
                from core.config import settings
                conv_params = {k: random.choice(v) for k, v in CONVERGENCE_SPACE.items()}
                for k, v in conv_params.items():
                    setattr(settings, k, v)

            StrategyRegistry.clear()
            StrategyRegistry.register_all_defaults()
            for s in StrategyRegistry.get_all():
                s.config.is_active = (s.name == sname)
                if s.name == sname:
                    for pk, pv in trial_params.items():
                        if pk in s.config.parameter_set:
                            s.config.parameter_set[pk] = pv

            strategies = StrategyRegistry.get_active()
            if not strategies:
                # 収束フィルタリセット
                if conv_params:
                    from core.config import Settings
                    defaults = Settings()
                    for k in conv_params:
                        setattr(settings, k, getattr(defaults, k))
                continue

            is_trades, oos_trades = await self._run_backtest(
                strategies, blocked_regimes=rf["blocked_regimes"],
            )

            # 収束フィルタリセット
            if conv_params:
                from core.config import Settings
                defaults = Settings()
                for k in conv_params:
                    setattr(settings, k, getattr(defaults, k))

            is_m = calc_metrics(is_trades)
            oos_m = calc_metrics(oos_trades)

            desc = f"{sname} param_search + {rf['name']}"
            self._record_result(
                "param_search", desc, [sname], trial_params,
                is_m, oos_m, regime_filter=rf["name"], conv_params=conv_params,
            )
            total_tests += 1

            if oos_m["total"] >= 3 and oos_m["pf"] >= 1.0:
                logger.info(
                    f"  ★ [{sname}+{rf['name']}] "
                    f"OOS PF={oos_m['pf']:.2f} WR={oos_m['win_rate']:.0%} ¥{oos_m['pnl']:+,.0f} "
                    f"({oos_m['total']}件) params={trial_params}"
                )

            if total_tests % 20 == 0:
                elapsed = (time.time() - start_time) / 60
                logger.info(f"  --- {total_tests}パターン完了 ({elapsed:.0f}分経過) ---")

        # ============================================================
        # Phase 4: 組み合わせテスト（残り時間）
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("Phase 4: 戦略組み合わせテスト")
        logger.info("=" * 60)

        # OOS PFが高い戦略を組み合わせてテスト
        top_individual = [
            r for r in self.all_results
            if r["test_type"] in ("individual", "regime_filter", "param_search")
            and r["oos"]["total"] >= 3 and r["oos"]["pf"] >= 0.8
        ]
        top_individual.sort(key=lambda r: r["oos"]["pf"], reverse=True)

        top_strategy_names = list(dict.fromkeys(r["strategies"][0] for r in top_individual[:10]))

        # 2戦略の組み合わせ
        combos_2 = list(itertools.combinations(top_strategy_names[:6], 2))
        # 3戦略の組み合わせ
        combos_3 = list(itertools.combinations(top_strategy_names[:5], 3))
        all_combos = combos_2 + combos_3
        random.shuffle(all_combos)

        for combo in all_combos:
            if time.time() >= end_time:
                break

            combo_names = list(combo)

            # best regime filter for this combo
            rf = random.choice(REGIME_FILTER_CONFIGS[:4])

            StrategyRegistry.clear()
            StrategyRegistry.register_all_defaults()
            for s in StrategyRegistry.get_all():
                s.config.is_active = (s.name in combo_names)

            strategies = StrategyRegistry.get_active()
            if not strategies:
                continue

            is_trades, oos_trades = await self._run_backtest(
                strategies, blocked_regimes=rf["blocked_regimes"],
            )
            is_m = calc_metrics(is_trades)
            oos_m = calc_metrics(oos_trades)

            desc = f"combo: {'+'.join(combo_names)} + {rf['name']}"
            self._record_result(
                "combination", desc, combo_names, {},
                is_m, oos_m, regime_filter=rf["name"],
            )
            total_tests += 1

            if oos_m["total"] >= 3:
                logger.info(
                    f"  [{'+'.join(combo_names)}+{rf['name']}] "
                    f"OOS: {oos_m['total']}件 PF={oos_m['pf']:.2f} WR={oos_m['win_rate']:.0%} ¥{oos_m['pnl']:+,.0f}"
                )

        # ============================================================
        # 結果サマリー
        # ============================================================
        elapsed_total = (time.time() - start_time) / 60
        logger.info("\n" + "=" * 60)
        logger.info(f"大規模バックテスト完了: {total_tests}パターン ({elapsed_total:.1f}分)")
        logger.info("=" * 60)

        # Phase別の集計
        for phase in ["individual", "regime_filter", "param_search", "combination"]:
            phase_results = [r for r in self.all_results if r["test_type"] == phase]
            if phase_results:
                logger.info(f"\n[{phase}] {len(phase_results)}パターン")
                good = [r for r in phase_results if r["oos"]["total"] >= 3 and r["oos"]["pf"] >= 1.0]
                logger.info(f"  OOS PF >= 1.0: {len(good)}件")

        # Top 20
        logger.info("\n" + "=" * 60)
        logger.info("Top 20 パターン (OOS PF順)")
        logger.info("=" * 60)
        for i, r in enumerate(self.best_results[:20], 1):
            logger.info(
                f"  {i:2d}. [{r['test_type']}] {r['description']}\n"
                f"      OOS: {r['oos']['total']}件 WR={r['oos']['win_rate']:.0%} "
                f"PF={r['oos']['pf']:.2f} ¥{r['oos']['pnl']:+,.0f} Sharpe={r['oos']['sharpe']:.2f}\n"
                f"      IS:  {r['is']['total']}件 WR={r['is']['win_rate']:.0%} "
                f"PF={r['is']['pf']:.2f} ¥{r['is']['pnl']:+,.0f}"
            )
            if r.get("params") and r["params"]:
                logger.info(f"      params: {r['params']}")
            if r.get("convergence_params"):
                logger.info(f"      conv: {r['convergence_params']}")

        # 戦略別ベストパラメータ
        logger.info("\n" + "=" * 60)
        logger.info("戦略別ベストパラメータ")
        logger.info("=" * 60)
        strategy_best: dict[str, dict] = {}
        for r in self.all_results:
            if r["oos"]["total"] < 3:
                continue
            for sname in r["strategies"]:
                if sname not in strategy_best or r["oos"]["pf"] > strategy_best[sname]["oos"]["pf"]:
                    strategy_best[sname] = r
        for sname, r in sorted(strategy_best.items(), key=lambda x: -x[1]["oos"]["pf"]):
            logger.info(
                f"  {sname}: OOS PF={r['oos']['pf']:.2f} WR={r['oos']['win_rate']:.0%} "
                f"¥{r['oos']['pnl']:+,.0f} ({r['oos']['total']}件) "
                f"filter={r['regime_filter']}"
            )
            if r.get("params"):
                logger.info(f"    params: {r['params']}")

        # JSON保存
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_patterns": total_tests,
            "duration_minutes": round(elapsed_total, 1),
            "top_20": [
                {
                    "rank": i + 1,
                    "test_type": r["test_type"],
                    "description": r["description"],
                    "strategies": r["strategies"],
                    "params": r.get("params", {}),
                    "convergence_params": r.get("convergence_params"),
                    "regime_filter": r["regime_filter"],
                    "oos": r["oos"],
                    "is": r["is"],
                }
                for i, r in enumerate(self.best_results[:20])
            ],
            "strategy_best": {
                sname: {
                    "oos": r["oos"],
                    "is": r["is"],
                    "params": r.get("params", {}),
                    "regime_filter": r["regime_filter"],
                    "convergence_params": r.get("convergence_params"),
                }
                for sname, r in sorted(strategy_best.items(), key=lambda x: -x[1]["oos"]["pf"])
            },
            "all_results_summary": {
                "total": total_tests,
                "oos_pf_over_1": sum(1 for r in self.all_results if r["oos"]["total"] >= 3 and r["oos"]["pf"] >= 1.0),
                "oos_pf_over_1_5": sum(1 for r in self.all_results if r["oos"]["total"] >= 3 and r["oos"]["pf"] >= 1.5),
                "oos_pf_over_2": sum(1 for r in self.all_results if r["oos"]["total"] >= 3 and r["oos"]["pf"] >= 2.0),
            },
        }
        Path("knowledge/massive_backtest_results.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"\n結果を knowledge/massive_backtest_results.json に保存")


if __name__ == "__main__":
    bt = MassiveBacktester()
    asyncio.run(bt.run())
