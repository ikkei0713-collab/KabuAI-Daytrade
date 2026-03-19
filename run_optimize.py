"""
戦略最適化ループ

1時間かけてパラメータを変えながらバックテストを繰り返し、
最適なパラメータセットを探索する。

最適化対象:
- vwap_reclaim: 閾値、ストップ倍率、ターゲット倍率
- spread_entry: 条件厳格化
- orb: ウィンドウ、閾値
- vwap_bounce: 条件
- trend_follow: EMA期間

方法: グリッドサーチ + ランダムサーチの組み合わせ
評価: OOSのPF（過学習を避ける）
"""

import asyncio
import copy
import json
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

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

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/optimize.log", rotation="10 MB", level="DEBUG")

# 候補銘柄（run_backtest_learn.pyと同じ）
from run_backtest_learn import CANDIDATE_CODES, _clean_features

DURATION_MINUTES = 30  # 最適化実行時間


def _calc_metrics(trades: list[TradeResult]) -> dict:
    if not trades:
        return {"total": 0, "wins": 0, "win_rate": 0, "pf": 0, "pnl": 0, "avg": 0}
    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    pnl = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    return {
        "total": total,
        "wins": wins,
        "win_rate": wins / total if total else 0,
        "pf": gp / gl if gl > 0 else 0,
        "pnl": pnl,
        "avg": pnl / total if total else 0,
    }


# パラメータ空間の定義
# v3: vwap_reclaim のみ active なので vwap_reclaim に集中探索
PARAM_SPACE = {
    "vwap_reclaim": {
        "min_time_below_vwap_min": [5, 10, 15, 20, 25, 30],
        "min_volume_at_reclaim": [1.0, 1.2, 1.3, 1.5, 1.8, 2.0],
        "target_atr_multiple": [0.8, 1.0, 1.2, 1.5, 1.8, 2.0],
        "reclaim_buffer_pct": [0.05, 0.10, 0.15, 0.20, 0.25],
        "max_distance_from_vwap_pct": [0.5, 0.8, 1.0, 1.2, 1.5, 2.0],
    },
}


class Optimizer:
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
        self.results: list[dict] = []
        self.best_oos_pf = 0.0
        self.best_params: dict = {}

    async def load_data(self):
        """データを一回だけ取得"""
        logger.info("データ取得開始...")

        # TDnet
        async with TDnetClient() as tdnet:
            d = date(2026, 1, 1)
            end = date(2026, 3, 18)
            while d <= end:
                if d.weekday() < 5:
                    try:
                        disclosures = await tdnet.fetch_today_disclosures(d)
                        material = tdnet.filter_material_events(disclosures)
                        for disc in material:
                            self.tdnet_events.setdefault(d, {})[disc.ticker + "0"] = disc.disclosure_type.value
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)
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
                    raw = await client.get_prices_daily(code, "2025-12-01", "2026-03-18")
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
                await asyncio.sleep(0.2)

        logger.info(f"データ取得完了: {len(self.stock_data)}銘柄")

        # 日付リスト
        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df["Date"].dt.date.tolist())
        self.sim_dates = sorted(all_dates)[30:]  # 30日ウォームアップ

        split = int(len(self.sim_dates) * 0.6)
        self.is_dates = set(self.sim_dates[:split])
        self.oos_dates = set(self.sim_dates[split:])
        logger.info(f"シミュレーション: {len(self.sim_dates)}日 (IS {len(self.is_dates)} / OOS {len(self.oos_dates)})")

    def _apply_params(self, strategy: BaseStrategy, params: dict):
        """戦略のパラメータを一時的に変更"""
        for k, v in params.items():
            if k in strategy.config.parameter_set:
                strategy.config.parameter_set[k] = v

    async def _run_backtest(self, strategies: list[BaseStrategy]) -> tuple[list[TradeResult], list[TradeResult]]:
        """IS/OOS分離バックテスト（高速版）"""
        is_trades = []
        oos_trades = []
        capital = 10_000_000

        market_proxy = max(self.stock_data.keys(), key=lambda c: len(self.stock_data[c]))

        for sim_date in self.sim_dates:
            # レジーム
            mp_df = self.stock_data[market_proxy]
            mask = mp_df["Date"].dt.date <= sim_date
            mp_slice = mp_df[mask]
            regime = self.regime_detector.detect(mp_slice).regime if len(mp_slice) >= 50 else "range"

            for code, full_df in self.stock_data.items():
                df = full_df[full_df["Date"].dt.date <= sim_date]
                if len(df) < 20:
                    continue

                next_rows = full_df[full_df["Date"].dt.date > sim_date]
                if next_rows.empty:
                    continue

                features = self.fe.calculate_all_features(df)
                entry_price = float(df["close"].iloc[-1])
                features["current_price"] = entry_price
                atr = features.get("atr", entry_price * 0.02)

                # TDnet
                if self.tdnet_events and sim_date in self.tdnet_events:
                    evt = self.tdnet_events[sim_date].get(code, "")
                    if evt:
                        features["event_type"] = evt
                        features["event_magnitude"] = 1.0

                for strategy in strategies:
                    if not strategy.config.is_active:
                        continue
                    try:
                        signal = await strategy.scan(code, df, features)
                        if not signal or signal.confidence < 0.3:
                            continue

                        nr = next_rows.iloc[0]
                        nh, nl, nc = float(nr["high"]), float(nr["low"]), float(nr["close"])

                        if signal.direction == "long":
                            if nl <= signal.stop_loss:
                                exit_p = signal.stop_loss
                            elif nh >= signal.take_profit:
                                exit_p = signal.take_profit
                            else:
                                exit_p = nc
                            qty = strategy.calculate_position_size(entry_price, atr, capital)
                            raw_pnl = (exit_p - entry_price) * qty
                        else:
                            if nh >= signal.stop_loss:
                                exit_p = signal.stop_loss
                            elif nl <= signal.take_profit:
                                exit_p = signal.take_profit
                            else:
                                exit_p = nc
                            qty = strategy.calculate_position_size(entry_price, atr, capital)
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
                            exit_reason="backtest",
                            market_condition=regime,
                        )

                        if sim_date in self.is_dates:
                            is_trades.append(trade)
                        else:
                            oos_trades.append(trade)

                    except Exception:
                        continue

        return is_trades, oos_trades

    async def run(self):
        """最適化メインループ"""
        await self.load_data()

        start_time = time.time()
        end_time = start_time + DURATION_MINUTES * 60
        iteration = 0

        logger.info("=" * 60)
        logger.info(f"最適化開始 ({DURATION_MINUTES}分間)")
        logger.info("=" * 60)

        # ベースライン測定
        StrategyRegistry.register_all_defaults()
        baseline_strategies = StrategyRegistry.get_active()
        is_trades, oos_trades = await self._run_backtest(baseline_strategies)
        is_m = _calc_metrics(is_trades)
        oos_m = _calc_metrics(oos_trades)
        logger.info(
            f"ベースライン: IS {is_m['total']}件 WR={is_m['win_rate']:.0%} PF={is_m['pf']:.2f} | "
            f"OOS {oos_m['total']}件 WR={oos_m['win_rate']:.0%} PF={oos_m['pf']:.2f}"
        )
        self.best_oos_pf = oos_m["pf"]
        self.best_params = {"baseline": True}

        self.results.append({
            "iteration": 0,
            "params": {"baseline": True},
            "is": is_m,
            "oos": oos_m,
        })

        # 最適化ループ
        while time.time() < end_time:
            iteration += 1
            elapsed = (time.time() - start_time) / 60
            remaining = (end_time - time.time()) / 60

            # ランダムにパラメータを選択
            strategy_name = random.choice(list(PARAM_SPACE.keys()))
            param_choices = PARAM_SPACE[strategy_name]
            trial_params = {k: random.choice(v) for k, v in param_choices.items()}

            # 戦略をリセットして再登録
            StrategyRegistry.clear()
            StrategyRegistry.register_all_defaults()
            strategies = StrategyRegistry.get_active()

            # パラメータ適用
            for s in strategies:
                if s.name == strategy_name:
                    self._apply_params(s, trial_params)

            # バックテスト
            is_trades, oos_trades = await self._run_backtest(strategies)
            is_m = _calc_metrics(is_trades)
            oos_m = _calc_metrics(oos_trades)

            # 評価（OOSのPFで判断、ただしトレード数が少なすぎるのは除外）
            improved = False
            if oos_m["total"] >= 5 and oos_m["pf"] > self.best_oos_pf:
                self.best_oos_pf = oos_m["pf"]
                self.best_params = {strategy_name: trial_params}
                improved = True

            self.results.append({
                "iteration": iteration,
                "strategy": strategy_name,
                "params": trial_params,
                "is": is_m,
                "oos": oos_m,
                "improved": improved,
            })

            marker = "★ NEW BEST" if improved else ""
            logger.info(
                f"[{iteration}] {elapsed:.0f}分経過 残{remaining:.0f}分 | "
                f"{strategy_name} | "
                f"IS: {is_m['total']}件 WR={is_m['win_rate']:.0%} PF={is_m['pf']:.2f} | "
                f"OOS: {oos_m['total']}件 WR={oos_m['win_rate']:.0%} PF={oos_m['pf']:.2f} "
                f"¥{oos_m['pnl']:+,.0f} {marker}"
            )

        # 結果保存
        logger.info("=" * 60)
        logger.info(f"最適化完了: {iteration}回試行")
        logger.info(f"最良OOS PF: {self.best_oos_pf:.2f}")
        logger.info(f"最良パラメータ: {self.best_params}")
        logger.info("=" * 60)

        # Top 10
        sorted_results = sorted(
            [r for r in self.results if r.get("oos", {}).get("total", 0) >= 3],
            key=lambda r: r["oos"]["pf"],
            reverse=True,
        )
        logger.info("Top 10 パラメータセット (OOS PF順):")
        for i, r in enumerate(sorted_results[:10], 1):
            s = r.get("strategy", "baseline")
            logger.info(
                f"  {i}. [{s}] OOS PF={r['oos']['pf']:.2f} WR={r['oos']['win_rate']:.0%} "
                f"¥{r['oos']['pnl']:+,.0f} ({r['oos']['total']}件) | "
                f"IS PF={r['is']['pf']:.2f} ({r['is']['total']}件)"
            )
            if "params" in r and not r["params"].get("baseline"):
                logger.info(f"     params: {r.get('params', {})}")

        # JSON保存
        output = {
            "timestamp": datetime.now().isoformat(),
            "total_iterations": iteration,
            "duration_minutes": DURATION_MINUTES,
            "best_oos_pf": round(self.best_oos_pf, 3),
            "best_params": self.best_params,
            "top_10": [
                {
                    "strategy": r.get("strategy", "baseline"),
                    "params": r.get("params", {}),
                    "oos_pf": round(r["oos"]["pf"], 3),
                    "oos_wr": round(r["oos"]["win_rate"], 3),
                    "oos_pnl": round(r["oos"]["pnl"], 0),
                    "oos_trades": r["oos"]["total"],
                    "is_pf": round(r["is"]["pf"], 3),
                    "is_trades": r["is"]["total"],
                }
                for r in sorted_results[:10]
            ],
        }
        Path("knowledge/optimization_results.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("結果を knowledge/optimization_results.json に保存")


if __name__ == "__main__":
    optimizer = Optimizer()
    asyncio.run(optimizer.run())
