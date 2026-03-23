"""
バックテスト型ナレッジ蓄積

過去3ヶ月の日足データで全16戦略を日次シミュレーション。
各日ごとに「その日の終値まで見える」状態でスキャン→仮想売買→翌日決済。
大量のトレード結果からナレッジを自動抽出する。
"""

import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from core.models import TradeResult, StrategyPerformance, KnowledgeEntry, CandidateUpdate
from db.database import DatabaseManager
from data_sources.jquants import JQuantsClient
from data_sources.tdnet import TDnetClient
from strategies.registry import StrategyRegistry
from tools.feature_engineering import FeatureEngineer
from tools.cost_model import CostModel
from tools.market_regime import RegimeDetector
from scanners.stock_selector import StockSelector
from core.ticker_map import update_from_jquants, format_ticker

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/backtest_learn.log", rotation="10 MB", level="DEBUG")

# 候補銘柄プール（StockSelectorで日次フィルタリングされる）
CANDIDATE_CODES = [
    "72030",  # トヨタ
    "67580",  # ソニー
    "99840",  # ソフトバンクG
    "83060",  # 三菱UFJ
    "69200",  # レーザーテック
    "69810",  # 村田製作所
    "69020",  # デンソー
    "41070",  # 伊勢化学
    "68610",  # キーエンス
    "79740",  # 任天堂
    "89010",  # 日本郵船
    "83160",  # 三井住友FG
    "90200",  # 東日本旅客鉄道
    "43850",  # メルカリ
    "40630",  # 信越化学
    "63670",  # ダイキン
    "91010",  # 日本郵船
    "70130",  # IHI
    "54010",  # 日本製鉄
    "87660",  # 東京海上HD
    "36590",  # ネクソン
    "27020",  # マクドナルド
    "92020",  # ANA
    "66450",  # オムロン
    "77350",  # SCREEN
    "63260",  # クボタ
    "28020",  # 味の素
    "43070",  # 野村総研
    "34360",  # SUMCO
    "66230",  # 愛知機械
    # 追加: 大型・値動き銘柄
    "98430",  # ニトリHD
    "60980",  # 日立製作所
    "75320",  # パン・パシフィック
    "41850",  # JSR
    "93070",  # 杉村倉庫
    "47560",  # カルナバイオ
    "63010",  # コマツ
    "80580",  # 三菱商事
    "91040",  # 商船三井
    "86010",  # 大和証券
    "95020",  # 中部電力
    "47680",  # 大塚商会
    "30920",  # ZOZO
    "21870",  # ジェイテクト
    "95310",  # 東京ガス
    "40050",  # 住友化学
    "25020",  # アサヒグループ
    "48490",  # エン・ジャパン
    "37690",  # GMOペイメント
    "61460",  # ディスコ
    "39230",  # ラクス
    "41200",  # スタンレー電気
    "65060",  # 安川電機
    "78320",  # バンダイナムコ
    "33820",  # セブン＆アイ
    "62730",  # SMC
    "80310",  # 三井物産
    "95830",  # セコム
    "47550",  # 楽天グループ
    "21410",  # 蝶理
    "60370",  # ファナック
    "94320",  # NTT
]

DEFAULT_CAPITAL = 10_000_000  # 1000万円


def _clean_features(features: dict) -> dict:
    """NaN/inf/非シリアライズ可能な値を除去"""
    clean = {}
    for k, v in features.items():
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                clean[k] = 0.0
            else:
                clean[k] = round(v, 6)
        elif isinstance(v, (int, str, bool, type(None))):
            clean[k] = v
        else:
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, ValueError):
                clean[k] = str(v)
    return clean


class BacktestLearner:
    def __init__(self, capital: float = DEFAULT_CAPITAL):
        self.db = DatabaseManager()
        self.fe = FeatureEngineer()
        self.cost_model = CostModel(commission_free=True)
        self.stock_selector = StockSelector()
        self.regime_detector = RegimeDetector()
        self.capital = capital
        self.all_trades: list[TradeResult] = []
        self.is_trades: list[TradeResult] = []   # in-sample
        self.oos_trades: list[TradeResult] = []  # out-of-sample
        # Per-strategy, per-regime tracking: {(strategy, regime): [TradeResult]}
        self.strategy_regime_trades: dict[tuple[str, str], list[TradeResult]] = {}

    async def run(self):
        await self.db.init_db()
        StrategyRegistry.register_all_defaults()
        strategies = StrategyRegistry.get_active()

        logger.info("=" * 60)
        logger.info("バックテスト型ナレッジ蓄積 開始")
        logger.info(f"  候補銘柄数: {len(CANDIDATE_CODES)}")
        logger.info(f"  戦略数: {len(strategies)}")
        logger.info(f"  初期資金: ¥{self.capital:,.0f}")
        logger.info("=" * 60)

        # TDnetイベントを日付別に取得
        tdnet_events: dict[date, dict[str, str]] = {}  # date -> {ticker: event_type}
        logger.info("TDnetイベント取得中...")
        async with TDnetClient() as tdnet:
            # 過去6ヶ月の営業日をスキャン
            d = date(2025, 9, 1)
            end = date(2026, 3, 19)
            while d <= end:
                if d.weekday() < 5:  # 平日のみ
                    try:
                        disclosures = await tdnet.fetch_today_disclosures(d)
                        material = tdnet.filter_material_events(disclosures)
                        for disc in material:
                            tdnet_events.setdefault(d, {})[disc.ticker + "0"] = disc.disclosure_type.value
                        if material:
                            logger.info(f"  {d}: {len(material)}件の重要開示")
                    except Exception as e:
                        logger.debug(f"  {d}: TDnet取得失敗 {e}")
                    await asyncio.sleep(0.5)
                d += timedelta(days=1)
        logger.info(f"TDnetイベント: {sum(len(v) for v in tdnet_events.values())}件取得")

        async with JQuantsClient() as client:
            # 銘柄名マスタ更新
            try:
                master = await client.get_listed_info()
                update_from_jquants(master)
            except Exception as e:
                logger.warning(f"銘柄名マスタ取得失敗: {e}")

            # 全銘柄のデータを一括取得
            stock_data: dict[str, pd.DataFrame] = {}
            for code in CANDIDATE_CODES:
                try:
                    raw = await client.get_prices_daily(code, "2025-09-01", "2026-03-19")
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
                        stock_data[code] = df
                        logger.info(f"  {code}: {len(df)}日分取得")
                except Exception as e:
                    logger.warning(f"  {code}: 取得失敗 {e}")
                await asyncio.sleep(0.3)

            logger.info(f"データ取得完了: {len(stock_data)}銘柄")

        # 日次シミュレーション
        if not stock_data:
            logger.error("データなし。終了。")
            return

        # 共通の日付リストを作成
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df["Date"].dt.date.tolist())
        sim_dates = sorted(all_dates)

        # 最初の30日はウォームアップ（特徴量計算用）
        sim_dates = sim_dates[30:]

        # In-sample / Out-of-sample split (60% / 40%)
        # IS/OOS 50/50 分割 (OOS サンプル増加のため)
        split_idx = int(len(sim_dates) * 0.5)
        is_dates = set(sim_dates[:split_idx])
        oos_dates = set(sim_dates[split_idx:])

        logger.info(f"シミュレーション期間: {sim_dates[0]} ~ {sim_dates[-1]} ({len(sim_dates)}日)")
        logger.info(f"  In-sample:  {sim_dates[0]} ~ {sim_dates[split_idx-1]} ({len(is_dates)}日)")
        logger.info(f"  Out-of-sample: {sim_dates[split_idx]} ~ {sim_dates[-1]} ({len(oos_dates)}日)")

        # Build a proxy market DataFrame for regime detection (use largest stock as proxy)
        market_proxy_code = max(stock_data.keys(), key=lambda c: len(stock_data[c]))
        market_proxy_df = stock_data[market_proxy_code]

        for sim_date in sim_dates:
            # --- Regime detection for this day ---
            market_mask = market_proxy_df["Date"].dt.date <= sim_date
            market_slice = market_proxy_df[market_mask]
            if len(market_slice) >= 50:
                regime_result = self.regime_detector.detect(market_slice)
                regime = regime_result.regime
            else:
                regime = "range"

            # --- Stock selection for this day ---
            day_codes = []
            for code, df in stock_data.items():
                mask = df["Date"].dt.date <= sim_date
                df_slice = df[mask]
                if len(df_slice) < 20:
                    continue
                has_event = bool(
                    tdnet_events and sim_date in tdnet_events and code in tdnet_events[sim_date]
                )
                score = self.stock_selector.score_stock(code, df_slice, has_event=has_event)
                if not score.excluded and score.total_score >= 0.15:
                    day_codes.append(code)

            day_trades = await self._simulate_day(
                sim_date, stock_data, strategies, tdnet_events,
                selected_codes=day_codes, regime=regime,
            )
            self.all_trades.extend(day_trades)

            # Classify into IS / OOS
            if sim_date in is_dates:
                self.is_trades.extend(day_trades)
            else:
                self.oos_trades.extend(day_trades)

            # Track per-strategy, per-regime
            for t in day_trades:
                key = (t.strategy_name, regime)
                self.strategy_regime_trades.setdefault(key, []).append(t)

            if day_trades:
                wins = sum(1 for t in day_trades if t.pnl > 0)
                total_pnl = sum(t.pnl for t in day_trades)
                logger.info(
                    f"  {sim_date} [{regime}]: {len(day_trades)}件 "
                    f"勝{wins}/負{len(day_trades)-wins} "
                    f"損益={total_pnl:+,.0f}円"
                )

        # ナレッジ抽出
        logger.info("=" * 60)
        logger.info(f"全トレード: {len(self.all_trades)}件")
        await self._extract_all_knowledge()
        await self._generate_improvements()

        # サマリー出力
        self._print_summary()

    async def _simulate_day(
        self, sim_date: date, stock_data: dict[str, pd.DataFrame], strategies,
        tdnet_events: dict[date, dict[str, str]] = None,
        selected_codes: list[str] | None = None,
        regime: str = "range",
    ) -> list[TradeResult]:
        """1日分のシミュレーション"""
        trades = []
        codes_to_sim = selected_codes if selected_codes is not None else list(stock_data.keys())

        for code in codes_to_sim:
            full_df = stock_data.get(code)
            if full_df is None:
                continue
            # sim_dateまでのデータでスキャン
            mask = full_df["Date"].dt.date <= sim_date
            df = full_df[mask].copy()
            if len(df) < 20:
                continue

            # sim_date翌日のデータ（決済用）
            next_mask = full_df["Date"].dt.date > sim_date
            next_rows = full_df[next_mask]

            features = self.fe.calculate_all_features(df)
            current_close = float(df["close"].iloc[-1])
            features["current_price"] = current_close

            # TDnetイベント注入
            if tdnet_events and sim_date in tdnet_events:
                event_type = tdnet_events[sim_date].get(code, "")
                if event_type:
                    features["event_type"] = event_type
                    features["event_magnitude"] = 1.0
                    features["historical_event_response"] = 0.5

            for strategy in strategies:
                try:
                    signal = await strategy.scan(code, df, features)
                    if not signal or signal.confidence < 0.3:
                        continue

                    # エントリー: 当日終値
                    entry_price = current_close
                    atr = features.get("atr", entry_price * 0.02)

                    # 決済: 翌日のデータで判断
                    if next_rows.empty:
                        continue

                    next_row = next_rows.iloc[0]
                    next_open = float(next_row["open"])
                    next_high = float(next_row["high"])
                    next_low = float(next_row["low"])
                    next_close = float(next_row["close"])

                    # Position sizing (ATR-based via BaseStrategy)
                    quantity = strategy.calculate_position_size(entry_price, atr, self.capital)

                    # Intraday exit simulation:
                    # 1. Check if stop hit during the day (low touches stop for long)
                    # 2. Check if TP hit during the day (high touches TP for long)
                    # 3. If neither, check strategy exit conditions at close
                    # 4. If still holding, exit at close
                    if signal.direction == "long":
                        # Stop is checked first (conservative: assume stop hit before TP if both possible)
                        if next_low <= signal.stop_loss:
                            exit_price = signal.stop_loss
                            exit_reason = "ストップロス(intraday)"
                        elif next_high >= signal.take_profit:
                            exit_price = signal.take_profit
                            exit_reason = "利確(intraday)"
                        else:
                            # Check if holding is losing at close
                            pnl_at_close = next_close - entry_price
                            if pnl_at_close < -atr * 0.5:
                                exit_price = next_close
                                exit_reason = "含み損決済"
                            else:
                                exit_price = next_close
                                exit_reason = "翌日決済"
                        raw_pnl = (exit_price - entry_price) * quantity
                    else:
                        if next_high >= signal.stop_loss:
                            exit_price = signal.stop_loss
                            exit_reason = "ストップロス(intraday)"
                        elif next_low <= signal.take_profit:
                            exit_price = signal.take_profit
                            exit_reason = "利確(intraday)"
                        else:
                            pnl_at_close = entry_price - next_close
                            if pnl_at_close < -atr * 0.5:
                                exit_price = next_close
                                exit_reason = "含み損決済"
                            else:
                                exit_price = next_close
                                exit_reason = "翌日決済"
                        raw_pnl = (entry_price - exit_price) * quantity

                    # Cost-adjusted prices and PnL
                    adj_entry = self.cost_model.adjust_entry_price(entry_price, signal.direction)
                    adj_exit = self.cost_model.adjust_exit_price(exit_price, signal.direction)
                    cost = self.cost_model.calculate_trade_cost(entry_price, exit_price, quantity)

                    if signal.direction == "long":
                        pnl = (adj_exit - adj_entry) * quantity - cost.total
                    else:
                        pnl = (adj_entry - adj_exit) * quantity - cost.total

                    pnl_pct = pnl / (entry_price * quantity) * 100 if quantity > 0 else 0.0

                    trade = TradeResult(
                        ticker=code,
                        strategy_name=strategy.name,
                        direction=signal.direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_time=datetime.combine(sim_date, datetime.min.time().replace(hour=9)),
                        exit_time=datetime.combine(sim_date + timedelta(days=1), datetime.min.time().replace(hour=15)),
                        pnl=round(pnl, 0),
                        pnl_pct=round(pnl_pct, 2),
                        holding_minutes=360,
                        entry_reason=signal.entry_reason,
                        exit_reason=exit_reason,
                        features_at_entry=_clean_features({
                            **features,
                            "_quantity": quantity,
                            "_raw_pnl": round(raw_pnl, 0),
                            "_cost_total": round(cost.total, 0),
                            "_regime": regime,
                        }),
                        market_condition=regime,
                    )
                    trades.append(trade)
                    try:
                        await self.db.save_trade(trade)
                    except Exception as e:
                        logger.debug(f"DB保存スキップ {code}/{strategy.name}: {e}")

                except Exception as e:
                    logger.debug(f"シミュレーションエラー {code}/{strategy.name}: {e}")
                    continue

        return trades

    async def _extract_all_knowledge(self):
        """全トレードからナレッジ抽出"""
        if not self.all_trades:
            return

        wins = [t for t in self.all_trades if t.pnl > 0]
        losses = [t for t in self.all_trades if t.pnl <= 0]

        logger.info(f"勝ちトレード: {len(wins)}件, 負けトレード: {len(losses)}件")
        win_rate = len(wins) / len(self.all_trades) if self.all_trades else 0
        logger.info(f"全体勝率: {win_rate:.1%}")

        # 戦略別分析
        strategy_trades: dict[str, list[TradeResult]] = {}
        for t in self.all_trades:
            strategy_trades.setdefault(t.strategy_name, []).append(t)

        for sname, strades in sorted(strategy_trades.items()):
            sw = sum(1 for t in strades if t.pnl > 0)
            sl = len(strades) - sw
            swr = sw / len(strades) if strades else 0
            total_pnl = sum(t.pnl for t in strades)
            avg_pnl = total_pnl / len(strades)
            gross_profit = sum(t.pnl for t in strades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in strades if t.pnl <= 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

            logger.info(
                f"  {sname}: {len(strades)}件 勝率{swr:.0%} "
                f"PF={pf:.2f} 平均{avg_pnl:+,.0f}円 累計{total_pnl:+,.0f}円"
            )

            # 戦略パフォーマンスをDB保存
            perf = StrategyPerformance(
                strategy_name=sname,
                total_trades=len(strades),
                wins=sw,
                losses=sl,
                win_rate=round(swr, 3),
                profit_factor=round(pf, 2),
                avg_pnl=round(avg_pnl, 0),
                avg_holding_minutes=360,
            )
            await self.db.update_strategy_performance(sname, perf)

            # 勝ちパターン抽出
            wins_s = [t for t in strades if t.pnl > 0]
            if len(wins_s) >= 3:
                # 勝ちトレードの共通特徴を抽出
                win_features = []
                for t in wins_s:
                    if t.features_at_entry:
                        win_features.append(t.features_at_entry)

                if win_features:
                    wf_df = pd.DataFrame(win_features)
                    numeric_cols = wf_df.select_dtypes(include=[np.number]).columns
                    insights = []
                    for col in numeric_cols:
                        vals = wf_df[col].dropna()
                        if len(vals) >= 3:
                            median = vals.median()
                            q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
                            insights.append(f"{col}: {q25:.2f}~{q75:.2f} (中央値{median:.2f})")

                    if insights:
                        content = f"{sname}の勝ちパターン ({len(wins_s)}勝, 勝率{swr:.0%}): " + "; ".join(insights[:5])
                        entry = KnowledgeEntry(
                            category="win_pattern",
                            content=content,
                            supporting_trades=[t.id for t in wins_s[:10]],
                            confidence=min(0.9, swr),
                        )
                        await self.db.save_knowledge(entry)

            # 負けパターン抽出
            losses_s = [t for t in strades if t.pnl <= 0]
            if len(losses_s) >= 3:
                loss_reasons = {}
                for t in losses_s:
                    loss_reasons[t.exit_reason] = loss_reasons.get(t.exit_reason, 0) + 1

                content = f"{sname}の負けパターン ({len(losses_s)}敗): " + ", ".join(
                    f"{r}={c}件" for r, c in sorted(loss_reasons.items(), key=lambda x: -x[1])
                )
                entry = KnowledgeEntry(
                    category="loss_pattern",
                    content=content,
                    supporting_trades=[t.id for t in losses_s[:10]],
                    confidence=min(0.9, len(losses_s) / len(strades)),
                )
                await self.db.save_knowledge(entry)

    async def _generate_improvements(self):
        """改善候補を生成"""
        strategy_trades: dict[str, list[TradeResult]] = {}
        for t in self.all_trades:
            strategy_trades.setdefault(t.strategy_name, []).append(t)

        for sname, strades in strategy_trades.items():
            swr = sum(1 for t in strades if t.pnl > 0) / len(strades) if strades else 0

            # 勝率が低い戦略
            if swr < 0.4 and len(strades) >= 5:
                update = CandidateUpdate(
                    strategy_name=sname,
                    proposed_changes={"action": "tighten_conditions", "reason": f"勝率{swr:.0%}が低い"},
                    reason=f"勝率{swr:.0%} ({len(strades)}件中{sum(1 for t in strades if t.pnl>0)}勝)",
                    expected_improvement="勝率向上（シグナル数減少と引き換え）",
                )
                await self.db.save_candidate_update(update)

            # ストップロスが多すぎる戦略
            sl_count = sum(1 for t in strades if "ストップロス" in (t.exit_reason or ""))
            sl_ratio = sl_count / len(strades) if strades else 0
            if sl_ratio > 0.5 and len(strades) >= 5:
                update = CandidateUpdate(
                    strategy_name=sname,
                    proposed_changes={"action": "widen_stoploss", "sl_ratio": sl_ratio},
                    reason=f"ストップロス率{sl_ratio:.0%}が高すぎる",
                    expected_improvement="ストップロス幅拡大でノイズ回避",
                )
                await self.db.save_candidate_update(update)

            # 勝率が高い戦略は確信度閾値を下げる提案
            if swr > 0.65 and len(strades) >= 5:
                update = CandidateUpdate(
                    strategy_name=sname,
                    proposed_changes={"action": "lower_confidence_threshold", "current_wr": swr},
                    reason=f"勝率{swr:.0%}が高い。閾値を下げてシグナル数を増やせる可能性",
                    expected_improvement="トレード機会の増加",
                )
                await self.db.save_candidate_update(update)

        logger.info("改善候補の生成完了")

    @staticmethod
    def _calc_metrics(trades: list[TradeResult]) -> dict:
        """Calculate standard metrics for a list of trades."""
        if not trades:
            return {"total": 0, "wins": 0, "win_rate": 0, "pf": 0,
                    "total_pnl": 0, "avg_pnl": 0, "max_win": 0, "max_loss": 0,
                    "raw_pnl": 0, "cost_total": 0}
        total = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in trades)
        avg_pnl = total_pnl / total
        gp = sum(t.pnl for t in trades if t.pnl > 0)
        gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        pf = gp / gl if gl > 0 else 99.0
        raw_pnl = sum(t.features_at_entry.get("_raw_pnl", t.pnl) for t in trades)
        cost_total = sum(t.features_at_entry.get("_cost_total", 0) for t in trades)
        return {
            "total": total, "wins": wins,
            "win_rate": wins / total,
            "pf": pf,
            "total_pnl": total_pnl, "avg_pnl": avg_pnl,
            "max_win": max(t.pnl for t in trades),
            "max_loss": min(t.pnl for t in trades),
            "raw_pnl": raw_pnl,
            "cost_total": cost_total,
        }

    @staticmethod
    def _print_metrics_block(label: str, m: dict):
        """Print a formatted metrics block."""
        logger.info(f"  [{label}]")
        logger.info(f"    トレード数: {m['total']}")
        logger.info(f"    勝ち/負け: {m['wins']} / {m['total'] - m['wins']}")
        logger.info(f"    勝率: {m['win_rate']:.1%}")
        logger.info(f"    PF: {m['pf']:.2f}")
        logger.info(f"    累計損益(コスト調整後): ¥{m['total_pnl']:+,.0f}")
        logger.info(f"    累計損益(コスト前):      ¥{m['raw_pnl']:+,.0f}")
        logger.info(f"    総コスト:                ¥{m['cost_total']:+,.0f}")
        logger.info(f"    平均損益: ¥{m['avg_pnl']:+,.0f}")
        if m["total"] > 0:
            logger.info(f"    最大勝ち: ¥{m['max_win']:+,.0f}")
            logger.info(f"    最大負け: ¥{m['max_loss']:+,.0f}")

    def _print_summary(self):
        """最終サマリー"""
        if not self.all_trades:
            logger.info("トレードなし")
            return

        m_all = self._calc_metrics(self.all_trades)
        m_is = self._calc_metrics(self.is_trades)
        m_oos = self._calc_metrics(self.oos_trades)

        logger.info("=" * 60)
        logger.info("バックテスト結果サマリー")
        logger.info("=" * 60)
        self._print_metrics_block("全体", m_all)
        logger.info("-" * 40)
        self._print_metrics_block("In-Sample (60%)", m_is)
        logger.info("-" * 40)
        self._print_metrics_block("Out-of-Sample (40%)", m_oos)

        # --- Per-strategy IS vs OOS comparison ---
        logger.info("=" * 60)
        logger.info("戦略別 In-Sample vs Out-of-Sample 比較")
        logger.info("=" * 60)

        strategy_names = sorted({t.strategy_name for t in self.all_trades})
        for sname in strategy_names:
            s_is = [t for t in self.is_trades if t.strategy_name == sname]
            s_oos = [t for t in self.oos_trades if t.strategy_name == sname]
            sm_is = self._calc_metrics(s_is)
            sm_oos = self._calc_metrics(s_oos)

            if sm_is["total"] == 0:
                continue

            # Flag significant degradation
            flag = ""
            if sm_oos["total"] >= 3 and sm_is["total"] >= 3:
                wr_drop = sm_is["win_rate"] - sm_oos["win_rate"]
                pf_drop = sm_is["pf"] - sm_oos["pf"]
                if wr_drop > 0.15 or pf_drop > 0.5:
                    flag = " *** OOS劣化 ***"

            logger.info(
                f"  {sname}:{flag}\n"
                f"    IS:  {sm_is['total']}件 勝率{sm_is['win_rate']:.0%} PF={sm_is['pf']:.2f} 累計¥{sm_is['total_pnl']:+,.0f}\n"
                f"    OOS: {sm_oos['total']}件 勝率{sm_oos['win_rate']:.0%} PF={sm_oos['pf']:.2f} 累計¥{sm_oos['total_pnl']:+,.0f}"
            )

        # --- Per-strategy, per-regime breakdown ---
        logger.info("=" * 60)
        logger.info("戦略×レジーム別 サマリー")
        logger.info("=" * 60)
        for (sname, regime), trades in sorted(self.strategy_regime_trades.items()):
            if len(trades) < 2:
                continue
            rm = self._calc_metrics(trades)
            logger.info(
                f"  {sname} [{regime}]: {rm['total']}件 "
                f"勝率{rm['win_rate']:.0%} PF={rm['pf']:.2f} 累計¥{rm['total_pnl']:+,.0f}"
            )

        logger.info("=" * 60)

        # Proxy usage summary
        proxy_summary = StrategyRegistry.get_proxy_summary()
        status_summary = StrategyRegistry.get_status_summary()

        logger.info("=" * 60)
        logger.info("Proxy 依存度 & 戦略 Status")
        logger.info("=" * 60)
        for name, info in sorted(proxy_summary.items(), key=lambda x: -x[1]["proxy_usage_rate"]):
            logger.info(
                f"  {name}: status={info['status']} "
                f"proxy_rate={info['proxy_usage_rate']:.0%} "
                f"penalty={info['proxy_penalty']:.3f} "
                f"deps={info['proxy_features']}"
            )

        # Overfitting warning
        if m_is["total"] >= 5 and m_oos["total"] >= 3:
            wr_drop = m_is["win_rate"] - m_oos["win_rate"]
            pf_drop = m_is["pf"] - m_oos["pf"]
            if wr_drop > 0.10 or pf_drop > 0.5:
                logger.warning(
                    f"⚠ OVERFITTING WARNING: IS→OOS 劣化検知 "
                    f"(WR: {m_is['win_rate']:.0%}→{m_oos['win_rate']:.0%}, "
                    f"PF: {m_is['pf']:.2f}→{m_oos['pf']:.2f})"
                )

        # JSONにも保存
        summary = {
            "date": datetime.now().isoformat(),
            "total_trades": m_all["total"],
            "wins": m_all["wins"],
            "losses": m_all["total"] - m_all["wins"],
            "win_rate": round(m_all["win_rate"], 3),
            "profit_factor": round(m_all["pf"], 2),
            "total_pnl_cost_adjusted": round(m_all["total_pnl"], 0),
            "total_pnl_raw": round(m_all["raw_pnl"], 0),
            "total_cost": round(m_all["cost_total"], 0),
            "avg_pnl": round(m_all["avg_pnl"], 0),
            "in_sample": {
                "trades": m_is["total"], "win_rate": round(m_is["win_rate"], 3),
                "pf": round(m_is["pf"], 2), "pnl": round(m_is["total_pnl"], 0),
            },
            "out_of_sample": {
                "trades": m_oos["total"], "win_rate": round(m_oos["win_rate"], 3),
                "pf": round(m_oos["pf"], 2), "pnl": round(m_oos["total_pnl"], 0),
            },
            "strategy_regime": {
                f"{sname}_{regime}": {
                    "trades": len(trades),
                    "win_rate": round(self._calc_metrics(trades)["win_rate"], 3),
                    "pf": round(self._calc_metrics(trades)["pf"], 2),
                }
                for (sname, regime), trades in self.strategy_regime_trades.items()
                if len(trades) >= 2
            },
            "proxy_summary": {
                name: {
                    "status": info["status"],
                    "proxy_usage_rate": info["proxy_usage_rate"],
                    "proxy_penalty": info["proxy_penalty"],
                }
                for name, info in proxy_summary.items()
            },
            "strategy_status": status_summary,
            "data_quality_warnings": [
                "intraday proxy features are estimated from daily OHLCV",
                "proxy-dependent strategies have limited evaluation reliability",
                "optimization_results rankings should not be taken at face value",
            ],
        }
        Path("knowledge/backtest_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Claude Code 向けフィードバックパッケージ生成
        try:
            from analytics.feedback_packet import FeedbackPacketGenerator
            fbg = FeedbackPacketGenerator(
                all_trades=self.all_trades,
                is_trades=self.is_trades,
                oos_trades=self.oos_trades,
                strategy_regime_trades=self.strategy_regime_trades,
            )
            fbg.generate()
        except Exception as e:
            logger.warning(f"フィードバックパッケージ生成失敗: {e}")


if __name__ == "__main__":
    learner = BacktestLearner()
    asyncio.run(learner.run())
