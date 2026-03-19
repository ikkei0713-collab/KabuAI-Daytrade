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

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/backtest_learn.log", rotation="10 MB", level="DEBUG")

# 対象銘柄（主要大型株 + 値動きの大きい銘柄）
TARGET_CODES = [
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
]


class BacktestLearner:
    def __init__(self):
        self.db = DatabaseManager()
        self.fe = FeatureEngineer()
        self.all_trades: list[TradeResult] = []

    async def run(self):
        await self.db.init_db()
        StrategyRegistry.register_all_defaults()
        strategies = StrategyRegistry.get_active()

        logger.info("=" * 60)
        logger.info("バックテスト型ナレッジ蓄積 開始")
        logger.info(f"  銘柄数: {len(TARGET_CODES)}")
        logger.info(f"  戦略数: {len(strategies)}")
        logger.info("=" * 60)

        # TDnetイベントを日付別に取得
        tdnet_events: dict[date, dict[str, str]] = {}  # date -> {ticker: event_type}
        logger.info("TDnetイベント取得中...")
        async with TDnetClient() as tdnet:
            # 過去3ヶ月の営業日をスキャン
            d = date(2026, 1, 1)
            end = date(2026, 3, 18)
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
            # 全銘柄のデータを一括取得
            stock_data: dict[str, pd.DataFrame] = {}
            for code in TARGET_CODES:
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
        logger.info(f"シミュレーション期間: {sim_dates[0]} ~ {sim_dates[-1]} ({len(sim_dates)}日)")

        for sim_date in sim_dates:
            day_trades = await self._simulate_day(sim_date, stock_data, strategies, tdnet_events)
            self.all_trades.extend(day_trades)

            if day_trades:
                wins = sum(1 for t in day_trades if t.pnl > 0)
                total_pnl = sum(t.pnl for t in day_trades)
                logger.info(
                    f"  {sim_date}: {len(day_trades)}件 "
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
    ) -> list[TradeResult]:
        """1日分のシミュレーション"""
        trades = []

        for code, full_df in stock_data.items():
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

                    # ストップ/利確判定
                    if signal.direction == "long":
                        # ストップロスにヒット?
                        if next_low <= signal.stop_loss:
                            exit_price = signal.stop_loss
                            exit_reason = "ストップロス"
                        # 利確にヒット?
                        elif next_high >= signal.take_profit:
                            exit_price = signal.take_profit
                            exit_reason = "利確"
                        else:
                            exit_price = next_close
                            exit_reason = "翌日決済"
                        pnl = (exit_price - entry_price) * 100  # 100株
                    else:
                        if next_high >= signal.stop_loss:
                            exit_price = signal.stop_loss
                            exit_reason = "ストップロス"
                        elif next_low <= signal.take_profit:
                            exit_price = signal.take_profit
                            exit_reason = "利確"
                        else:
                            exit_price = next_close
                            exit_reason = "翌日決済"
                        pnl = (entry_price - exit_price) * 100

                    pnl_pct = pnl / (entry_price * 100) * 100

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
                        features_at_entry=features,
                        market_condition="",
                    )
                    trades.append(trade)
                    await self.db.save_trade(trade)

                except Exception:
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
            sl_count = sum(1 for t in strades if t.exit_reason == "ストップロス")
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

    def _print_summary(self):
        """最終サマリー"""
        if not self.all_trades:
            logger.info("トレードなし")
            return

        total = len(self.all_trades)
        wins = sum(1 for t in self.all_trades if t.pnl > 0)
        total_pnl = sum(t.pnl for t in self.all_trades)
        avg_pnl = total_pnl / total
        gross_profit = sum(t.pnl for t in self.all_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.all_trades if t.pnl <= 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        logger.info("=" * 60)
        logger.info("バックテスト結果サマリー")
        logger.info("=" * 60)
        logger.info(f"  総トレード数: {total}")
        logger.info(f"  勝ち: {wins} / 負け: {total - wins}")
        logger.info(f"  勝率: {wins/total:.1%}")
        logger.info(f"  PF: {pf:.2f}")
        logger.info(f"  累計損益: ¥{total_pnl:+,.0f}")
        logger.info(f"  平均損益: ¥{avg_pnl:+,.0f}")
        logger.info(f"  最大勝ち: ¥{max(t.pnl for t in self.all_trades):+,.0f}")
        logger.info(f"  最大負け: ¥{min(t.pnl for t in self.all_trades):+,.0f}")
        logger.info("=" * 60)

        # JSONにも保存
        summary = {
            "date": datetime.now().isoformat(),
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / total, 3),
            "profit_factor": round(pf, 2),
            "total_pnl": round(total_pnl, 0),
            "avg_pnl": round(avg_pnl, 0),
        }
        Path("knowledge/backtest_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    learner = BacktestLearner()
    asyncio.run(learner.run())
