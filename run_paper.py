"""
ペーパートレード常時稼働スクリプト

J-Quantsの日足データから特徴量を計算し、
16戦略をスキャンしてシミュレーション売買を実行。
トレード結果とナレッジをDBに蓄積し続ける。
"""

import asyncio
import json
import random
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from core.config import settings
from core.models import TradeSignal, TradeResult, StrategyPerformance, KnowledgeEntry
from db.database import DatabaseManager
from data_sources.jquants import JQuantsClient
from data_sources.tdnet import TDnetClient
from strategies.registry import StrategyRegistry
from strategies.base import BaseStrategy
from tools.feature_engineering import FeatureEngineer

# ── 設定 ──────────────────────────────────────────────────────────────────────
SCAN_INTERVAL = 90   # 秒ごとにスキャン（レート制限対策）
TOP_UNIVERSE = 20    # 上位N銘柄をスキャン（レート制限対策）
MAX_POSITIONS = 5
POSITION_SIZE = 500_000  # 円
MIN_CONFIDENCE = 0.3  # シグナル閾値（日足ベースなので緩めに）

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/paper_trade.log", rotation="1 day", retention="30 days", level="DEBUG")


class PaperTrader:
    def __init__(self):
        self.db = DatabaseManager()
        self.fe = FeatureEngineer()
        self.positions: dict[str, dict] = {}  # ticker -> position info
        self.trades: list[TradeResult] = []
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0

    async def init(self):
        await self.db.init_db()
        StrategyRegistry.register_all_defaults()
        logger.info(f"戦略 {len(StrategyRegistry.get_active())} 個を読み込み")

    async def fetch_universe(self, client: JQuantsClient) -> list[dict]:
        """出来高上位の銘柄を取得"""
        info = await client.get_listed_info()
        # プライム市場のみ
        prime = [s for s in info if s.get("MktNm") in ("プライム", "スタンダード")]
        random.shuffle(prime)
        return prime[:TOP_UNIVERSE]

    async def fetch_ohlcv(self, client: JQuantsClient, code: str, days: int = 60) -> pd.DataFrame:
        """日足OHLCVを取得してDataFrameに変換"""
        end = date.today()
        start = end - timedelta(days=days)
        raw = await client.get_prices_daily(code, start.isoformat(), end.isoformat())
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        rename = {"O": "Open", "H": "High", "L": "Low", "C": "Close", "Vo": "Volume", "Date": "Date"}
        # Adjusted prices if available
        if "AdjO" in df.columns:
            rename.update({"AdjO": "Open", "AdjH": "High", "AdjL": "Low", "AdjC": "Close", "AdjVo": "Volume"})
        df = df.rename(columns=rename)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        # 小文字カラム追加（戦略が小文字を期待）
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        # 大文字も残す
        df["Open"] = df["open"]
        df["High"] = df["high"]
        df["Low"] = df["low"]
        df["Close"] = df["close"]
        df["Volume"] = df["volume"]
        return df

    def simulate_current_price(self, df: pd.DataFrame) -> float:
        """最新終値にランダムウォークを加えて現在価格をシミュレート"""
        if df.empty:
            return 0.0
        last_close = float(df["Close"].iloc[-1])
        atr = float((df["High"] - df["Low"]).tail(14).mean()) if len(df) >= 14 else last_close * 0.02
        noise = random.gauss(0, atr * 0.3)
        return round(last_close + noise, 1)

    async def scan_and_trade(self, client: JQuantsClient):
        """全銘柄をスキャンして売買判断"""
        universe = await self.fetch_universe(client)
        strategies = StrategyRegistry.get_active()

        logger.info(f"スキャン開始: {len(universe)}銘柄 x {len(strategies)}戦略")

        signals: list[tuple[TradeSignal, BaseStrategy]] = []

        for stock in universe:
            code = stock.get("Code", "")
            name = stock.get("CoName", "")

            if code in self.positions:
                continue

            try:
                df = await self.fetch_ohlcv(client, code)
                if len(df) < 20:
                    continue

                features = self.fe.calculate_all_features(df)
                current_price = self.simulate_current_price(df)
                features["current_price"] = current_price

                # TDnetイベント注入
                if code in tdnet_events:
                    features["event_type"] = tdnet_events[code]
                    features["event_magnitude"] = 1.0
                    features["historical_event_response"] = 0.5

                for strategy in strategies:
                    try:
                        signal = await strategy.scan(code, df, features)
                        if signal and signal.confidence >= MIN_CONFIDENCE:
                            signals.append((signal, strategy))
                    except Exception as e:
                        logger.debug(f"{strategy.name}/{code}: {e}")

            except Exception as e:
                logger.debug(f"{code} データ取得失敗: {e}")
                continue

            # レート制限対策
            await asyncio.sleep(0.5)

        # 確信度順にソートして上位を実行
        signals.sort(key=lambda x: x[0].confidence, reverse=True)

        executed = 0
        for signal, strategy in signals:
            if len(self.positions) >= MAX_POSITIONS:
                break
            if signal.ticker in self.positions:
                continue

            await self.open_position(signal, strategy)
            executed += 1

        logger.info(f"スキャン完了: シグナル{len(signals)}件, 新規{executed}件")

    async def open_position(self, signal: TradeSignal, strategy: BaseStrategy):
        """ポジションを開く"""
        quantity = max(100, int(POSITION_SIZE / signal.entry_price / 100) * 100)
        slippage = signal.entry_price * 0.001 * (1 if signal.direction == "long" else -1)
        fill_price = round(signal.entry_price + slippage, 1)

        self.positions[signal.ticker] = {
            "strategy": strategy,
            "signal": signal,
            "entry_price": fill_price,
            "entry_time": datetime.now(),
            "quantity": quantity,
            "direction": signal.direction,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        }

        logger.info(
            f"▶ エントリー {signal.ticker} {signal.direction} "
            f"@{fill_price:,.0f} x{quantity} [{strategy.name}] "
            f"理由: {signal.entry_reason}"
        )

        # DBに保存
        await self.db.save_signal_skipped(signal, reason="EXECUTED")

    async def check_exits(self, client: JQuantsClient):
        """保有ポジションの決済判断"""
        to_close = []

        for ticker, pos in self.positions.items():
            try:
                df = await self.fetch_ohlcv(client, ticker, days=30)
                if df.empty:
                    continue

                current = self.simulate_current_price(df)
                features = self.fe.calculate_all_features(df)
                features["current_price"] = current

                holding_min = int((datetime.now() - pos["entry_time"]).total_seconds() / 60)

                # ストップロス
                if pos["direction"] == "long" and current <= pos["stop_loss"]:
                    to_close.append((ticker, current, "ストップロス", features))
                    continue
                if pos["direction"] == "short" and current >= pos["stop_loss"]:
                    to_close.append((ticker, current, "ストップロス", features))
                    continue

                # テイクプロフィット
                if pos["direction"] == "long" and current >= pos["take_profit"]:
                    to_close.append((ticker, current, "利確", features))
                    continue
                if pos["direction"] == "short" and current <= pos["take_profit"]:
                    to_close.append((ticker, current, "利確", features))
                    continue

                # 最大保有時間
                if holding_min >= settings.MAX_HOLDING_MINUTES:
                    to_close.append((ticker, current, "時間切れ", features))
                    continue

                # 戦略の決済判断
                strategy: BaseStrategy = pos["strategy"]
                should_exit, reason = strategy.should_exit(pos, df, features)
                if should_exit:
                    to_close.append((ticker, current, reason, features))

            except Exception as e:
                logger.debug(f"決済チェック失敗 {ticker}: {e}")

            await asyncio.sleep(0.1)

        for ticker, exit_price, reason, features in to_close:
            await self.close_position(ticker, exit_price, reason, features)

    async def close_position(self, ticker: str, exit_price: float, reason: str, features: dict):
        """ポジションを決済"""
        pos = self.positions.pop(ticker, None)
        if not pos:
            return

        slippage = exit_price * 0.001 * (-1 if pos["direction"] == "long" else 1)
        fill_price = round(exit_price + slippage, 1)

        if pos["direction"] == "long":
            pnl = (fill_price - pos["entry_price"]) * pos["quantity"]
        else:
            pnl = (pos["entry_price"] - fill_price) * pos["quantity"]

        pnl_pct = pnl / (pos["entry_price"] * pos["quantity"]) * 100
        holding_min = int((datetime.now() - pos["entry_time"]).total_seconds() / 60)

        self.daily_pnl += pnl
        self.trade_count += 1
        if pnl > 0:
            self.win_count += 1

        win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0

        trade = TradeResult(
            ticker=ticker,
            strategy_name=pos["strategy"].name,
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            entry_time=pos["entry_time"],
            exit_time=datetime.now(),
            pnl=round(pnl, 0),
            pnl_pct=round(pnl_pct, 2),
            holding_minutes=holding_min,
            entry_reason=pos["signal"].entry_reason,
            exit_reason=reason,
            features_at_entry=pos["signal"].features_snapshot,
            features_at_exit=features,
        )

        self.trades.append(trade)
        await self.db.save_trade(trade)

        icon = "✅" if pnl > 0 else "❌"
        logger.info(
            f"{icon} 決済 {ticker} {pos['direction']} "
            f"@{fill_price:,.0f} 損益={pnl:+,.0f}円 ({pnl_pct:+.1f}%) "
            f"[{pos['strategy'].name}] 理由: {reason} "
            f"| 累計: {self.daily_pnl:+,.0f}円 勝率{win_rate:.0%} ({self.trade_count}件)"
        )

    async def extract_knowledge(self):
        """蓄積されたトレードからナレッジを抽出"""
        if len(self.trades) < 3:
            return

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        # 勝ちパターン抽出
        if wins:
            strategies_used = {}
            for t in wins:
                strategies_used.setdefault(t.strategy_name, []).append(t)

            for sname, strades in strategies_used.items():
                if len(strades) >= 2:
                    avg_pnl = sum(t.pnl for t in strades) / len(strades)
                    entry = KnowledgeEntry(
                        category="win_pattern",
                        content=f"{sname}: {len(strades)}勝 平均損益+{avg_pnl:,.0f}円",
                        supporting_trades=[t.id for t in strades],
                        confidence=min(0.9, len(strades) / 10),
                    )
                    await self.db.save_knowledge(entry)

        # 負けパターン抽出
        if losses:
            strategies_used = {}
            for t in losses:
                strategies_used.setdefault(t.strategy_name, []).append(t)

            for sname, strades in strategies_used.items():
                if len(strades) >= 2:
                    avg_loss = sum(t.pnl for t in strades) / len(strades)
                    entry = KnowledgeEntry(
                        category="loss_pattern",
                        content=f"{sname}: {len(strades)}敗 平均損失{avg_loss:,.0f}円",
                        supporting_trades=[t.id for t in strades],
                        confidence=min(0.9, len(strades) / 10),
                    )
                    await self.db.save_knowledge(entry)

        # 戦略別パフォーマンス更新
        all_strategies = {}
        for t in self.trades:
            all_strategies.setdefault(t.strategy_name, []).append(t)

        for sname, strades in all_strategies.items():
            w = sum(1 for t in strades if t.pnl > 0)
            l = sum(1 for t in strades if t.pnl <= 0)
            gross_profit = sum(t.pnl for t in strades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in strades if t.pnl <= 0))
            pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

            perf = StrategyPerformance(
                strategy_name=sname,
                total_trades=len(strades),
                wins=w,
                losses=l,
                win_rate=w / len(strades) if strades else 0,
                profit_factor=round(pf, 2),
                avg_pnl=round(sum(t.pnl for t in strades) / len(strades), 0),
                avg_holding_minutes=round(sum(t.holding_minutes for t in strades) / len(strades), 0),
            )
            await self.db.update_strategy_performance(sname, perf)

        logger.info(f"ナレッジ更新: {len(self.trades)}件のトレードから抽出完了")

    async def run(self):
        """メインループ"""
        await self.init()

        if settings.ALLOW_LIVE_TRADING:
            logger.error("本番取引が有効です。安全のため停止します。")
            return

        logger.info("=" * 60)
        logger.info("ペーパートレード開始")
        logger.info(f"  資金: ¥{settings.TOTAL_CAPITAL:,.0f}")
        logger.info(f"  最大ポジション: {MAX_POSITIONS}")
        logger.info(f"  スキャン間隔: {SCAN_INTERVAL}秒")
        logger.info("=" * 60)

        # TDnet本日のイベント取得
        tdnet_events: dict[str, str] = {}
        try:
            async with TDnetClient() as tdnet:
                disclosures = await tdnet.fetch_today_disclosures()
                material = tdnet.filter_material_events(disclosures)
                for d in material:
                    tdnet_events[d.ticker + "0"] = d.disclosure_type.value
                logger.info(f"TDnet: {len(disclosures)}件の開示, {len(material)}件の重要イベント")
        except Exception as e:
            logger.warning(f"TDnet取得失敗: {e}")

        cycle = 0
        async with JQuantsClient() as client:
            while True:
                cycle += 1
                try:
                    logger.info(f"--- サイクル {cycle} ({datetime.now().strftime('%H:%M:%S')}) ---")

                    # 新規エントリースキャン
                    await self.scan_and_trade(client)

                    # 保有ポジション決済チェック
                    if self.positions:
                        await self.check_exits(client)

                    # 定期的にナレッジ抽出
                    if cycle % 5 == 0 and self.trades:
                        await self.extract_knowledge()

                    status = (
                        f"ポジション: {len(self.positions)} | "
                        f"トレード: {self.trade_count} | "
                        f"損益: ¥{self.daily_pnl:+,.0f} | "
                        f"勝率: {self.win_count}/{self.trade_count}"
                    )
                    logger.info(status)

                except Exception as e:
                    logger.error(f"サイクルエラー: {e}")

                await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    trader = PaperTrader()
    asyncio.run(trader.run())
