"""
ペーパートレード常時稼働スクリプト v2

改修内容:
- random.shuffle廃止 → PreMarketScanner + StockSelector で銘柄選定
- simulate_current_price廃止 → 日足終値ベースの判断に統一
- RegimeDetector統合 → 全戦略にレジーム情報を供給
- 戦略階層: VWAP Reclaim(主), ORB Continuation(補), TrendFollow(フィルタ), SpreadEntry(補助)
- 異常検知: 連敗・急速ドローダウンで一時停止
- 戦略auto on/off: ローリングPFで自動制御
- 銘柄相性学習: strategy×ticker勝率をDB蓄積
"""

import asyncio
import json
import sys
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
from core.safety import SafetyGuard
from db.database import DatabaseManager
from data_sources.jquants import JQuantsClient
from data_sources.tdnet import TDnetClient
from strategies.registry import StrategyRegistry
from strategies.base import BaseStrategy
from strategies.momentum.trend_follow import TrendFollowStrategy
from strategies.orderbook.spread_entry import SpreadEntryStrategy
from tools.feature_engineering import FeatureEngineer
from tools.market_regime import RegimeDetector
from scanners.premarket import PreMarketScanner
from scanners.stock_selector import StockSelector
from core.ticker_map import update_from_jquants, format_ticker

# ── 設定 ──────────────────────────────────────────────────────────────────────
SCAN_INTERVAL = 90   # 秒ごとにスキャン（レート制限対策）
TOP_UNIVERSE = 20    # 上位N銘柄をスキャン（レート制限対策）
MAX_POSITIONS = 5
POSITION_SIZE = 500_000  # 円
MIN_CONFIDENCE = 0.3  # シグナル閾値（日足ベースなので緩めに）

# 戦略階層: 主戦略に高い優先度
STRATEGY_PRIORITY = {
    "vwap_reclaim": 1.20,   # 主戦略: 最優先
    "orb": 1.05,            # 継続型: 準優先
    "gap_go": 1.00,
    "gap_fade": 1.00,
    "tdnet_event": 1.10,    # イベント: 高優先
    "earnings_momentum": 1.10,
    "catalyst_initial": 1.05,
    "vwap_bounce": 0.95,
    "trend_follow": 0.90,   # フィルタ兼用: 抑制
    "spread_entry": 0.85,   # 補助シグナル: 抑制
    "overextension": 0.90,
    "rsi_reversal": 0.90,
    "crash_rebound": 0.90,
}

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/paper_trade.log", rotation="1 day", retention="30 days", level="DEBUG")


class PaperTrader:
    def __init__(self):
        self.db = DatabaseManager()
        self.fe = FeatureEngineer()
        self.regime_detector = RegimeDetector()
        self.stock_selector = StockSelector()
        self.safety = SafetyGuard()
        self.positions: dict[str, dict] = {}  # ticker -> position info
        self.trades: list[TradeResult] = []
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self._eod_done = False
        self._halted = False
        self._halt_reason = ""
        # 当日ウォッチリスト（UI表示用）
        self.watchlist: list[dict] = []
        self.current_regime = None

    async def init(self):
        await self.db.init_db()
        StrategyRegistry.register_all_defaults()
        logger.info(f"戦略 {len(StrategyRegistry.get_active())} 個を読み込み")

    async def build_watchlist(self, client: JQuantsClient, tdnet_events: dict[str, str]) -> list[dict]:
        """PreMarketScanner + StockSelector で当日ウォッチリストを構築"""
        # Step 1: 上場銘柄マスタからプライム/スタンダードを取得
        info = await client.get_listed_info()
        candidates = [s for s in info if s.get("MktNm") in ("プライム", "スタンダード")]
        candidate_codes = [s.get("Code", "") for s in candidates]

        # Step 2: PreMarketScanner でギャップ・出来高・イベントをスキャン
        scanner = PreMarketScanner(client)
        scan_results = await scanner.generate_watchlist(candidate_codes[:200])

        # Step 3: StockSelector で各銘柄をスコアリング
        scored: list[dict] = []
        for result in scan_results:
            code = result.ticker
            try:
                df = await self.fetch_ohlcv(client, code)
                if len(df) < 20:
                    continue
                has_event = code in tdnet_events
                stock_score = self.stock_selector.score_stock(code, df, has_event=has_event)
                if stock_score.excluded:
                    continue
                scored.append({
                    "code": code,
                    "scan_score": result.score,
                    "stock_score": stock_score.total_score,
                    "combined": result.score * 0.4 + stock_score.total_score * 0.6,
                    "reason": result.reason,
                    "has_event": has_event,
                    "gap_pct": stock_score.gap_pct,
                    "relative_volume": stock_score.relative_volume,
                })
            except Exception as e:
                logger.debug(f"銘柄選定失敗 {code}: {e}")
            await asyncio.sleep(0.3)

        # Step 4: combined score で上位N銘柄を選出
        scored.sort(key=lambda x: x["combined"], reverse=True)
        watchlist = scored[:TOP_UNIVERSE]

        if watchlist:
            logger.info(f"ウォッチリスト構築完了: {len(scored)}銘柄中 {len(watchlist)}銘柄を選出")
            for w in watchlist[:5]:
                logger.info(
                    f"  {format_ticker(w['code'])}: score={w['combined']:.3f} "
                    f"gap={w['gap_pct']:.1f}% vol={w['relative_volume']:.1f}x"
                    f"{' [EVENT]' if w['has_event'] else ''}"
                )
        else:
            logger.warning("ウォッチリスト: 条件を満たす銘柄なし")

        return watchlist

    async def fetch_ohlcv(self, client: JQuantsClient, code: str, days: int = 60) -> pd.DataFrame:
        """日足OHLCVを取得してDataFrameに変換"""
        end = date.today()
        start = end - timedelta(days=days)
        raw = await client.get_prices_daily(code, start.isoformat(), end.isoformat())
        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        rename = {"O": "Open", "H": "High", "L": "Low", "C": "Close", "Vo": "Volume", "Date": "Date"}
        if "AdjO" in df.columns:
            rename.update({"AdjO": "Open", "AdjH": "High", "AdjL": "Low", "AdjC": "Close", "AdjVo": "Volume"})
        df = df.rename(columns=rename)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df["Open"] = df["open"]
        df["High"] = df["high"]
        df["Low"] = df["low"]
        df["Close"] = df["close"]
        df["Volume"] = df["volume"]
        return df

    def _detect_regime(self, df: pd.DataFrame):
        """レジーム判定（代表銘柄のDFで実行）"""
        if len(df) >= 20:
            self.current_regime = self.regime_detector.detect(df)
            logger.info(
                f"レジーム: {self.current_regime.regime} "
                f"(確信度{self.current_regime.confidence:.0%})"
            )

    async def scan_and_trade(self, client: JQuantsClient, tdnet_events: dict[str, str]):
        """ウォッチリスト銘柄をスキャンして売買判断"""
        if self._halted:
            logger.warning(f"異常検知により停止中: {self._halt_reason}")
            return

        strategies = StrategyRegistry.get_active()
        logger.info(f"スキャン開始: {len(self.watchlist)}銘柄 x {len(strategies)}戦略")

        signals: list[tuple[TradeSignal, BaseStrategy, float]] = []

        for item in self.watchlist:
            code = item["code"]
            if code in self.positions:
                continue

            try:
                df = await self.fetch_ohlcv(client, code)
                if len(df) < 20:
                    continue

                features = self.fe.calculate_all_features(df)
                # 日足終値をそのまま使用（simulate_current_price廃止）
                current_price = float(df["close"].iloc[-1])
                features["current_price"] = current_price

                # レジーム情報を注入
                if self.current_regime is not None:
                    features["regime_result"] = self.current_regime

                # TDnetイベント注入
                if code in tdnet_events:
                    features["event_type"] = tdnet_events[code]
                    features["event_magnitude"] = 1.0
                    features["historical_event_response"] = 0.5

                # トレンドフィルタ情報を注入
                is_trending, trend_dir, trend_strength = TrendFollowStrategy.is_trending(features)
                features["_is_trending"] = is_trending
                features["_trend_direction"] = trend_dir

                # スプレッド補助シグナルを注入
                spread_boost = SpreadEntryStrategy.get_spread_boost(features)
                features["_spread_boost"] = spread_boost

                # 銘柄スコアを注入
                features["selector_score"] = item.get("stock_score", 0.5)

                for strategy in strategies:
                    try:
                        signal = await strategy.scan(code, df, features)
                        if signal and signal.confidence >= MIN_CONFIDENCE:
                            # 戦略優先度とフィルタを適用
                            priority = STRATEGY_PRIORITY.get(strategy.name, 1.0)
                            adjusted_conf = signal.confidence * priority

                            # トレンドフィルタ: トレンド方向と一致でブースト
                            if is_trending:
                                if (signal.direction == "long" and trend_dir == "up") or \
                                   (signal.direction == "short" and trend_dir == "down"):
                                    adjusted_conf += 0.05 * trend_strength
                                elif (signal.direction == "long" and trend_dir == "down") or \
                                     (signal.direction == "short" and trend_dir == "up"):
                                    adjusted_conf -= 0.05

                            # スプレッド補助: 全戦略にブースト適用
                            adjusted_conf += spread_boost * 0.5

                            # 銘柄相性を確認
                            affinity = await self.db.get_ticker_affinity(strategy.name, code)
                            if affinity and affinity["trades"] >= 5:
                                if affinity["win_rate"] > 0.6:
                                    adjusted_conf += 0.05
                                elif affinity["win_rate"] < 0.3:
                                    adjusted_conf -= 0.10

                            signals.append((signal, strategy, adjusted_conf))
                    except Exception as e:
                        logger.debug(f"{strategy.name}/{code}: {e}")

            except Exception as e:
                logger.debug(f"{code} データ取得失敗: {e}")
                continue

            await asyncio.sleep(0.5)

        # 調整後確信度でソートして上位を実行
        signals.sort(key=lambda x: x[2], reverse=True)

        executed = 0
        for signal, strategy, adj_conf in signals:
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
            f"▶ エントリー {format_ticker(signal.ticker)} {signal.direction} "
            f"@{fill_price:,.0f} x{quantity} [{strategy.name}] "
            f"理由: {signal.entry_reason}"
        )

        await self.db.save_signal_skipped(signal, reason="EXECUTED")

    async def check_exits(self, client: JQuantsClient):
        """保有ポジションの決済判断"""
        to_close = []

        for ticker, pos in self.positions.items():
            try:
                df = await self.fetch_ohlcv(client, ticker, days=30)
                if df.empty:
                    continue

                # 日足終値で判断（simulate_current_price廃止）
                current = float(df["close"].iloc[-1])
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

        regime_str = self.current_regime.regime if self.current_regime else ""

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
            market_condition=regime_str,
        )

        self.trades.append(trade)
        await self.db.save_trade(trade)

        # 銘柄相性学習
        await self.db.update_ticker_affinity(
            pos["strategy"].name, ticker, pnl, pnl > 0,
        )

        icon = "+" if pnl > 0 else "-"
        logger.info(
            f"[{icon}] 決済 {format_ticker(ticker)} {pos['direction']} "
            f"@{fill_price:,.0f} 損益={pnl:+,.0f}円 ({pnl_pct:+.1f}%) "
            f"[{pos['strategy'].name}] 理由: {reason} "
            f"| 累計: {self.daily_pnl:+,.0f}円 勝率{win_rate:.0%} ({self.trade_count}件)"
        )

        # 異常検知チェック
        should_halt, halt_reason = self.safety.check_anomalies(
            self.trades, capital=settings.TOTAL_CAPITAL,
        )
        if should_halt:
            self._halted = True
            self._halt_reason = halt_reason
            logger.warning(f"異常検知: {halt_reason}")

    async def extract_knowledge(self):
        """蓄積されたトレードからナレッジを抽出"""
        if len(self.trades) < 3:
            return

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

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

    def _print_daily_summary(self):
        """日次サマリーをログ出力"""
        logger.info("=" * 60)
        logger.info("日次サマリー")
        logger.info(f"  トレード数: {self.trade_count}")
        win_rate = self.win_count / self.trade_count * 100 if self.trade_count > 0 else 0
        logger.info(f"  勝率: {self.win_count}/{self.trade_count} ({win_rate:.1f}%)")
        logger.info(f"  累計損益: Y{self.daily_pnl:+,.0f}")
        if self.current_regime:
            logger.info(f"  レジーム: {self.current_regime.regime}")

        if self.trades:
            by_strategy: dict[str, list] = {}
            for t in self.trades:
                by_strategy.setdefault(t.strategy_name, []).append(t)
            logger.info("  -- 戦略別 --")
            for sname, strades in sorted(by_strategy.items()):
                s_pnl = sum(t.pnl for t in strades)
                s_wins = sum(1 for t in strades if t.pnl > 0)
                logger.info(f"    {sname}: {len(strades)}件 {s_wins}勝 損益Y{s_pnl:+,.0f}")

        if self._halted:
            logger.info(f"  [!] 異常停止: {self._halt_reason}")

        logger.info("=" * 60)

    def _save_watchlist_json(self):
        """ウォッチリストをJSONで保存（UI用）"""
        data = {
            "date": date.today().isoformat(),
            "regime": self.current_regime.regime if self.current_regime else "unknown",
            "regime_confidence": self.current_regime.confidence if self.current_regime else 0,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "watchlist": self.watchlist[:20],
            "active_strategies": [s.name for s in StrategyRegistry.get_active()],
            "disabled_strategies": [
                s.name for s in StrategyRegistry.get_all() if not s.config.is_active
            ],
            "positions": {
                ticker: {
                    "strategy": pos["strategy"].name,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "entry_time": pos["entry_time"].isoformat(),
                }
                for ticker, pos in self.positions.items()
            },
        }
        Path("knowledge/paper_state.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    async def run(self):
        """メインループ"""
        await self.init()

        if settings.ALLOW_LIVE_TRADING:
            logger.error("本番取引が有効です。安全のため停止します。")
            return

        logger.info("=" * 60)
        logger.info("ペーパートレード v2 開始")
        logger.info(f"  資金: Y{settings.TOTAL_CAPITAL:,.0f}")
        logger.info(f"  最大ポジション: {MAX_POSITIONS}")
        logger.info(f"  スキャン間隔: {SCAN_INTERVAL}秒")
        logger.info(f"  銘柄選定: PreMarketScanner + StockSelector")
        logger.info(f"  擬似intraday: 廃止 (日足終値ベース)")
        logger.info("=" * 60)

        tdnet_events: dict[str, str] = {}
        cycle = 0
        tdnet_fetched_today: str = ""
        watchlist_built_today: str = ""

        async with JQuantsClient() as client:
            # 銘柄名マスタ更新
            try:
                master = await client.get_listed_info()
                update_from_jquants(master)
            except Exception as e:
                logger.warning(f"銘柄名マスタ取得失敗: {e}")

            while True:
                now = datetime.now()
                today_str = now.strftime("%Y-%m-%d")
                market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
                market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                force_close_time = now.replace(hour=15, minute=20, second=0, microsecond=0)

                # ── 市場開場前: 待機 ──
                if now < market_open:
                    pre_open = now.replace(hour=8, minute=55, second=0, microsecond=0)
                    if now >= pre_open and tdnet_fetched_today != today_str:
                        try:
                            async with TDnetClient() as tdnet:
                                disclosures = await tdnet.fetch_today_disclosures()
                                material = tdnet.filter_material_events(disclosures)
                                tdnet_events.clear()
                                for d in material:
                                    tdnet_events[d.ticker + "0"] = d.disclosure_type.value
                                logger.info(f"TDnet: {len(disclosures)}件の開示, {len(material)}件の重要イベント")
                            tdnet_fetched_today = today_str
                        except Exception as e:
                            logger.warning(f"TDnet取得失敗: {e}")

                    wait_sec = (market_open - now).total_seconds()
                    if not hasattr(self, '_pre_market_logged') or self._pre_market_logged != today_str:
                        logger.info(f"市場開場まで {wait_sec/60:.0f}分 待機中...")
                        self._pre_market_logged = today_str
                    await asyncio.sleep(min(wait_sec, 60))
                    continue

                # ── 市場閉場後: EOD処理 → 翌営業日まで長時間スリープ ──
                if now >= market_close:
                    if self.trades and not self._eod_done:
                        await self.extract_knowledge()
                        self._print_daily_summary()
                        self._save_watchlist_json()
                        self._eod_done = True

                    tomorrow_pre = (now + timedelta(days=1)).replace(hour=8, minute=50, second=0, microsecond=0)
                    if now.weekday() == 4:
                        tomorrow_pre += timedelta(days=2)
                    elif now.weekday() == 5:
                        tomorrow_pre += timedelta(days=1)
                    wait_sec = (tomorrow_pre - now).total_seconds()
                    if not hasattr(self, '_post_market_logged') or self._post_market_logged != today_str:
                        logger.info(f"市場閉場。次回 {tomorrow_pre.strftime('%m/%d %H:%M')} まで停止")
                        self._post_market_logged = today_str
                    await asyncio.sleep(max(wait_sec, 60))
                    continue

                # ── 市場時間中 ──
                self._eod_done = False

                # 当日初回: ウォッチリスト構築 + レジーム判定 + 日次リセット
                if watchlist_built_today != today_str:
                    # 日次リセット
                    self.daily_pnl = 0.0
                    self.trade_count = 0
                    self.win_count = 0
                    self.trades.clear()
                    self._halted = False
                    self._halt_reason = ""

                    logger.info("当日ウォッチリスト構築中...")
                    self.watchlist = await self.build_watchlist(client, tdnet_events)

                    # レジーム判定（ウォッチリスト先頭銘柄のDFで実施）
                    if self.watchlist:
                        regime_df = await self.fetch_ohlcv(client, self.watchlist[0]["code"])
                        self._detect_regime(regime_df)

                    # 戦略auto on/off
                    all_recent = await self.db.get_trades(limit=100)
                    toggled = StrategyRegistry.auto_toggle(all_recent)
                    if toggled:
                        logger.info(f"戦略auto toggle: {toggled}")

                    self._save_watchlist_json()
                    watchlist_built_today = today_str

                # 15:20 以降: 全ポジション強制決済
                if now >= force_close_time and self.positions:
                    logger.info("15:20 全ポジション強制決済")
                    for ticker in list(self.positions.keys()):
                        try:
                            df = await self.fetch_ohlcv(client, ticker, days=30)
                            if df.empty:
                                continue
                            current = float(df["close"].iloc[-1])
                            features = self.fe.calculate_all_features(df)
                            features["current_price"] = current
                            await self.close_position(ticker, current, "大引け強制決済", features)
                        except Exception as e:
                            logger.error(f"強制決済失敗 {ticker}: {e}")
                        await asyncio.sleep(0.3)
                    continue

                # 通常トレードサイクル
                cycle += 1
                try:
                    logger.info(f"--- サイクル {cycle} ({now.strftime('%H:%M:%S')}) ---")

                    # TDnet未取得なら取得
                    if tdnet_fetched_today != today_str:
                        try:
                            async with TDnetClient() as tdnet:
                                disclosures = await tdnet.fetch_today_disclosures()
                                material = tdnet.filter_material_events(disclosures)
                                tdnet_events.clear()
                                for d in material:
                                    tdnet_events[d.ticker + "0"] = d.disclosure_type.value
                                logger.info(f"TDnet: {len(disclosures)}件の開示, {len(material)}件の重要イベント")
                            tdnet_fetched_today = today_str
                        except Exception as e:
                            logger.warning(f"TDnet取得失敗: {e}")

                    # 新規エントリースキャン
                    await self.scan_and_trade(client, tdnet_events)

                    # 保有ポジション決済チェック
                    if self.positions:
                        await self.check_exits(client)

                    # 定期的にナレッジ抽出
                    if cycle % 5 == 0 and self.trades:
                        await self.extract_knowledge()

                    # 定期的にUI用状態保存
                    if cycle % 3 == 0:
                        self._save_watchlist_json()

                    status = (
                        f"ポジション: {len(self.positions)} | "
                        f"トレード: {self.trade_count} | "
                        f"損益: Y{self.daily_pnl:+,.0f} | "
                        f"勝率: {self.win_count}/{self.trade_count}"
                    )
                    if self.current_regime:
                        status += f" | レジーム: {self.current_regime.regime}"
                    logger.info(status)

                except Exception as e:
                    logger.error(f"サイクルエラー: {e}")

                await asyncio.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    trader = PaperTrader()
    asyncio.run(trader.run())
