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
from tools.feature_engineering import FeatureEngineer, JST
from tools.market_regime import RegimeDetector
from scanners.premarket import PreMarketScanner
from scanners.stock_selector import StockSelector
from tools.sector_bias import SectorBiasCalculator, SectorBiasResult
from tools.disclosure_analyzer import DisclosureAnalyzer
from tools.telegram_notify import TelegramNotifier
from core.ticker_map import update_from_jquants, format_ticker

# ── 設定 ── 攻撃的チューニング (2026-03-25) ──────────────────────────────
# 目標: 3万円→6万円/月 (月利100%)
SCAN_INTERVAL = 90        # 90秒間隔
TOP_UNIVERSE = 30         # 30銘柄
MAX_POSITIONS = 3         # 同時3ポジション: 機会最大化
POSITION_SIZE = 30_000    # 3万円: 1ポジション=全資金
MIN_CONFIDENCE = 0.30     # 0.30: シグナルを広く拾う

# 戦略階層: vwap_reclaim 一本に集中
# NOTE: 他戦略は registry で off にするが、priority は残す（将来用）
STRATEGY_PRIORITY = {
    "vwap_reclaim": 1.20,   # 唯一の active 主戦略
    "tdnet_event": 1.10,    # watch: イベント時のみ参考
    "orb": 1.00,            # watch: 将来再開用
    "gap_go": 0.90,         # watch
    "trend_follow": 0.00,   # filter only: 単独発火しない
    "spread_entry": 0.00,   # supplement only: 単独発火しない
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
        self.sector_bias_calc = SectorBiasCalculator()
        self.notifier = TelegramNotifier()
        self.sector_bias: SectorBiasResult | None = None
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
        """PreMarketScanner + StockSelector + SectorBias で当日ウォッチリストを構築"""
        from run_backtest_learn import CANDIDATE_CODES
        # Step 0: 米国セクターバイアスを計算
        try:
            self.sector_bias = await self.sector_bias_calc.calculate()
            if self.sector_bias.fetch_success:
                logger.info(
                    f"セクターバイアス取得: SPY={self.sector_bias.spy_return:+.2%} "
                    f"risk_off={self.sector_bias.risk_off}"
                )
        except Exception as e:
            logger.warning(f"セクターバイアス取得失敗: {e}")
            self.sector_bias = None

        # Step 1: 上場銘柄マスタからプライム/スタンダードを取得
        info = await client.get_listed_info()
        candidates = [s for s in info if s.get("MktNm") in ("プライム", "スタンダード")]
        candidate_codes = [s.get("Code", "") for s in candidates]
        # S33マップを構築 (Code → S33コード)
        code_to_s33 = {s.get("Code", ""): s.get("S33", "") for s in info}

        # Step 2: PreMarketScanner でギャップ・出来高・イベントをスキャン
        # 有料プラン + 一括プリロード済み: 全プライム/スタンダード銘柄をスキャン
        priority_codes = [c for c in CANDIDATE_CODES if c in candidate_codes]
        remaining = [c for c in candidate_codes if c not in priority_codes]
        scan_target = priority_codes + remaining
        logger.info(f"PreMarketScan対象: {len(scan_target)}銘柄 (優先{len(priority_codes)} + 補完{len(remaining)})")
        scanner = PreMarketScanner(client, price_cache=self._raw_price_cache)
        scan_results = await scanner.generate_watchlist(scan_target)

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
                # Sector bias adjustment
                s33 = code_to_s33.get(code, "")
                bias_adj = 0.0
                bias_label = "neutral"
                if self.sector_bias and self.sector_bias.fetch_success and s33:
                    bias_adj = self.sector_bias_calc.get_watchlist_adjustment(self.sector_bias, s33)
                    bias_label = self.sector_bias.get_bias_label(s33)

                wl_clk = datetime.now(JST)
                wl_feat = self.fe.calculate_all_features(df, clock=wl_clk)

                base_combined = result.score * 0.4 + stock_score.total_score * 0.6
                scored.append({
                    "code": code,
                    "scan_score": result.score,
                    "stock_score": stock_score.total_score,
                    "combined": base_combined + bias_adj,
                    "reason": result.reason,
                    "has_event": has_event,
                    "gap_pct": stock_score.gap_pct,
                    "relative_volume": stock_score.relative_volume,
                    "s33": s33,
                    "sector_bias": round(bias_adj, 4),
                    "sector_bias_label": bias_label,
                    "pm_relative_volume": wl_feat.get("pm_relative_volume"),
                    "pm_turnover": wl_feat.get("pm_turnover"),
                    "pm_vwap_reclaim_flag": wl_feat.get("pm_vwap_reclaim_flag"),
                    "pm_vwap_hold_count": wl_feat.get("pm_vwap_hold_count"),
                    "pm_low_price_bonus": wl_feat.get("pm_low_price_bonus"),
                    "pm_intraday_quality_score": wl_feat.get("pm_intraday_quality_score"),
                    "pm_intraday_is_proxy": wl_feat.get("pm_intraday_is_proxy"),
                    "pm_setup_bonus": stock_score.pm_setup_bonus,
                })
            except Exception as e:
                logger.debug(f"銘柄選定失敗 {code}: {e}")

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
            # フォールバック: PreMarketScan で 0 件でも CANDIDATE_CODES 上位を採用
            logger.warning("ウォッチリスト: PreMarketScan 0件 → CANDIDATE_CODES フォールバック")
            fb_tried = 0
            fb_excluded = 0
            # TOP_UNIVERSE の3倍を試行 (レートリミットでの失敗を考慮)
            for code in CANDIDATE_CODES[:TOP_UNIVERSE * 3]:
                if len(scored) >= TOP_UNIVERSE:
                    break
                try:
                    df = await self.fetch_ohlcv(client, code)
                    if len(df) < 20:
                        continue
                    has_event = code in tdnet_events
                    stock_score = self.stock_selector.score_stock(code, df, has_event=has_event)
                    fb_tried += 1
                    if stock_score.excluded:
                        fb_excluded += 1
                        logger.debug(f"フォールバック除外 {code}: {stock_score.exclude_reason}")
                        continue
                    s33 = code_to_s33.get(code, "")
                    bias_adj = 0.0
                    bias_label = "neutral"
                    if self.sector_bias and self.sector_bias.fetch_success and s33:
                        bias_adj = self.sector_bias_calc.get_watchlist_adjustment(self.sector_bias, s33)
                        bias_label = self.sector_bias.get_bias_label(s33)
                    wl_clk = datetime.now(JST)
                    wl_feat = self.fe.calculate_all_features(df, clock=wl_clk)
                    scored.append({
                        "code": code,
                        "scan_score": 0.0,
                        "stock_score": stock_score.total_score,
                        "combined": stock_score.total_score + bias_adj,
                        "reason": "CANDIDATE fallback",
                        "has_event": has_event,
                        "gap_pct": stock_score.gap_pct,
                        "relative_volume": stock_score.relative_volume,
                        "s33": s33,
                        "sector_bias": round(bias_adj, 4),
                        "sector_bias_label": bias_label,
                        "pm_relative_volume": wl_feat.get("pm_relative_volume"),
                        "pm_turnover": wl_feat.get("pm_turnover"),
                        "pm_vwap_reclaim_flag": wl_feat.get("pm_vwap_reclaim_flag"),
                        "pm_vwap_hold_count": wl_feat.get("pm_vwap_hold_count"),
                        "pm_low_price_bonus": wl_feat.get("pm_low_price_bonus"),
                        "pm_intraday_quality_score": wl_feat.get("pm_intraday_quality_score"),
                        "pm_intraday_is_proxy": wl_feat.get("pm_intraday_is_proxy"),
                        "pm_setup_bonus": stock_score.pm_setup_bonus,
                    })
                except Exception as e:
                    logger.debug(f"フォールバック銘柄取得失敗 {code}: {e}")
                    continue
            logger.info(f"フォールバック: {fb_tried}銘柄試行, {fb_excluded}除外, {len(scored)}銘柄scored")
            scored.sort(key=lambda x: x["combined"], reverse=True)
            watchlist = scored[:TOP_UNIVERSE]
            if watchlist:
                logger.info(f"フォールバック採用: {len(watchlist)}銘柄")
                for w in watchlist[:3]:
                    logger.info(f"  {w['code']}: score={w['combined']:.3f}")

        return watchlist

    _ohlcv_cache: dict[str, pd.DataFrame] = {}  # 日中キャッシュ (当日限り)
    _raw_price_cache: dict[str, list[dict]] = {}  # PreMarketScanner用raw形式キャッシュ

    async def preload_bulk_prices(self, client: JQuantsClient, days: int = 60) -> None:
        """全銘柄の日足を一括取得してキャッシュにプリロード。

        J-Quants V2 の date パラメータで日ごとに全銘柄の日足を1回のAPIコールで取得。
        個別リクエストの代わりに使うことで API 呼び出し回数を大幅削減。
        """
        end = date.today()
        start = end - timedelta(days=days)
        logger.info(f"一括日足プリロード開始: {start} → {end}")

        # 日ごとに一括取得
        bulk_data: dict[str, list[dict]] = {}  # code -> [records]
        d = start
        while d <= end:
            try:
                records = await client.get_prices_daily_bulk(d.isoformat())
                for rec in records:
                    code = rec.get("Code", "")
                    if code:
                        bulk_data.setdefault(code, []).append(rec)
            except Exception as e:
                logger.debug(f"一括取得失敗 {d}: {e}")
            d += timedelta(days=1)

        # コードごとにDataFrameに変換してキャッシュ
        loaded = 0
        for code, raw in bulk_data.items():
            df = self._raw_to_df(raw)
            if not df.empty:
                self._ohlcv_cache[f"{code}_{days}"] = df
                self._raw_price_cache[code] = raw  # PreMarketScanner用
                loaded += 1

        logger.info(f"一括プリロード完了: {loaded}銘柄キャッシュ済み ({len(bulk_data)}銘柄取得)")

    @staticmethod
    def _raw_to_df(raw: list[dict]) -> pd.DataFrame:
        """生データをOHLCV DataFrameに変換"""
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw)
        if "AdjO" in df.columns:
            df = df.rename(columns={
                "AdjO": "open", "AdjH": "high", "AdjL": "low",
                "AdjC": "close", "AdjVo": "volume",
            })
        else:
            df = df.rename(columns={
                "O": "open", "H": "high", "L": "low",
                "C": "close", "Vo": "volume",
            })
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
            df = df[["Date", "open", "high", "low", "close", "volume"]].dropna()
        return df

    async def fetch_intraday_today(self, client: JQuantsClient, code: str) -> pd.DataFrame | None:
        """当日の分足（後場 PM-VWAP 用）。取得失敗時は None。"""
        try:
            raw = await client.get_prices_intraday(code, date.today().isoformat())
            if not raw:
                return None
            idf = FeatureEngineer.intraday_records_to_df(raw)
            return idf if not idf.empty else None
        except Exception as e:
            logger.debug(f"intraday取得失敗 {code}: {e}")
            return None

    async def fetch_ohlcv(self, client: JQuantsClient, code: str, days: int = 60) -> pd.DataFrame:
        """日足OHLCVを取得してDataFrameに変換 (プリロードキャッシュ優先)"""
        cache_key = f"{code}_{days}"
        if cache_key in self._ohlcv_cache:
            return self._ohlcv_cache[cache_key]
        # キャッシュミス: 個別取得にフォールバック
        end = date.today()
        start = end - timedelta(days=days)
        raw = await client.get_prices_daily(code, start.isoformat(), end.isoformat())
        df = self._raw_to_df(raw)
        if not df.empty:
            self._ohlcv_cache[cache_key] = df
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

                now = datetime.now(JST)
                features = self.fe.calculate_all_features(
                    df,
                    clock=now,
                )
                # 日足終値をそのまま使用（simulate_current_price廃止）
                current_price = float(df["close"].iloc[-1])
                features["current_price"] = current_price

                # レジーム情報を注入
                if self.current_regime is not None:
                    features["regime_result"] = self.current_regime

                # TDnetイベント注入
                if code in tdnet_events:
                    features["event_type"] = tdnet_events[code]
                    # LLM 分析結果があれば使用、なければフォールバック
                    analysis = getattr(self, '_disclosure_analysis', {}).get(code)
                    if analysis and analysis.analyzed:
                        features["event_magnitude"] = analysis.magnitude
                        features["event_direction"] = analysis.direction
                        features["event_category"] = analysis.category
                    else:
                        features["event_magnitude"] = 0.5  # フォールバック (旧: 1.0)
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

                # Sector bias: 米国→日本セクターバイアスを注入
                features["_sector_bias_score"] = item.get("sector_bias", 0.0)
                features["_sector_bias_label"] = item.get("sector_bias_label", "neutral")

                for strategy in strategies:
                    try:
                        signal = await strategy.scan(code, df, features)
                        if signal:
                            logger.info(f"  [{strategy.name}] {code} conf={signal.confidence:.3f} (min={MIN_CONFIDENCE})")
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
        quantity = max(1, int(POSITION_SIZE / signal.entry_price))  # 1株単位
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

        now = datetime.now()
        table = (
            f"\n{'='*62}\n"
            f"  ENTRY\n"
            f"{'='*62}\n"
            f"  {'時刻':<8} | {now:%Y-%m-%d %H:%M:%S}\n"
            f"  {'銘柄':<8} | {format_ticker(signal.ticker)}\n"
            f"  {'方向':<8} | {signal.direction}\n"
            f"  {'約定価格':<6} | {fill_price:>12,.0f} 円\n"
            f"  {'数量':<8} | {quantity:>12,} 株\n"
            f"  {'金額':<8} | {fill_price * quantity:>12,.0f} 円\n"
            f"  {'戦略':<8} | {strategy.name}\n"
            f"  {'SL':<8} | {signal.stop_loss:>12,.0f} 円\n"
            f"  {'TP':<8} | {signal.take_profit:>12,.0f} 円\n"
            f"  {'理由':<8} | {signal.entry_reason}\n"
            f"{'='*62}"
        )
        logger.info(table)

        try:
            await self.db.save_signal_skipped(signal, reason="EXECUTED")
        except Exception as e:
            logger.debug(f"シグナル保存スキップ: {e}")

        await self.notifier.notify_entry(
            ticker=format_ticker(signal.ticker),
            direction=signal.direction,
            price=fill_price,
            quantity=quantity,
            strategy=strategy.name,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reason=signal.entry_reason,
        )

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
                features = self.fe.calculate_all_features(df, clock=datetime.now(JST))
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

        icon = "WIN" if pnl > 0 else "LOSE"
        now = datetime.now()
        table = (
            f"\n{'='*62}\n"
            f"  EXIT  [{icon}]\n"
            f"{'='*62}\n"
            f"  {'時刻':<8} | {now:%Y-%m-%d %H:%M:%S}\n"
            f"  {'銘柄':<8} | {format_ticker(ticker)}\n"
            f"  {'方向':<8} | {pos['direction']}\n"
            f"  {'参入時刻':<6} | {pos['entry_time']:%Y-%m-%d %H:%M:%S}\n"
            f"  {'参入価格':<6} | {pos['entry_price']:>12,.0f} 円\n"
            f"  {'決済価格':<6} | {fill_price:>12,.0f} 円\n"
            f"  {'数量':<8} | {pos['quantity']:>12,} 株\n"
            f"  {'損益':<8} | {pnl:>+12,.0f} 円 ({pnl_pct:+.1f}%)\n"
            f"  {'保有時間':<6} | {holding_min:>12,} 分\n"
            f"  {'戦略':<8} | {pos['strategy'].name}\n"
            f"  {'理由':<8} | {reason}\n"
            f"{'-'*62}\n"
            f"  {'累計損益':<6} | {self.daily_pnl:>+12,.0f} 円\n"
            f"  {'勝率':<8} | {win_rate:>11.0%} ({self.trade_count}件)\n"
            f"{'='*62}"
        )
        logger.info(table)

        await self.notifier.notify_exit(
            ticker=format_ticker(ticker),
            direction=pos["direction"],
            entry_price=pos["entry_price"],
            exit_price=fill_price,
            quantity=pos["quantity"],
            pnl=pnl,
            pnl_pct=pnl_pct,
            reason=reason,
            daily_pnl=self.daily_pnl,
            win_rate=win_rate,
            trade_count=self.trade_count,
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
        # Proxy & status summary
        proxy_summary = StrategyRegistry.get_proxy_summary()
        status_summary = StrategyRegistry.get_status_summary()

        data = {
            "date": date.today().isoformat(),
            "regime": self.current_regime.regime if self.current_regime else "unknown",
            "regime_confidence": self.current_regime.confidence if self.current_regime else 0,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "sector_bias": self.sector_bias.to_dict() if self.sector_bias else None,
            "watchlist": self.watchlist[:20],
            "active_strategies": [s.name for s in StrategyRegistry.get_active()],
            "disabled_strategies": [
                s.name for s in StrategyRegistry.get_all() if not s.config.is_active
            ],
            "strategy_status": status_summary,
            "proxy_summary": {
                name: {
                    "status": info["status"],
                    "proxy_usage_rate": info["proxy_usage_rate"],
                    "proxy_penalty": info["proxy_penalty"],
                }
                for name, info in proxy_summary.items()
            },
            "positions": {
                ticker: {
                    "strategy": pos["strategy"].name,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "entry_time": pos["entry_time"].isoformat(),
                }
                for ticker, pos in self.positions.items()
            },
            "data_quality_warnings": [
                "intraday proxy features are estimated from daily OHLCV when minute bars unavailable",
                "PM-VWAP reclaim requires real intraday for full quality; check pm_intraday_is_proxy",
                "proxy-dependent strategies have limited evaluation reliability",
            ],
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

        # 市場時間中は日足データが更新される可能性があるためキャッシュを短縮
        async with JQuantsClient(cache_ttl_seconds=300) as client:  # 5分キャッシュ
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
                            disclosure_analyzer = DisclosureAnalyzer()
                            async with TDnetClient() as tdnet:
                                disclosures = await tdnet.fetch_today_disclosures()
                                material = tdnet.filter_material_events(disclosures)
                                tdnet_events.clear()
                                for d in material:
                                    key = d.ticker + "0"
                                    # LLM で開示内容を分析
                                    analysis = disclosure_analyzer.analyze(d.title, d.company_name)
                                    tdnet_events[key] = d.disclosure_type.value
                                    # 分析結果を _disclosure_analysis に保存
                                    if not hasattr(self, '_disclosure_analysis'):
                                        self._disclosure_analysis = {}
                                    self._disclosure_analysis[key] = analysis
                                logger.info(
                                    f"TDnet: {len(disclosures)}件の開示, {len(material)}件の重要イベント"
                                    f" (LLM分析済: {sum(1 for a in getattr(self, '_disclosure_analysis', {}).values() if a.analyzed)}件)"
                                )
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
                    self._ohlcv_cache.clear()  # 日次キャッシュクリア
                    self._raw_price_cache.clear()

                    # 全銘柄日足を一括プリロード (API呼び出し大幅削減)
                    await self.preload_bulk_prices(client, days=60)

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
                            features = self.fe.calculate_all_features(df, clock=datetime.now(JST))
                            features["current_price"] = current
                            await self.close_position(ticker, current, "大引け強制決済", features)
                        except Exception as e:
                            logger.error(f"強制決済失敗 {ticker}: {e}")
                    continue

                # 通常トレードサイクル
                cycle += 1
                try:
                    logger.info(f"--- サイクル {cycle} ({now.strftime('%H:%M:%S')}) ---")

                    # TDnet未取得なら取得 (LLM分析付き)
                    if tdnet_fetched_today != today_str:
                        try:
                            disclosure_analyzer = DisclosureAnalyzer()
                            async with TDnetClient() as tdnet:
                                disclosures = await tdnet.fetch_today_disclosures()
                                material = tdnet.filter_material_events(disclosures)
                                tdnet_events.clear()
                                for d in material:
                                    key = d.ticker + "0"
                                    tdnet_events[key] = d.disclosure_type.value
                                    analysis = disclosure_analyzer.analyze(d.title, d.company_name)
                                    if not hasattr(self, '_disclosure_analysis'):
                                        self._disclosure_analysis = {}
                                    self._disclosure_analysis[key] = analysis
                                logger.info(f"TDnet: {len(disclosures)}件の開示, {len(material)}件の重要イベント")
                            tdnet_fetched_today = today_str
                        except Exception as e:
                            logger.warning(f"TDnet取得失敗: {e}")

                    # 日足データは日中不変のためキャッシュ維持 (一括プリロード済み)

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
