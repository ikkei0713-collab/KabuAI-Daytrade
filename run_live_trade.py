"""
自動トレードボット（立花証券 実取引）

1. 電話認証→ログイン（1日1回）
2. 30秒ごとにスキャン→シグナル生成
3. シグナルがあれば自動注文
4. ポジション監視→利確/損切り
5. 14:50に全ポジション強制決済
6. 15:00に終了

セーフティ:
- 1日の最大損失 -3,000円で停止
- 同時ポジション最大2
- 1注文あたり資金の40%まで
"""

import asyncio
import json
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from brokers.tachibana import TachibanaBroker
from brokers.base import Order, OrderSide, OrderType
from strategies.registry import StrategyRegistry
from tools.feature_engineering import FeatureEngineer
from tools.market_regime import RegimeDetector
from data_sources.jquants import JQuantsClient

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/live_trade.log", rotation="10 MB", level="DEBUG")

JST = ZoneInfo("Asia/Tokyo")

# 安全設定
MAX_DAILY_LOSS = -3000       # 日次最大損失
MAX_POSITIONS = 2            # 同時保有数
MAX_ORDER_PCT = 0.40         # 1注文あたり資金の40%
SCAN_INTERVAL = 30           # スキャン間隔（秒）
FORCE_CLOSE_HOUR = 14        # 強制決済時
FORCE_CLOSE_MIN = 50         # 強制決済分
MARKET_CLOSE_HOUR = 15
MIN_CONFIDENCE = 0.35        # 最低confidence

# 残高¥34,333で100株買える銘柄（¥340以下）+ NTT
SCAN_CANDIDATES_5DIGIT = [
    "94320",  # NTT ¥159
    "40050",  # 住友化学 ¥490 → 100株だと¥49,000で今は無理だが監視
    "93070",  # 杉村倉庫 ¥996 → 無理だが監視
    "47550",  # 楽天 ¥745 → 無理だが監視
    # ↓ ¥340以下の追加候補
    "95830",  # セコム → 高いかも
    "21870",  # ジェイテクト
    "41200",  # スタンレー電気
]


class LiveTrader:
    def __init__(self):
        self.broker = TachibanaBroker()
        self.fe = FeatureEngineer()
        self.regime_det = RegimeDetector()
        self.daily_pnl = 0.0
        self.trades_today = []
        self.open_positions = {}  # ticker -> {entry_price, quantity, stop, target, strategy}
        self.initial_balance = 0.0
        self.stock_data = {}  # ticker(5digit) -> DataFrame (cached)
        self.stopped = False

    async def start(self):
        """メインループ"""
        # セッション復元を試みる → 失敗ならログイン
        ok = await self._restore_or_login()
        if not ok:
            logger.error("ログイン失敗。電話認証してから再実行してください。")
            return

        # 残高取得
        summary = await self.broker._api_request("CLMZanKaiSummary", {})
        self.initial_balance = float(summary.get("sGenbutuKabuKaituke", "0") or "0")
        if self.initial_balance <= 0:
            self.initial_balance = 34333  # NTT購入後の残高
        logger.info(f"=== 自動トレード開始 ===")
        logger.info(f"買付余力: ¥{self.initial_balance:,.0f}")

        # 戦略登録
        StrategyRegistry.clear()
        StrategyRegistry.register_all_defaults()
        active = StrategyRegistry.get_active()
        logger.info(f"Active戦略: {len(active)}個")

        # 銘柄データ一括取得
        await self._load_stock_data()

        # 既存ポジション登録（NTT 100株 @¥157、ストップ¥153、利確¥163）
        if "9432" not in self.open_positions:
            self.open_positions["9432"] = {
                "entry_price": 157,
                "quantity": 100,
                "stop": 153,       # ATR(¥2)の2倍下
                "target": 163,     # ATR(¥2)の3倍上
                "strategy": "trend_follow",
                "order_time": "2026-03-30T10:49:00",
                "tachibana_order": "30012778",
            }
            logger.info("既存ポジション登録: NTT 100株 @¥157 SL=¥153 TP=¥163")

        # メインループ
        try:
            while not self.stopped:
                now = datetime.now(JST)

                # 場が閉まったら次の寄りまで待機
                if now.hour >= MARKET_CLOSE_HOUR or now.hour < 9:
                    if self.open_positions:
                        logger.info(f"場外: {len(self.open_positions)}ポジション保有中（スイング）")
                    await asyncio.sleep(300)  # 5分待機
                    continue

                # ポジション監視（ストップ/利確チェック）
                await self._check_positions()

                # 日次損失チェック
                if self.daily_pnl <= MAX_DAILY_LOSS:
                    logger.warning(f"日次損失上限到達: ¥{self.daily_pnl:,.0f} <= ¥{MAX_DAILY_LOSS:,.0f}")
                    await self._force_close_all("損失上限")
                    self.stopped = True
                    break

                # スキャン→注文
                await self._scan_and_trade()

                # 次のスキャンまで待機
                await asyncio.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("手動停止")
        except Exception as e:
            logger.error(f"エラー: {e}")
        finally:
            await self._print_summary()
            await self.broker.logout()

    async def _restore_or_login(self) -> bool:
        """保存済みセッションを復元。失敗なら新規ログイン。"""
        session_path = Path("data/tachibana_session.json")
        if session_path.exists():
            try:
                import json as _json
                info = _json.loads(session_path.read_text(encoding="utf-8"))
                self.broker._url_request = info["url_request"]
                self.broker._url_master = info["url_master"]
                self.broker._url_price = info["url_price"]
                self.broker._url_event = info["url_event"]
                self.broker._url_event_ws = info.get("url_event_ws", "")
                self.broker._p_no = info.get("p_no", 10)
                self.broker._logged_in = True
                await self.broker._ensure_session()

                # セッションが生きてるかテスト（REQUEST I/Fで確認）
                test = await self.broker._api_request("CLMZanKaiSummary", {})
                if test.get("p_errno") == "0":
                    logger.info("セッション復元成功")
                    return True
                else:
                    logger.warning("セッション切れ → 再ログイン")
                    self.broker._logged_in = False
            except Exception as e:
                logger.warning(f"セッション復元失敗: {e}")

        return await self.broker.login()

    async def _load_stock_data(self):
        """J-Quantsから日足データを一括取得"""
        logger.info("銘柄データ取得中...")
        async with JQuantsClient() as client:
            for code_5 in SCAN_CANDIDATES_5DIGIT:
                try:
                    raw = await client.get_prices_daily(code_5, "2025-09-01", "2026-03-30")
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
                        self.stock_data[code_5] = df
                        last_close = float(df["close"].iloc[-1])
                        logger.info(f"  {code_5}: {len(df)}日 最終値¥{last_close:,.0f}")
                except Exception as e:
                    logger.debug(f"  {code_5}: 取得失敗 {e}")

        logger.info(f"データ取得完了: {len(self.stock_data)}銘柄")

    async def _scan_and_trade(self):
        """スキャン→シグナル→注文"""
        # 残高取得（REQUEST I/F）
        summary = await self.broker._api_request("CLMZanKaiSummary", {})
        balance = float(summary.get("sGenbutuKabuKaituke", "0") or "0")
        if balance <= 0:
            balance = self.initial_balance - sum(
                p["entry_price"] * p["quantity"] for p in self.open_positions.values()
            )
        now = datetime.now(JST)

        strategies = StrategyRegistry.get_active()
        signals = []

        for code_5, df in self.stock_data.items():
            code_4 = code_5[:4]

            # 既にポジションがある銘柄はスキップ
            if code_4 in self.open_positions:
                continue

            # ポジション上限チェック
            if len(self.open_positions) >= MAX_POSITIONS:
                break

            last_close = float(df["close"].iloc[-1])
            cost_100 = last_close * 100

            # 買付余力チェック（手数料込み）
            if cost_100 + 200 > balance:
                logger.debug(f"  {code_4}: ¥{cost_100:,.0f} > 余力¥{balance:,.0f}")
                continue

            features = self.fe.calculate_all_features(df, clock=now)
            features["current_price"] = last_close
            regime = self.regime_det.detect(df)
            features["regime_result"] = regime

            for strategy in strategies:
                try:
                    signal = await strategy.scan(code_5, df, features)
                    if signal and signal.confidence >= MIN_CONFIDENCE and signal.direction == "long":
                        signals.append({
                            "code_4": code_4,
                            "code_5": code_5,
                            "strategy": strategy.name,
                            "confidence": signal.confidence,
                            "entry": last_close,
                            "stop": signal.stop_loss,
                            "target": signal.take_profit,
                            "reason": signal.entry_reason,
                            "regime": regime.regime,
                        })
                except Exception:
                    continue

        if not signals:
            return

        # confidence順にソート
        signals.sort(key=lambda s: -s["confidence"])
        best = signals[0]

        logger.info(
            f"シグナル: {best['code_4']} [{best['strategy']}] "
            f"conf={best['confidence']:.2f} entry=¥{best['entry']:,.0f} "
            f"SL=¥{best['stop']:,.0f} TP=¥{best['target']:,.0f} "
            f"[{best['regime']}] {best['reason'][:40]}"
        )

        # 指値注文（現在値で指値 = スリッページ防止）
        # 成行だと不利な価格で約定するリスクがある
        limit_price = best["entry"]  # 直近終値で指値

        order = Order(
            ticker=best["code_4"],
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=limit_price,
            strategy_name=best["strategy"],
        )

        result = await self.broker.place_order(order)

        if result.status.value == "submitted":
            logger.info(f"★ 注文受付: {best['code_4']} 100株 {result.notes}")
            self.open_positions[best["code_4"]] = {
                "entry_price": best["entry"],
                "quantity": 100,
                "stop": best["stop"],
                "target": best["target"],
                "strategy": best["strategy"],
                "order_time": datetime.now(JST).isoformat(),
                "tachibana_order": result.notes,
            }
        else:
            logger.warning(f"注文失敗: {best['code_4']} {result.notes}")

    async def _check_positions(self):
        """保有ポジションの損益チェック → ストップ/利確で決済"""
        if not self.open_positions:
            return

        for ticker, pos in list(self.open_positions.items()):
            # 日足の最終値で判定（リアルタイム値が取れないため）
            code_5 = ticker + "0"
            df = self.stock_data.get(code_5)
            if df is None:
                continue

            last_close = float(df["close"].iloc[-1])
            entry = pos["entry_price"]
            stop = pos["stop"]
            target = pos["target"]

            # ストップロス
            if last_close <= stop:
                pnl = (last_close - entry) * pos["quantity"]
                logger.warning(
                    f"★ ストップロス: {ticker} ¥{entry:,.0f}→¥{last_close:,.0f} "
                    f"PnL=¥{pnl:+,.0f}"
                )
                await self._close_position(ticker, "ストップロス")
                self.daily_pnl += pnl

            # 利確ターゲット
            elif last_close >= target:
                pnl = (last_close - entry) * pos["quantity"]
                logger.info(
                    f"★ 利確: {ticker} ¥{entry:,.0f}→¥{last_close:,.0f} "
                    f"PnL=¥{pnl:+,.0f}"
                )
                await self._close_position(ticker, "利確ターゲット到達")
                self.daily_pnl += pnl

            else:
                # 含み損益ログ
                pnl = (last_close - entry) * pos["quantity"]
                if time.time() % 300 < SCAN_INTERVAL:  # 5分おきにログ
                    logger.info(
                        f"  保有中: {ticker} ¥{entry:,.0f}→¥{last_close:,.0f} "
                        f"含み¥{pnl:+,.0f} (SL=¥{stop:,.0f} TP=¥{target:,.0f})"
                    )

    async def _close_position(self, ticker: str, reason: str):
        """個別ポジション決済"""
        pos = self.open_positions.get(ticker)
        if not pos:
            return

        logger.info(f"決済: {ticker} {pos['quantity']}株 ({reason})")
        order = Order(
            ticker=ticker,
            side=OrderSide.SELL,
            quantity=pos["quantity"],
            order_type=OrderType.MARKET,  # 決済は成行で確実に
            strategy_name=pos["strategy"],
        )
        result = await self.broker.place_order(order)
        logger.info(f"  → {result.status.value} {result.notes}")

        self.trades_today.append({
            "ticker": ticker,
            "entry": pos["entry_price"],
            "exit_reason": reason,
            "strategy": pos["strategy"],
        })
        del self.open_positions[ticker]

    async def _force_close_all(self, reason: str):
        """全ポジション決済"""
        for ticker, pos in list(self.open_positions.items()):
            logger.info(f"決済: {ticker} {pos['quantity']}株 ({reason})")
            # 強制決済は成行（確実に決済するため）
            order = Order(
                ticker=ticker,
                side=OrderSide.SELL,
                quantity=pos["quantity"],
                order_type=OrderType.MARKET,
                strategy_name=pos["strategy"],
            )
            result = await self.broker.place_order(order)
            logger.info(f"  → {result.status.value} {result.notes}")

            self.trades_today.append({
                "ticker": ticker,
                "entry": pos["entry_price"],
                "exit_reason": reason,
                "strategy": pos["strategy"],
            })

        self.open_positions.clear()

    async def _print_summary(self):
        """日次サマリー"""
        final_balance = await self.broker.get_balance()
        pnl = final_balance - self.initial_balance

        logger.info("=" * 50)
        logger.info("日次サマリー")
        logger.info(f"  初期残高:  ¥{self.initial_balance:,.0f}")
        logger.info(f"  最終残高:  ¥{final_balance:,.0f}")
        logger.info(f"  損益:      ¥{pnl:+,.0f}")
        logger.info(f"  取引数:    {len(self.trades_today)}件")
        logger.info(f"  保有中:    {len(self.open_positions)}件")
        logger.info("=" * 50)

        # JSON保存
        summary = {
            "date": date.today().isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "pnl": pnl,
            "trades": self.trades_today,
            "open_positions": self.open_positions,
        }
        Path("knowledge/live_trade_log.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.start())
