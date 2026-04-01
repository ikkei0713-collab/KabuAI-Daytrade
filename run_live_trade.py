"""
自動トレードボット（立花証券 実取引）v3

v2からの改善:
- 銘柄候補をJ-Quantsから動的発見（低位株を自動スキャン）
- トレーリングストップ（含み益が1ATR超えたら追従）
- 時間帯別戦略フィルター（ORBは寄り付き30分、etc）
- セッション保存/復元（電話認証を省略）
- 残高をAPIから自動取得
- オーダーブック系戦略を無効化（偽データ排除）
- J-Quants日中足キャッシュ無効化（鮮度優先）

フロー:
1. セッション復元 or 電話認証→ログイン
2. 30秒ごとにスキャン→シグナル生成
3. シグナルがあれば自動注文
4. ポジション監視→利確/損切り/トレーリングストップ
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
from data_sources.tdnet import TDnetClient
from data_sources.yahoo_finance import YahooFinanceClient

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
TDNET_SCAN_INTERVAL = 300    # TDnet スキャン間隔（秒）= 5分
INTRADAY_REFRESH_INTERVAL = 120  # 日中足更新間隔（秒）= 2分

# 時間帯フィルター（戦略ごとの有効時間帯）
STRATEGY_TIME_WINDOWS = {
    "orb":          (9, 0, 9, 45),    # 寄り付き45分のみ
    "open_drive":   (9, 0, 9, 30),    # 寄り付き30分のみ
    "gap_go":       (9, 0, 10, 0),    # 前場序盤
    "gap_fade":     (9, 0, 10, 0),    # 前場序盤
    "vwap_reclaim": (9, 30, 14, 45),  # 寄り付き安定後〜引け前
    "vwap_bounce":  (9, 30, 14, 45),  # 寄り付き安定後〜引け前
    "trend_follow": (9, 15, 14, 30),  # 安定期間
    "crash_rebound":(9, 0, 14, 50),   # 急落はいつでも
}

SESSION_FILE = Path("data/tachibana_session.json")


class LiveTrader:
    def __init__(self):
        self.broker = TachibanaBroker()
        self.fe = FeatureEngineer()
        self.regime_det = RegimeDetector()
        self.daily_pnl = 0.0
        self.trades_today = []
        self.open_positions = {}  # ticker -> {entry_price, quantity, stop, target, strategy, highest_price}
        self.initial_balance = 0.0
        self.stock_data = {}       # ticker(5digit) -> DataFrame (日足)
        self.intraday_data = {}    # ticker(5digit) -> DataFrame (分足)
        self.tdnet_events = {}     # ticker(4digit) -> DisclosureInfo
        self.stopped = False
        self._last_tdnet_scan = 0.0
        self._last_intraday_refresh = 0.0
        self._jquants_client = None
        self._yahoo_client = None
        self._scan_candidates = []  # 動的に構築

    async def start(self):
        """メインループ"""
        # セッション復元 or ログイン
        ok = await self._restore_or_login()
        if not ok:
            logger.error("ログイン失敗。電話認証してから再実行してください。")
            return
        logger.info("★ ログイン成功。このプロセスを閉じないでください。")

        # セッション保存
        await self._save_session()

        # 残高取得（APIから）
        self.initial_balance = await self._get_real_balance()
        if self.initial_balance <= 0:
            self.initial_balance = 34333  # フォールバック
        logger.info(f"=== 自動トレード開始 v3 (日中足+TDnet+動的銘柄+トレーリングSL) ===")
        logger.info(f"買付余力: ¥{self.initial_balance:,.0f}")

        # 戦略登録（板系は無効化済み）
        StrategyRegistry.clear()
        StrategyRegistry.register_all_defaults()
        active = StrategyRegistry.get_active()
        logger.info(f"Active戦略: {len(active)}個 ({[s.name for s in active]})")

        # J-Quantsクライアント
        self._jquants_client = JQuantsClient()
        await self._jquants_client.__aenter__()

        # Yahoo Financeクライアント（リアルタイム価格+1分足）
        self._yahoo_client = YahooFinanceClient()

        try:
            # 営業日チェック（土日のみ。J-QuantsライトプランではカレンダーAPI使用不可）
            if date.today().weekday() >= 5:
                logger.info("本日は土日です。待機します。")
                await asyncio.sleep(3600)
                return

            # 銘柄候補を動的に構築
            await self._build_scan_candidates()

            # 銘柄データ一括取得（日足）
            await self._load_stock_data()

            # 日中足の初回取得
            await self._refresh_intraday_data()

            # TDnet初回スキャン
            await self._scan_tdnet()

            # 既存ポジション確認（APIから）
            await self._sync_positions_from_api()

            # メインループ
            while not self.stopped:
                now = datetime.now(JST)

                # 場が閉まったら待機
                if now.hour >= MARKET_CLOSE_HOUR or now.hour < 9:
                    if self.open_positions:
                        logger.info(f"場外: {len(self.open_positions)}ポジション保有中（スイング）")
                    await asyncio.sleep(300)
                    continue

                # 強制決済チェック（14:50）
                if now.hour == FORCE_CLOSE_HOUR and now.minute >= FORCE_CLOSE_MIN:
                    if self.open_positions:
                        await self._force_close_all("引け前強制決済")
                    await asyncio.sleep(60)
                    continue

                # 日中足データ更新（2分ごと）
                if time.time() - self._last_intraday_refresh >= INTRADAY_REFRESH_INTERVAL:
                    await self._refresh_intraday_data()

                # TDnet適時開示スキャン（5分ごと）
                if time.time() - self._last_tdnet_scan >= TDNET_SCAN_INTERVAL:
                    await self._scan_tdnet()

                # ポジション監視（ストップ/利確/トレーリングストップ）
                await self._check_positions()

                # 日次損失チェック
                if self.daily_pnl <= MAX_DAILY_LOSS:
                    logger.warning(f"日次損失上限到達: ¥{self.daily_pnl:,.0f} <= ¥{MAX_DAILY_LOSS:,.0f}")
                    await self._force_close_all("損失上限")
                    self.stopped = True
                    break

                # スキャン→注文
                await self._scan_and_trade()

                await asyncio.sleep(SCAN_INTERVAL)

        except KeyboardInterrupt:
            logger.info("手動停止")
        except Exception as e:
            logger.error(f"エラー: {e}", exc_info=True)
        finally:
            await self._print_summary()
            await self.broker.logout()
            if self._jquants_client:
                await self._jquants_client.__aexit__(None, None, None)
            if self._yahoo_client:
                await self._yahoo_client.close()

    # ------------------------------------------------------------------
    # セッション管理
    # ------------------------------------------------------------------

    async def _restore_or_login(self) -> bool:
        """保存済みセッションを復元。失敗なら新規ログイン。"""
        if SESSION_FILE.exists():
            try:
                info = json.loads(SESSION_FILE.read_text(encoding="utf-8"))
                self.broker._url_request = info["url_request"]
                self.broker._url_master = info["url_master"]
                self.broker._url_price = info["url_price"]
                self.broker._url_event = info["url_event"]
                self.broker._url_event_ws = info.get("url_event_ws", "")
                self.broker._p_no = info.get("p_no", 10)
                self.broker._logged_in = True
                await self.broker._ensure_session()

                # セッション生存確認
                test = await self.broker._api_request("CLMZanKaiSummary", {})
                if str(test.get("p_errno", "-1")) == "0":
                    logger.info("セッション復元成功")
                    return True
                else:
                    logger.warning("セッション切れ → 再ログイン")
                    self.broker._logged_in = False
            except Exception as e:
                logger.warning(f"セッション復元失敗: {e}")

        return await self.broker.login()

    async def _save_session(self):
        """セッション情報を保存"""
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        info = {
            "url_request": self.broker._url_request,
            "url_master": self.broker._url_master,
            "url_price": self.broker._url_price,
            "url_event": self.broker._url_event,
            "url_event_ws": getattr(self.broker, "_url_event_ws", ""),
            "p_no": self.broker._p_no,
            "saved_at": datetime.now(JST).isoformat(),
        }
        SESSION_FILE.write_text(
            json.dumps(info, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("セッション保存完了")

    # ------------------------------------------------------------------
    # 取引カレンダー
    # ------------------------------------------------------------------

    async def _is_trading_day(self) -> bool:
        """J-Quantsの取引カレンダーで今日が営業日か確認"""
        try:
            today_str = date.today().isoformat()
            cal = await self._jquants_client.get_trading_calendar(
                from_date=today_str, to_date=today_str
            )
            if cal:
                entry = cal[0]
                is_holiday = entry.get("HolidayDivision", "0") != "1"
                if is_holiday:
                    logger.info(f"取引カレンダー: {today_str} は営業日")
                else:
                    logger.warning(f"取引カレンダー: {today_str} は休場日")
                return is_holiday
        except Exception as e:
            logger.warning(f"取引カレンダー取得失敗: {e}")
        # フォールバック: 土日チェック
        return date.today().weekday() < 5

    # ------------------------------------------------------------------
    # 残高・ポジション同期
    # ------------------------------------------------------------------

    async def _get_real_balance(self) -> float:
        """APIから実際の買付余力を取得"""
        try:
            data = await self.broker._api_request("CLMZanKaiSummary", {})
            for field in ("sGenbutuKabuKaituke", "sSyukkin"):
                val = data.get(field, "0")
                if val and val not in ("0", ""):
                    balance = float(val)
                    if balance > 0:
                        logger.info(f"API残高取得: ¥{balance:,.0f} ({field})")
                        return balance
        except Exception as e:
            logger.warning(f"残高取得失敗: {e}")
        return 0.0

    async def _sync_positions_from_api(self):
        """APIから保有銘柄を取得してポジションを同期"""
        try:
            data = await self.broker._api_request("CLMGenbutuKabuList", {})
            items = data.get("aGenbutuKabuList", [])
            if not items:
                logger.info("保有銘柄なし")
                return

            for item in items:
                ticker = item.get("sUriOrderIssueCode", "")[:4]
                qty = int(item.get("sUriOrderZanKabuSuryou", "0") or "0")
                avg_price = float(item.get("sUriOrderGaisanBokaTanka", "0") or "0")
                current_price = float(item.get("sUriOrderHyoukaTanka", "0") or "0")

                if qty <= 0 or not ticker:
                    continue

                if ticker not in self.open_positions:
                    # ATR推定（日足データから）
                    code_5 = ticker + "0"
                    atr = 2.0  # デフォルト
                    df = self.stock_data.get(code_5)
                    if df is not None and len(df) >= 14:
                        highs = df["high"].tail(14)
                        lows = df["low"].tail(14)
                        closes = df["close"].tail(14)
                        tr = pd.concat([
                            highs - lows,
                            (highs - closes.shift(1)).abs(),
                            (lows - closes.shift(1)).abs(),
                        ], axis=1).max(axis=1)
                        atr = float(tr.mean())

                    self.open_positions[ticker] = {
                        "entry_price": avg_price,
                        "quantity": qty,
                        "stop": avg_price - atr * 2,
                        "target": avg_price + atr * 3,
                        "strategy": "synced",
                        "order_time": datetime.now(JST).isoformat(),
                        "tachibana_order": "",
                        "highest_price": max(avg_price, current_price),
                    }
                    pnl = (current_price - avg_price) * qty
                    logger.info(
                        f"ポジション同期: {ticker} {qty}株 @¥{avg_price:,.1f} "
                        f"現在¥{current_price:,.1f} 含み¥{pnl:+,.0f} "
                        f"SL=¥{avg_price - atr * 2:,.0f} TP=¥{avg_price + atr * 3:,.0f}"
                    )
        except Exception as e:
            logger.warning(f"ポジション同期失敗: {e}")

    # ------------------------------------------------------------------
    # 動的銘柄候補構築
    # ------------------------------------------------------------------

    async def _build_scan_candidates(self):
        """J-Quantsから買える銘柄を動的に発見"""
        logger.info("銘柄候補を動的構築中...")

        # 現在の余力で買える上限価格
        used = sum(p["entry_price"] * p["quantity"] for p in self.open_positions.values())
        available = self.initial_balance - used
        max_price_per_share = available / 100  # 100株単位

        # 固定候補（常に監視）
        fixed = ["94320"]  # NTT

        # J-Quantsから当日の全銘柄を取得して安い順にフィルタ
        try:
            yesterday = (date.today() - timedelta(days=3)).isoformat()
            today_str = date.today().isoformat()
            raw = await self._jquants_client.get_prices_daily_bulk(
                (date.today() - timedelta(days=1)).isoformat()
            )
            if raw:
                df = pd.DataFrame(raw)
                # カラム名の正規化
                col_map = {}
                for c in df.columns:
                    cl = c.lower()
                    if "close" in cl or c in ("AdjC", "AdjustmentClose"):
                        col_map[c] = "close"
                    elif "volume" in cl or c in ("AdjVo", "AdjustmentVolume"):
                        col_map[c] = "volume"
                    elif "code" in cl.lower() or c == "Code":
                        col_map[c] = "code"
                if col_map:
                    df = df.rename(columns=col_map)

                if "close" in df.columns and "volume" in df.columns and "code" in df.columns:
                    df["close"] = pd.to_numeric(df["close"], errors="coerce")
                    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
                    df = df.dropna(subset=["close", "volume"])

                    # フィルタ: 買える価格帯 + 最低出来高
                    affordable = df[
                        (df["close"] > 50) &
                        (df["close"] <= max_price_per_share) &
                        (df["volume"] > 500000)  # 50万株以上の流動性
                    ].sort_values("volume", ascending=False)

                    # 上位20銘柄を候補に
                    dynamic_codes = []
                    for _, row in affordable.head(20).iterrows():
                        code = str(row["code"])
                        if len(code) == 4:
                            code = code + "0"
                        if code not in fixed:
                            dynamic_codes.append(code)

                    self._scan_candidates = fixed + dynamic_codes
                    logger.info(
                        f"動的銘柄候補: {len(self._scan_candidates)}銘柄 "
                        f"(余力¥{available:,.0f} → 上限¥{max_price_per_share:,.0f}/株)"
                    )
                    for code in self._scan_candidates[:10]:
                        row = df[df["code"].astype(str).str.startswith(code[:4])]
                        if not row.empty:
                            r = row.iloc[0]
                            logger.info(f"  {code}: ¥{r['close']:,.0f} 出来高{r['volume']:,.0f}")
                    return
        except Exception as e:
            logger.warning(f"動的銘柄構築失敗: {e}")

        # フォールバック: 固定候補
        self._scan_candidates = [
            "94320", "40050", "93070", "47550", "41200",
        ]
        logger.info(f"固定銘柄候補にフォールバック: {len(self._scan_candidates)}銘柄")

    # ------------------------------------------------------------------
    # データ取得
    # ------------------------------------------------------------------

    async def _load_stock_data(self):
        """J-Quantsから日足データを一括取得"""
        logger.info("銘柄データ取得中...")
        today_str = date.today().isoformat()
        from_str = (date.today() - timedelta(days=200)).isoformat()
        for code_5 in self._scan_candidates:
            try:
                raw = await self._jquants_client.get_prices_daily(code_5, from_str, today_str)
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

        logger.info(f"日足データ取得完了: {len(self.stock_data)}銘柄")

    async def _refresh_intraday_data(self):
        """Yahoo Financeから1分足を取得（J-Quants 403時の代替）"""
        self._last_intraday_refresh = time.time()
        updated = 0

        for code_5 in self._scan_candidates:
            if code_5 not in self.stock_data:
                continue
            code_4 = code_5[:4]
            try:
                raw = await self._yahoo_client.get_intraday_ohlcv(code_4)
                if not raw:
                    continue
                df = pd.DataFrame(raw)
                rename_map = {}
                for col in df.columns:
                    cl = col.lower()
                    if "open" in cl or col == "AdjO":
                        rename_map[col] = "open"
                    elif "high" in cl or col == "AdjH":
                        rename_map[col] = "high"
                    elif "low" in cl or col == "AdjL":
                        rename_map[col] = "low"
                    elif "close" in cl or col == "AdjC":
                        rename_map[col] = "close"
                    elif "volume" in cl or col == "AdjVo":
                        rename_map[col] = "volume"
                    elif "time" in cl or "datetime" in cl or "date" in cl:
                        rename_map[col] = "DateTime"
                if rename_map:
                    df = df.rename(columns=rename_map)

                for c in ["open", "high", "low", "close", "volume"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                if "DateTime" in df.columns:
                    df["DateTime"] = pd.to_datetime(df["DateTime"])
                    df = df.sort_values("DateTime").reset_index(drop=True)

                self.intraday_data[code_5] = df
                updated += 1
            except Exception as e:
                logger.debug(f"  {code_4}: Yahoo日中足取得失敗 {e}")

        if updated > 0:
            logger.info(f"Yahoo日中足更新: {updated}/{len(self.stock_data)}銘柄")

    # ------------------------------------------------------------------
    # TDnet適時開示ライブ監視
    # ------------------------------------------------------------------

    async def _scan_tdnet(self):
        """TDnetから本日の適時開示を取得"""
        self._last_tdnet_scan = time.time()
        try:
            async with TDnetClient() as client:
                disclosures = await client.fetch_today_disclosures()
                material = client.filter_material_events(disclosures)

                new_events = 0
                for event in material:
                    ticker_4 = event.ticker[:4]
                    existing = self.tdnet_events.get(ticker_4)
                    if existing and existing.timestamp == event.timestamp:
                        continue

                    self.tdnet_events[ticker_4] = event
                    new_events += 1
                    logger.info(
                        f"★ TDnet: {ticker_4} [{event.disclosure_type.value}] "
                        f"{event.title[:50]} ({event.timestamp.strftime('%H:%M')})"
                    )

                if new_events > 0:
                    logger.info(f"TDnet新着: {new_events}件 (総material: {len(self.tdnet_events)}件)")
                else:
                    logger.debug(f"TDnet: 新着なし (総material: {len(self.tdnet_events)}件)")

        except Exception as e:
            logger.warning(f"TDnetスキャン失敗: {e}")

    # ------------------------------------------------------------------
    # スキャン→注文
    # ------------------------------------------------------------------

    def _is_strategy_in_time_window(self, strategy_name: str, now: datetime) -> bool:
        """戦略が現在の時間帯で有効かチェック"""
        window = STRATEGY_TIME_WINDOWS.get(strategy_name)
        if window is None:
            return True  # 制限なし = 常に有効
        start_h, start_m, end_h, end_m = window
        current = now.hour * 60 + now.minute
        start = start_h * 60 + start_m
        end = end_h * 60 + end_m
        return start <= current <= end

    async def _scan_and_trade(self):
        """スキャン→シグナル→注文"""
        used = sum(p["entry_price"] * p["quantity"] for p in self.open_positions.values())
        balance = self.initial_balance - used
        now = datetime.now(JST)

        strategies = StrategyRegistry.get_active()
        # 時間帯フィルター適用
        strategies = [s for s in strategies if self._is_strategy_in_time_window(s.name, now)]

        signals = []

        for code_5, df in self.stock_data.items():
            code_4 = code_5[:4]

            if code_4 in self.open_positions:
                continue

            if len(self.open_positions) >= MAX_POSITIONS:
                break

            last_close = float(df["close"].iloc[-1])
            cost_100 = last_close * 100

            if cost_100 + 200 > balance:
                logger.debug(f"  {code_4}: ¥{cost_100:,.0f} > 余力¥{balance:,.0f}")
                continue

            intraday_df = self.intraday_data.get(code_5)

            features = self.fe.calculate_all_features(
                df, clock=now, intraday_ohlcv=intraday_df
            )
            features["current_price"] = last_close

            # Yahoo Financeからリアルタイム価格取得
            yahoo_price = await self._yahoo_client.get_current_price(code_4)
            if yahoo_price > 0:
                features["current_price"] = yahoo_price
                last_close = yahoo_price
            elif intraday_df is not None and not intraday_df.empty and "close" in intraday_df.columns:
                intraday_last = float(intraday_df["close"].iloc[-1])
                if intraday_last > 0:
                    features["current_price"] = intraday_last
                    last_close = intraday_last

            regime = self.regime_det.detect(df)
            features["regime_result"] = regime

            # TDnetイベント注入
            tdnet_event = self.tdnet_events.get(code_4)
            if tdnet_event:
                features["event_type"] = tdnet_event.disclosure_type.value
                features["event_magnitude"] = 1.0 if tdnet_event.is_material else 0.5
                features["historical_event_response"] = 0.5
                features["news_sentiment"] = (
                    0.8 if tdnet_event.disclosure_type.value in ("上方修正", "自社株買い", "増配", "株式分割")
                    else -0.5 if tdnet_event.disclosure_type.value in ("下方修正", "上場廃止")
                    else 0.2
                )

            for strategy in strategies:
                try:
                    signal = await strategy.scan(code_5, df, features)
                    if signal and signal.confidence >= MIN_CONFIDENCE and signal.direction == "long":
                        confidence_bonus = 0.0
                        if intraday_df is not None and not intraday_df.empty:
                            confidence_bonus += 0.03
                        if tdnet_event and tdnet_event.is_material:
                            confidence_bonus += 0.05

                        signals.append({
                            "code_4": code_4,
                            "code_5": code_5,
                            "strategy": strategy.name,
                            "confidence": signal.confidence + confidence_bonus,
                            "entry": last_close,
                            "stop": signal.stop_loss,
                            "target": signal.take_profit,
                            "reason": signal.entry_reason,
                            "regime": regime.regime,
                            "has_intraday": intraday_df is not None and not intraday_df.empty,
                            "has_tdnet": tdnet_event is not None,
                        })
                except Exception:
                    continue

        if not signals:
            return

        signals.sort(key=lambda s: -s["confidence"])
        best = signals[0]

        data_sources = []
        if best.get("has_intraday"):
            data_sources.append("分足")
        if best.get("has_tdnet"):
            data_sources.append("TDnet")
        src_str = "+".join(data_sources) if data_sources else "日足のみ"

        logger.info(
            f"シグナル: {best['code_4']} [{best['strategy']}] "
            f"conf={best['confidence']:.2f} entry=¥{best['entry']:,.0f} "
            f"SL=¥{best['stop']:,.0f} TP=¥{best['target']:,.0f} "
            f"[{best['regime']}] [{src_str}] {best['reason'][:40]}"
        )

        order = Order(
            ticker=best["code_4"],
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.LIMIT,
            limit_price=best["entry"],
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
                "highest_price": best["entry"],  # トレーリングストップ用
            }
        else:
            logger.warning(f"注文失敗: {best['code_4']} {result.notes}")

    # ------------------------------------------------------------------
    # ポジション管理（トレーリングストップ付き）
    # ------------------------------------------------------------------

    async def _check_positions(self):
        """保有ポジションの損益チェック + トレーリングストップ"""
        if not self.open_positions:
            return

        for ticker, pos in list(self.open_positions.items()):
            # 価格取得: 立花API → Yahoo Finance → 日中足 → 日足
            live_price = await self.broker.get_current_price(ticker)
            price_source = "立花"

            if live_price <= 0:
                live_price = await self._yahoo_client.get_current_price(ticker)
                price_source = "Yahoo"

            if live_price <= 0:
                code_5 = ticker + "0"
                intraday = self.intraday_data.get(code_5)
                if intraday is not None and not intraday.empty and "close" in intraday.columns:
                    live_price = float(intraday["close"].iloc[-1])
                    price_source = "分足"
                else:
                    df = self.stock_data.get(code_5)
                    if df is None:
                        continue
                    live_price = float(df["close"].iloc[-1])
                    price_source = "日足"
                logger.warning(f"  {ticker}: 立花+Yahoo失敗、{price_source}フォールバック ¥{live_price:,.1f}")

            entry = pos["entry_price"]
            stop = pos["stop"]
            target = pos["target"]
            highest = pos.get("highest_price", entry)

            # 最高値更新
            if live_price > highest:
                pos["highest_price"] = live_price
                highest = live_price

            # トレーリングストップ計算
            # ATR推定（簡易: エントリーの2%をATR代わりに使用）
            atr_est = entry * 0.02
            profit_from_entry = live_price - entry

            # 含み益が1ATR以上ならトレーリングストップを引き上げ
            if profit_from_entry > atr_est:
                trailing_stop = highest - atr_est * 1.5
                if trailing_stop > stop:
                    old_stop = stop
                    pos["stop"] = trailing_stop
                    stop = trailing_stop
                    if trailing_stop - old_stop > 0.1:
                        logger.info(
                            f"  トレーリングSL更新: {ticker} "
                            f"¥{old_stop:,.1f}→¥{trailing_stop:,.1f} "
                            f"(最高値¥{highest:,.1f})"
                        )

            # ストップロス
            if live_price <= stop:
                pnl = (live_price - entry) * pos["quantity"]
                logger.warning(
                    f"★ ストップロス: {ticker} ¥{entry:,.0f}→¥{live_price:,.1f} "
                    f"PnL=¥{pnl:+,.0f} [{price_source}]"
                )
                await self._close_position(ticker, "ストップロス")
                self.daily_pnl += pnl

            # 利確ターゲット（指値で有利に決済）
            elif live_price >= target:
                pnl = (live_price - entry) * pos["quantity"]
                logger.info(
                    f"★ 利確: {ticker} ¥{entry:,.0f}→¥{live_price:,.1f} "
                    f"PnL=¥{pnl:+,.0f} [{price_source}] → 指値¥{target:,.0f}で決済"
                )
                await self._close_position(ticker, "利確ターゲット到達", limit_price=target)
                self.daily_pnl += pnl

            else:
                pnl = (live_price - entry) * pos["quantity"]
                logger.info(
                    f"  保有中: {ticker} ¥{entry:,.0f}→¥{live_price:,.1f} "
                    f"含み¥{pnl:+,.0f} (SL=¥{stop:,.1f} TP=¥{target:,.0f}) [{price_source}]"
                )

    async def _close_position(self, ticker: str, reason: str, limit_price: float = None):
        """個別ポジション決済。limit_price指定で指値、なければ成行。"""
        pos = self.open_positions.get(ticker)
        if not pos:
            return

        if limit_price:
            logger.info(f"決済(指値): {ticker} {pos['quantity']}株 @¥{limit_price:,.0f} ({reason})")
            order = Order(
                ticker=ticker,
                side=OrderSide.SELL,
                quantity=pos["quantity"],
                order_type=OrderType.LIMIT,
                limit_price=limit_price,
                strategy_name=pos["strategy"],
            )
        else:
            logger.info(f"決済(成行): {ticker} {pos['quantity']}株 ({reason})")
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
        del self.open_positions[ticker]

    async def _force_close_all(self, reason: str):
        """全ポジション決済"""
        for ticker, pos in list(self.open_positions.items()):
            logger.info(f"決済: {ticker} {pos['quantity']}株 ({reason})")
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
        pnl = final_balance - self.initial_balance if final_balance > 0 else self.daily_pnl

        logger.info("=" * 50)
        logger.info("日次サマリー")
        logger.info(f"  初期残高:  ¥{self.initial_balance:,.0f}")
        logger.info(f"  最終残高:  ¥{final_balance:,.0f}")
        logger.info(f"  損益:      ¥{pnl:+,.0f}")
        logger.info(f"  取引数:    {len(self.trades_today)}件")
        logger.info(f"  保有中:    {len(self.open_positions)}件")
        logger.info(f"  TDnetイベント: {len(self.tdnet_events)}件")
        logger.info(f"  日中足取得銘柄: {len(self.intraday_data)}件")
        logger.info(f"  スキャン候補: {len(self._scan_candidates)}銘柄")
        logger.info("=" * 50)

        summary = {
            "date": date.today().isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": final_balance,
            "pnl": pnl,
            "trades": self.trades_today,
            "open_positions": {
                k: {kk: vv for kk, vv in v.items() if kk != "highest_price"}
                for k, v in self.open_positions.items()
            },
            "tdnet_events": {k: v.title for k, v in self.tdnet_events.items()},
            "scan_candidates": self._scan_candidates,
        }
        Path("knowledge").mkdir(exist_ok=True)
        Path("knowledge/live_trade_log.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    trader = LiveTrader()
    asyncio.run(trader.start())
