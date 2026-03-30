"""
Tachibana Securities e-Trade API broker integration.

立花証券 e支店 API を使った実取引ブローカー。
公式サンプル (https://github.com/e-shiten-jp) に準拠した実装。

API仕様:
- リクエスト: 仮想URLに ?{URLエンコード済みJSON} をGETで送信
- レスポンス: Shift_JIS エンコードの JSON
- 認証: ログインで仮想URL（1日有効）を取得
- sCLMID で機能を識別（p_no はリクエスト連番）

SAFETY: ALLOW_LIVE_TRADING=True のときのみ実注文を送信。
"""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from datetime import datetime
from typing import Optional

import aiohttp
from loguru import logger

from brokers.base import (
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from core.config import settings


# 立花証券 API ベースURL (v4r8が現行有効バージョン)
DEFAULT_AUTH_URL = "https://kabuka.e-shiten.jp/e_api_v4r8/auth/"
DEMO_AUTH_URL = "https://demo-kabuka.e-shiten.jp/e_api_v4r8/auth/"

# 売買区分
_SIDE_MAP = {
    OrderSide.BUY: "3",   # 買い
    OrderSide.SELL: "1",   # 売り
}

# sCLMID 一覧
CLMID_LOGIN = "CLMAuthLoginRequest"
CLMID_LOGOUT = "CLMAuthLogoutRequest"
CLMID_NEW_ORDER = "CLMKabuNewOrder"
CLMID_CANCEL_ORDER = "CLMKabuCancelOrder"
CLMID_ORDER_LIST = "CLMOrderList"
CLMID_ORDER_DETAIL = "CLMOrderListDetail"
CLMID_GENBUTSU_LIST = "CLMGenbutuKabuList"
CLMID_BUYING_POWER = "CLMZanKaiKanougaku"
CLMID_MARKET_PRICE = "CLMMfdsGetMarketPrice"


def _encode_json_for_url(data: dict) -> str:
    """JSONをURL用にエンコード。立花API形式。

    JSONをダンプ→特殊文字をパーセントエンコード。
    """
    json_str = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # 立花API独自のエンコード
    encoded = json_str.replace("%", "%25")
    encoded = encoded.replace("#", "%23")
    encoded = encoded.replace("+", "%2B")
    encoded = encoded.replace("/", "%2F")
    encoded = encoded.replace(":", "%3A")
    encoded = encoded.replace("=", "%3D")
    return encoded


def _make_sd_date() -> str:
    """p_sd_date 形式の現在時刻文字列を生成"""
    now = datetime.now()
    return now.strftime("%Y.%m.%d-%H:%M:%S.") + f"{now.microsecond // 1000:03d}"


class TachibanaBroker(BaseBroker):
    """立花証券 e-Trade API ブローカー。

    Usage:
        async with TachibanaBroker() as broker:
            balance = await broker.get_balance()
            order = Order(ticker="7203", side=OrderSide.BUY, quantity=1)
            filled = await broker.place_order(order)
    """

    def __init__(
        self,
        user: str = "",
        password: str = "",
        second_password: str = "",
        auth_url: str = "",
        demo: bool = False,
        account_type: str = "specific",  # "specific"=特定, "general"=一般, "nisa"=NISA
    ) -> None:
        self._user = user or settings.TACHIBANA_USER
        self._password = password or settings.TACHIBANA_PASSWORD
        self._second_password = second_password or self._password  # 取引暗証番号
        self._auth_url = auth_url or settings.TACHIBANA_API_URL or ""
        if not self._auth_url:
            self._auth_url = DEMO_AUTH_URL if demo else DEFAULT_AUTH_URL
        self._demo = demo

        # 口座区分
        account_map = {"specific": "1", "general": "3", "nisa": "5"}
        self._tax_category = account_map.get(account_type, "1")

        # Session state
        self._session: Optional[aiohttp.ClientSession] = None
        self._p_no: int = 0  # リクエスト連番
        self._logged_in: bool = False

        # ログイン応答の仮想URL
        self._url_request: str = ""
        self._url_master: str = ""
        self._url_price: str = ""
        self._url_event: str = ""
        self._url_event_ws: str = ""

        self._lock = asyncio.Lock()
        self._orders: list[Order] = []

        if not self._user or not self._password:
            logger.warning(
                "TachibanaBroker: 認証情報未設定。"
                ".envにKABUAI_TACHIBANA_USER/PASSWORDを設定してください"
            )

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    def _next_p_no(self) -> str:
        self._p_no += 1
        return str(self._p_no)

    # ------------------------------------------------------------------
    # Login / Logout
    # ------------------------------------------------------------------

    async def login(self) -> bool:
        """ログイン。仮想URL（1日有効）を取得。"""
        if not self._user or not self._password:
            logger.error("認証情報が未設定です")
            return False

        session = await self._ensure_session()
        self._p_no = 0

        req_data = {
            "p_no": self._next_p_no(),
            "p_sd_date": _make_sd_date(),
            "sCLMID": CLMID_LOGIN,
            "sUserId": self._user,
            "sPassword": self._password,
            "sJsonOfmt": "5",
        }

        url = self._auth_url + "?" + _encode_json_for_url(req_data)

        try:
            logger.info("立花証券ログイン中... (user={})", self._user[:3] + "***")

            async with session.get(url, ssl=True) as resp:
                raw = await resp.read()
                text = raw.decode("shift_jis", errors="ignore")
                data = json.loads(text)

                p_errno = str(data.get("p_errno", "-1"))
                result_code = str(data.get("sResultCode", "-1"))

                if p_errno != "0" or result_code != "0":
                    error_msg = data.get("sResultText", data.get("p_err_text", "unknown"))
                    logger.error("ログイン失敗: errno={} code={} msg={}", p_errno, result_code, error_msg)
                    return False

                self._url_request = data.get("sUrlRequest", "")
                self._url_master = data.get("sUrlMaster", "")
                self._url_price = data.get("sUrlPrice", "")
                self._url_event = data.get("sUrlEvent", "")
                self._url_event_ws = data.get("sUrlEventWebSocket", "")
                self._logged_in = True

                logger.info(
                    "立花証券ログイン成功 (request_url={}...)",
                    self._url_request[:40] if self._url_request else "N/A",
                )

                # マスタ初期化（時価取得に必要）
                await self._init_event_download()

                return True

        except json.JSONDecodeError as e:
            logger.error("ログイン応答JSON解析失敗: {}", e)
            return False
        except Exception as e:
            logger.error("ログインエラー: {}", e)
            return False

    async def logout(self) -> None:
        """ログアウト"""
        if self._logged_in and self._url_request:
            try:
                await self._api_request(CLMID_LOGOUT, {})
            except Exception as e:
                logger.warning("ログアウトエラー: {}", e)

        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

        self._logged_in = False
        self._url_request = ""

    # ------------------------------------------------------------------
    # Generic API Request
    # ------------------------------------------------------------------

    async def _api_request(self, clmid: str, params: dict, url_type: str = "request") -> dict:
        """汎用APIリクエスト送信。

        Args:
            clmid: sCLMID (例: "CLMKabuNewOrder")
            params: 追加パラメータ
            url_type: "request", "master", "price", "event"

        Returns:
            パース済みJSON辞書
        """
        if not self._logged_in:
            raise RuntimeError("未ログイン。先にlogin()を呼んでください")

        session = await self._ensure_session()

        base_url = {
            "request": self._url_request,
            "master": self._url_master,
            "price": self._url_price,
            "event": self._url_event,
        }.get(url_type, self._url_request)

        if not base_url:
            raise RuntimeError(f"仮想URL未取得: {url_type}")

        req_data = {
            "p_no": self._next_p_no(),
            "p_sd_date": _make_sd_date(),
            "sCLMID": clmid,
            "sJsonOfmt": "5",
            **params,
        }

        url = base_url + "?" + _encode_json_for_url(req_data)

        try:
            async with session.get(url, ssl=True) as resp:
                raw = await resp.read()
                text = raw.decode("shift_jis", errors="ignore")
                data = json.loads(text)

                p_errno = str(data.get("p_errno", "0"))
                result_code = str(data.get("sResultCode", "0"))

                if p_errno != "0" or (result_code != "0" and result_code != ""):
                    error_msg = data.get("sResultText", data.get("p_err_text", ""))
                    if error_msg:
                        logger.warning(
                            "API {}: errno={} code={} msg={}",
                            clmid, p_errno, result_code, error_msg,
                        )

                return data

        except json.JSONDecodeError as e:
            logger.error("API {} JSON解析失敗: {}", clmid, e)
            return {}
        except Exception as e:
            logger.error("API {} 通信エラー: {}", clmid, e)
            return {}

    # ------------------------------------------------------------------
    # BaseBroker Implementation
    # ------------------------------------------------------------------

    async def place_order(self, order: Order) -> Order:
        """注文送信（現物、1株単位対応）"""
        async with self._lock:
            if not self._logged_in:
                if not await self.login():
                    order.status = OrderStatus.REJECTED
                    order.notes = "ログイン失敗"
                    return order

            # 銘柄コード（4桁で送信。5桁なら末尾0を除去）
            issue_code = order.ticker
            if len(issue_code) == 5 and issue_code.endswith("0"):
                issue_code = issue_code[:4]

            # 注文値段: 成行=0, 指値=価格
            if order.order_type == OrderType.MARKET:
                order_price = "0"
            elif order.limit_price:
                order_price = str(int(order.limit_price))
            else:
                order_price = "0"

            params = {
                "sIssueCode": issue_code,
                "sSizyouC": "00",  # 東証
                "sBaibaiKubun": _SIDE_MAP[order.side],
                "sCondition": "0",  # 指定なし（成行/指値はsOrderPriceで制御）
                "sOrderPrice": order_price,
                "sOrderSuryou": str(order.quantity),
                "sGenkinShinyouKubun": "0",  # 現物
                "sZyoutoekiKazeiC": self._tax_category,
                "sOrderExpireDay": "0",  # 当日限り
                "sGyakusasiOrderType": "0",
                "sGyakusasiZyouken": "0",
                "sGyakusasiPrice": "*",
                "sTatebiType": "*",
                "sTategyokuZyoutoekiKazeiC": "*",
                "sSecondPassword": self._second_password,
            }

            logger.info(
                "立花注文: {} {} x{} @ {} ({})",
                order.side.value, issue_code, order.quantity,
                order_price, order.strategy_name or "manual",
            )

            data = await self._api_request(CLMID_NEW_ORDER, params)

            result_code = str(data.get("sResultCode", "-1"))
            if result_code == "0":
                tachibana_id = data.get("sOrderNumber", "")
                order.status = OrderStatus.SUBMITTED
                order.notes = f"tachibana_id={tachibana_id}"
                order.updated_at = datetime.now()
                self._orders.append(order)
                logger.info(
                    "注文受付: {} order_no={} 手数料=¥{}",
                    issue_code, tachibana_id,
                    data.get("sOrderTesuryou", "0"),
                )
            else:
                order.status = OrderStatus.REJECTED
                error_text = data.get("sResultText", "unknown")
                warning_text = data.get("sWarningText", "")
                order.notes = f"エラー: {error_text} {warning_text}".strip()
                logger.error("注文拒否: {} - {}", issue_code, order.notes)

            return order

    async def cancel_order(self, order_id: str) -> bool:
        """注文取消"""
        async with self._lock:
            if not self._logged_in:
                return False

            # 内部IDからTachibana注文番号を取得
            tachibana_id = ""
            for o in self._orders:
                if o.order_id == order_id and "tachibana_id=" in (o.notes or ""):
                    tachibana_id = o.notes.split("tachibana_id=")[1].split()[0]
                    break

            if not tachibana_id:
                logger.warning("注文番号が見つかりません: {}", order_id)
                return False

            params = {
                "sOrderNumber": tachibana_id,
                "sEigyouDay": "",  # 空欄=当日
                "sSecondPassword": self._second_password,
            }

            data = await self._api_request(CLMID_CANCEL_ORDER, params)

            if str(data.get("sResultCode", "-1")) == "0":
                for o in self._orders:
                    if o.order_id == order_id:
                        o.status = OrderStatus.CANCELLED
                        o.updated_at = datetime.now()
                logger.info("注文取消成功: {}", tachibana_id)
                return True
            else:
                logger.error("注文取消失敗: {}", data.get("sResultText", "unknown"))
                return False

    async def get_positions(self) -> list[Position]:
        """現物保有銘柄一覧"""
        async with self._lock:
            if not self._logged_in:
                if not await self.login():
                    return []

            data = await self._api_request(CLMID_GENBUTSU_LIST, {})

            positions = []
            # 応答のリスト部分を探す
            for key, val in data.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "sIssueCode" in item:
                            ticker = item.get("sIssueCode", "")
                            qty = int(item.get("sKabusuuTanka", {}).get("sKabusuu", "0") or "0")
                            avg = float(item.get("sKabusuuTanka", {}).get("sTanka", "0") or "0")
                            if not qty:
                                qty = int(item.get("sOrderSuryou", "0") or "0")
                            if qty > 0:
                                positions.append(Position(
                                    ticker=ticker, quantity=qty,
                                    average_price=avg,
                                ))

            # フラットな応答の場合
            if not positions and "sIssueCode" in data:
                positions.append(Position(
                    ticker=data["sIssueCode"],
                    quantity=int(data.get("sBalanceSuryou", "0") or "0"),
                    average_price=float(data.get("sAveragePrice", "0") or "0"),
                ))

            return positions

    async def get_orders(self) -> list[Order]:
        """注文一覧"""
        async with self._lock:
            if not self._logged_in:
                if not await self.login():
                    return []

            data = await self._api_request(CLMID_ORDER_LIST, {})

            orders = []
            for key, val in data.items():
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict) and "sOrderNumber" in item:
                            side_code = item.get("sBaibaiKubun", "3")
                            orders.append(Order(
                                ticker=item.get("sIssueCode", ""),
                                side=OrderSide.BUY if side_code == "3" else OrderSide.SELL,
                                quantity=int(item.get("sOrderSuryou", "0") or "0"),
                                order_type=OrderType.MARKET if item.get("sOrderPrice") in ("0", "*") else OrderType.LIMIT,
                                limit_price=float(item.get("sOrderPrice", "0") or "0") if item.get("sOrderPrice") not in ("0", "*", "") else None,
                                filled_quantity=int(item.get("sYakuzyouSuryou", "0") or "0"),
                                filled_price=float(item.get("sYakuzyouPrice", "0") or "0"),
                                notes=f"tachibana_id={item.get('sOrderNumber', '')}",
                            ))

            return orders

    async def get_balance(self) -> float:
        """買付余力（現物可能額）"""
        async with self._lock:
            if not self._logged_in:
                if not await self.login():
                    return 0.0

            data = await self._api_request(CLMID_BUYING_POWER, {})

            # 買付余力フィールド (CLMZanKaiKanougaku)
            genkabu = data.get("sSummaryGenkabuKaituke", "0")
            power = float(genkabu) if genkabu else 0.0
            if power > 0:
                logger.info("買付余力(現物): ¥{:,.0f}", power)
                return power

            # フォールバック: サマリーAPIから取得
            summary = await self._api_request("CLMZanKaiSummary", {})
            for field in ("sGenbutuKabuKaituke", "sSyukkin"):
                val = summary.get(field, "0")
                if val and val != "0":
                    power = float(val)
                    if power > 0:
                        logger.info("買付余力: ¥{:,.0f} ({})", power, field)
                        return power

            logger.info("買付余力: ¥0 (未入金)")
            return power
            return 0.0

    # ------------------------------------------------------------------
    # EVENT I/F 初期化
    # ------------------------------------------------------------------

    async def _init_event_download(self) -> None:
        """マスタダウンロード（ログイン後に必須）。時価フィード開始。"""
        if not self._url_event:
            return

        try:
            session = await self._ensure_session()
            req_data = {
                "p_no": self._next_p_no(),
                "p_sd_date": _make_sd_date(),
                "sCLMID": "CLMEventDownload",
                "sJsonOfmt": "5",
            }
            url = self._url_event + "?" + _encode_json_for_url(req_data)
            async with session.get(url, ssl=True) as resp:
                raw = await resp.read()
                if len(raw) > 0:
                    logger.info("EVENT初期化完了 ({}bytes)", len(raw))
                else:
                    logger.warning("EVENT初期化: 空レスポンス")
        except Exception as e:
            logger.warning("EVENT初期化エラー: {}", e)

    # ------------------------------------------------------------------
    # 時価取得
    # ------------------------------------------------------------------

    async def get_current_price(self, ticker: str) -> float:
        """銘柄の現在値を取得"""
        if not self._logged_in:
            return 0.0

        issue_code = ticker if len(ticker) >= 5 else ticker + "0"

        data = await self._api_request(
            CLMID_MARKET_PRICE,
            {
                "sTargetIssueCode": issue_code,
                "sTargetSizyouC": "00",
                "sTargetColumn": "sGenzaiKabuka,sZenzituOwarine,sBaibaiTakane,sBaibaiYasune",
            },
            url_type="price",
        )

        items = data.get("aCLMMfdsMarketPrice", [])
        if items and isinstance(items, list):
            item = items[0]
            # 現在値 → 前日終値 の順で探す
            for field in ("sGenzaiKabuka", "sZenzituOwarine"):
                val = item.get(field, "")
                if val and val not in ("0", "*", ""):
                    return float(val)

        return 0.0

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    async def __aenter__(self):
        await self.login()
        return self

    async def __aexit__(self, *args):
        await self.logout()
