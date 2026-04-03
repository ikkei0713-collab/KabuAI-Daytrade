"""
Telegram通知+対話モジュール

シグナル発生・エントリー・決済時にTelegramへ通知を送信する。
フリーテキストでの対話にも対応（ポジション確認・損益照会等）。

BotFather で作成した Bot Token と Chat ID を .env に設定:
  KABUAI_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
  KABUAI_TELEGRAM_CHAT_ID=123456789
"""

import os
import asyncio
import time
import aiohttp
from loguru import logger
from core.ticker_map import format_ticker, get_name


class TelegramNotifier:
    """Telegram Bot API を使った非同期通知+対話"""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or os.getenv("KABUAI_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("KABUAI_TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)
        self._last_update_id = 0
        self._trader = None  # LiveTraderへの参照（後からセット）
        self._polling = False

        if not self._token or not self._chat_id:
            logger.warning("Telegram設定なし。通知無効。")
        else:
            logger.info("Telegram通知+対話: 有効")

    def set_trader(self, trader) -> None:
        """LiveTraderへの参照をセット"""
        self._trader = trader

    async def send(self, message: str) -> bool:
        """メッセージを送信。失敗しても例外を投げない。"""
        if not self._enabled:
            return False
        try:
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self._chat_id,
                    "text": message,
                }
                if '<a href="' in message or "<b>" in message or "<code>" in message:
                    payload["parse_mode"] = "HTML"
                await session.post(url, json=payload)
            return True
        except Exception as e:
            logger.warning(f"Telegram送信失敗: {e}")
            return False

    async def notify_entry(
        self,
        ticker: str,
        direction: str,
        price: float,
        quantity: int,
        strategy: str,
        stop_loss: float,
        take_profit: float,
        reason: str,
    ) -> None:
        """エントリー通知"""
        name = get_name(ticker + "0") or get_name(ticker) or ""
        label = f"{name}({ticker})" if name else ticker
        msg = (
            f"🟢 ENTRY\n"
            f"{'='*30}\n"
            f"銘柄: {label}\n"
            f"方向: {direction}\n"
            f"価格: {price:,.0f}円\n"
            f"数量: {quantity}\n"
            f"金額: {price * quantity:,.0f}円\n"
            f"SL:   {stop_loss:,.0f}円\n"
            f"TP:   {take_profit:,.0f}円\n"
            f"戦略: {strategy}\n"
            f"理由: {reason}\n"
            f"{'='*30}"
        )
        await self.send(msg)

    async def notify_exit(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: int,
        pnl: float,
        pnl_pct: float,
        reason: str,
        daily_pnl: float,
        win_rate: float,
        trade_count: int,
    ) -> None:
        """決済通知"""
        icon = "🟢" if pnl > 0 else "🔴"
        result = "WIN" if pnl > 0 else "LOSE"
        name = get_name(ticker + "0") or get_name(ticker) or ""
        label = f"{name}({ticker})" if name else ticker
        msg = (
            f"{icon} EXIT [{result}]\n"
            f"{'='*30}\n"
            f"銘柄: {label}\n"
            f"方向: {direction}\n"
            f"参入: {entry_price:,.0f}円\n"
            f"決済: {exit_price:,.0f}円\n"
            f"数量: {quantity}\n"
            f"損益: {pnl:+,.0f}円 ({pnl_pct:+.1f}%)\n"
            f"理由: {reason}\n"
            f"{'─'*30}\n"
            f"累計: {daily_pnl:+,.0f}円\n"
            f"勝率: {win_rate:.0%} ({trade_count}件)\n"
            f"{'='*30}"
        )
        await self.send(msg)

    async def notify_daily_summary(
        self,
        trade_count: int,
        daily_pnl: float,
        win_count: int,
        positions: int,
    ) -> None:
        """日次サマリー通知"""
        win_rate = win_count / trade_count if trade_count > 0 else 0
        msg = (
            f"📊 日次サマリー\n"
            f"{'='*30}\n"
            f"トレード: {trade_count}件\n"
            f"勝率: {win_rate:.0%} ({win_count}勝)\n"
            f"損益: {daily_pnl:+,.0f}円\n"
            f"保有中: {positions}件\n"
            f"{'='*30}"
        )
        await self.send(msg)

    # ------------------------------------------------------------------
    # フリーテキスト対話
    # ------------------------------------------------------------------

    async def start_polling(self):
        """バックグラウンドでメッセージをポーリングして対話する"""
        if not self._enabled:
            return
        self._polling = True
        logger.info("Telegram対話ポーリング開始")
        while self._polling:
            try:
                await self._poll_once()
            except Exception as e:
                logger.debug(f"Telegram polling error: {e}")
            await asyncio.sleep(3)

    def stop_polling(self):
        self._polling = False

    async def _poll_once(self):
        """新着メッセージを取得して応答"""
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        params = {"offset": self._last_update_id + 1, "timeout": 1}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                data = await resp.json()

        if not data.get("ok"):
            return

        for update in data.get("result", []):
            self._last_update_id = update["update_id"]
            msg = update.get("message", {})
            text = msg.get("text", "").strip()
            chat_id = str(msg.get("chat", {}).get("id", ""))

            if chat_id != self._chat_id or not text:
                continue

            response = await self._handle_message(text)
            await self.send(response)

    async def _handle_message(self, text: str) -> str:
        """フリーテキストを解釈して応答を生成"""
        t = text.lower()
        trader = self._trader

        # ポジション・保有
        if any(w in t for w in ["ポジション", "保有", "持ってる", "status", "どう"]):
            if not trader or not trader.open_positions:
                return "保有ポジションなし"
            lines = ["📋 保有ポジション"]
            for ticker, pos in trader.open_positions.items():
                name = get_name(ticker + "0") or get_name(ticker) or ticker
                yahoo_price = 0.0
                if trader._yahoo_client:
                    yahoo_price = await trader._yahoo_client.get_current_price(ticker)
                pnl = (yahoo_price - pos["entry_price"]) * pos["quantity"] if yahoo_price > 0 else 0
                lines.append(
                    f"  {name}({ticker})\n"
                    f"    ¥{pos['entry_price']:,.0f}→¥{yahoo_price:,.1f} "
                    f"含み¥{pnl:+,.0f}\n"
                    f"    SL=¥{pos['stop']:,.0f} TP=¥{pos['target']:,.0f}"
                )
            return "\n".join(lines)

        # 損益・PnL
        if any(w in t for w in ["損益", "pnl", "利益", "儲", "負け"]):
            if not trader:
                return "トレーダー未接続"
            trades = len(trader.trades_today)
            wins = sum(1 for t in trader.trades_today if t.get("pnl", 0) > 0)
            return (
                f"📊 本日の損益\n"
                f"確定: ¥{trader.daily_pnl:+,.0f}\n"
                f"取引: {trades}件 (勝{wins}件)\n"
                f"保有: {len(trader.open_positions)}件"
            )

        # 残高・余力
        if any(w in t for w in ["残高", "余力", "資金", "balance"]):
            if not trader:
                return "トレーダー未接続"
            used = sum(p["entry_price"] * p["quantity"] for p in trader.open_positions.values())
            available = trader.initial_balance - used
            return (
                f"💰 資金状況\n"
                f"総資金: ¥{trader.initial_balance:,.0f}\n"
                f"使用中: ¥{used:,.0f}\n"
                f"余力: ¥{available:,.0f}"
            )

        # 候補・銘柄
        if any(w in t for w in ["候補", "銘柄", "スキャン", "candidate"]):
            if not trader or not trader._scan_candidates:
                return "スキャン候補なし"
            codes = trader._scan_candidates[:10]
            lines = [f"📋 スキャン候補 (上位{len(codes)}件)"]
            for code in codes:
                name = get_name(code) or get_name(code[:4] + "0") or ""
                df = trader.stock_data.get(code)
                price = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0
                display = f"{name}({code[:4]})" if name else code[:4]
                lines.append(f"  {display}: ¥{price:,.0f}")
            return "\n".join(lines)

        # TDnet・ニュース
        if any(w in t for w in ["tdnet", "ニュース", "開示", "イベント"]):
            if not trader or not trader.tdnet_events:
                return "TDnetイベントなし"
            lines = [f"📰 TDnet ({len(trader.tdnet_events)}件)"]
            for ticker, ev in list(trader.tdnet_events.items())[:5]:
                lines.append(f"  {ticker} [{ev.disclosure_type.value}] {ev.title[:30]}")
            return "\n".join(lines)

        # ヘルプ
        if any(w in t for w in ["help", "ヘルプ", "使い方", "何ができる"]):
            return (
                "🤖 KabuAI Bot\n"
                "話しかけてください:\n"
                "・ポジション / どう → 保有状況\n"
                "・損益 / PnL → 本日の成績\n"
                "・残高 / 余力 → 資金状況\n"
                "・候補 / 銘柄 → スキャン候補\n"
                "・TDnet / ニュース → 適時開示\n"
                "・何でも自由に聞いてください"
            )

        # フリーテキスト → AI応答
        if not trader:
            return "🤖 ボット起動中です。しばらくお待ちください。"

        return await self._ai_response(text, trader)

    async def _build_context(self, trader) -> str:
        """トレーダーの現在状況をテキスト化"""
        lines = []
        lines.append(f"買付余力: ¥{trader.initial_balance:,.0f}")
        used = sum(p["entry_price"] * p["quantity"] for p in trader.open_positions.values())
        lines.append(f"使用中: ¥{used:,.0f}")
        lines.append(f"残余力: ¥{trader.initial_balance - used:,.0f}")
        lines.append(f"確定損益: ¥{trader.daily_pnl:+,.0f}")
        lines.append(f"本日取引: {len(trader.trades_today)}件")
        lines.append(f"監視銘柄: {len(trader._scan_candidates)}件")
        lines.append(f"Active戦略: 14個")

        if trader.open_positions:
            lines.append("\n保有ポジション:")
            for ticker, pos in trader.open_positions.items():
                name = get_name(ticker + "0") or get_name(ticker) or ticker
                yahoo_price = 0.0
                if trader._yahoo_client:
                    yahoo_price = await trader._yahoo_client.get_current_price(ticker)
                pnl = (yahoo_price - pos["entry_price"]) * pos["quantity"] if yahoo_price > 0 else 0
                lines.append(
                    f"  {name}({ticker}): ¥{pos['entry_price']:,.0f}→¥{yahoo_price:,.1f} 含み¥{pnl:+,.0f}"
                )
        else:
            lines.append("\n保有ポジション: なし")

        if trader._scan_candidates:
            lines.append("\nスキャン候補(上位5):")
            for code in trader._scan_candidates[:5]:
                name = get_name(code) or get_name(code[:4] + "0") or ""
                df = trader.stock_data.get(code)
                price = float(df["close"].iloc[-1]) if df is not None and not df.empty else 0
                label = f"{name}({code[:4]})" if name else code[:4]
                lines.append(f"  {label}: ¥{price:,.0f}")

        if trader.tdnet_events:
            lines.append(f"\nTDnetイベント: {len(trader.tdnet_events)}件")
            for ticker, ev in list(trader.tdnet_events.items())[:3]:
                lines.append(f"  {ticker} [{ev.disclosure_type.value}] {ev.title[:30]}")

        return "\n".join(lines)

    async def _ai_response(self, user_text: str, trader) -> str:
        """OpenAI GPTでフリーテキスト応答"""
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            # AIなしフォールバック
            pos_count = len(trader.open_positions)
            return (
                f"🤖 KabuAI 稼働中\n"
                f"保有: {pos_count}件 / 監視: {len(trader._scan_candidates)}件\n"
                f"確定損益: ¥{trader.daily_pnl:+,.0f}\n"
                f"\n「ポジション」「損益」「残高」「候補」で詳細確認"
            )

        context = await self._build_context(trader)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "あなたはKabuAI自動トレードボットのアシスタントです。"
                                    "日本株のデイトレードを自動で行うシステムのTelegram窓口として、"
                                    "ユーザーの質問に簡潔に日本語で答えてください。"
                                    "以下がボットの現在の状況です:\n\n"
                                    f"{context}"
                                ),
                            },
                            {"role": "user", "content": user_text},
                        ],
                        "max_tokens": 300,
                        "temperature": 0.7,
                    },
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
                    reply = data["choices"][0]["message"]["content"].strip()
                    return reply
        except Exception as e:
            logger.warning(f"AI応答エラー: {e}")
            return f"🤖 KabuAI 稼働中（AI応答エラー）\n確定損益: ¥{trader.daily_pnl:+,.0f}"
