"""
Telegram通知モジュール

シグナル発生・エントリー・決済時にTelegramへ通知を送信する。
BotFather で作成した Bot Token と Chat ID を .env に設定:
  KABUAI_TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
  KABUAI_TELEGRAM_CHAT_ID=123456789
"""

import os
import asyncio
from loguru import logger

try:
    from telegram import Bot
    _HAS_TELEGRAM = True
except ImportError:
    _HAS_TELEGRAM = False


class TelegramNotifier:
    """Telegram Bot API を使った非同期通知"""

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self._token = bot_token or os.getenv("KABUAI_TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("KABUAI_TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id and _HAS_TELEGRAM)

        if not _HAS_TELEGRAM:
            logger.warning("python-telegram-bot 未インストール。通知無効。")
        elif not self._token or not self._chat_id:
            logger.warning("Telegram設定なし (KABUAI_TELEGRAM_BOT_TOKEN / KABUAI_TELEGRAM_CHAT_ID)。通知無効。")
        else:
            logger.info("Telegram通知: 有効")

    async def send(self, message: str) -> bool:
        """メッセージを送信。失敗しても例外を投げない。"""
        if not self._enabled:
            return False
        try:
            bot = Bot(token=self._token)
            await bot.send_message(
                chat_id=self._chat_id,
                text=message,
                parse_mode="Monospace",
            )
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
        msg = (
            f"🟢 ENTRY\n"
            f"{'='*30}\n"
            f"銘柄: {ticker}\n"
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
        msg = (
            f"{icon} EXIT [{result}]\n"
            f"{'='*30}\n"
            f"銘柄: {ticker}\n"
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
