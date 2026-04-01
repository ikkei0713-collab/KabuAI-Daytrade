"""
Yahoo Finance Japan real-time price client.

Fetches current stock prices via Yahoo Finance Japan's chart JSON API,
with an in-memory cache (30s TTL) to avoid excessive requests.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import aiohttp
from loguru import logger

# JSON chart API is far more reliable than HTML scraping.
_CHART_URL = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T"
    "?interval=1m&range=1d"
)

_CACHE_TTL = 30  # seconds


class YahooFinanceClient:
    """Async client for Yahoo Finance Japan stock prices."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[float, float]] = {}  # ticker -> (price, ts)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
                timeout=aiohttp.ClientTimeout(total=10),
            )
        return self._session

    async def get_current_price(self, ticker: str) -> float:
        """Return the latest price for a Japanese stock ticker (4-digit code).

        Uses Yahoo Finance Japan's chart JSON API.  Returns 0.0 on any failure.
        """
        # --- cache check ---
        now = time.monotonic()
        cached = self._cache.get(ticker)
        if cached is not None:
            price, ts = cached
            if now - ts < _CACHE_TTL:
                return price

        # --- fetch ---
        url = _CHART_URL.format(ticker=ticker)
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        "Yahoo Finance returned HTTP {} for {}", resp.status, ticker
                    )
                    return 0.0
                data = await resp.json()

            result = data["chart"]["result"][0]
            meta = result["meta"]
            price = float(meta["regularMarketPrice"])

            self._cache[ticker] = (price, now)
            logger.debug("Yahoo価格取得 {} → ¥{:.1f}", ticker, price)
            return price

        except Exception as exc:
            logger.warning("Yahoo価格取得失敗 {}: {}", ticker, exc)
            return 0.0

    async def get_intraday_ohlcv(self, ticker: str) -> list[dict]:
        """当日の1分足OHLCVを取得する。

        Returns:
            list of dicts with keys: DateTime, open, high, low, close, volume
        """
        url = _CHART_URL.format(ticker=ticker)
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            indicators = result.get("indicators", {})
            quotes = indicators.get("quote", [{}])[0]

            if not timestamps:
                return []

            from datetime import datetime
            from zoneinfo import ZoneInfo
            jst = ZoneInfo("Asia/Tokyo")

            bars = []
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            closes = quotes.get("close", [])
            volumes = quotes.get("volume", [])

            for i, ts in enumerate(timestamps):
                if i >= len(closes) or closes[i] is None:
                    continue
                bars.append({
                    "DateTime": datetime.fromtimestamp(ts, tz=jst).isoformat(),
                    "open": opens[i] if i < len(opens) and opens[i] else 0,
                    "high": highs[i] if i < len(highs) and highs[i] else 0,
                    "low": lows[i] if i < len(lows) and lows[i] else 0,
                    "close": closes[i] if i < len(closes) and closes[i] else 0,
                    "volume": volumes[i] if i < len(volumes) and volumes[i] else 0,
                })

            logger.debug("Yahoo日中足取得 {}: {}本", ticker, len(bars))
            return bars

        except Exception as exc:
            logger.warning("Yahoo日中足取得失敗 {}: {}", ticker, exc)
            return []

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
