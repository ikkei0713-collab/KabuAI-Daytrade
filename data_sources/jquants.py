"""
J-Quants API V2 client for Japanese stock market data.

Uses API-key authentication (x-api-key header).
Base URL: https://api.jquants.com/v2

API Documentation: https://jpx.gitbook.io/j-quants-pro
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp
from loguru import logger

BASE_URL = "https://api.jquants.com/v2"

_DEFAULT_MAX_CONCURRENT = 5
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_DELAY = 1.0
_CACHE_DIR = Path("data/cache")


class JQuantsClient:
    """
    Async client for the J-Quants V2 API (API-key auth).

    Usage:
        async with JQuantsClient() as client:
            prices = await client.get_prices_daily("72030", "2024-01-01", "2024-01-31")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        cache_dir: Optional[Path] = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._api_key = (
            api_key
            or os.getenv("KABUAI_JQUANTS_API_KEY", "")
            or os.getenv("JQUANTS_API_KEY", "")
        )
        if not self._api_key:
            raise RuntimeError(
                "J-Quants API key not configured. "
                "Set KABUAI_JQUANTS_API_KEY in .env"
            )
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_ttl = cache_ttl_seconds
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> JQuantsClient:
        self._session = aiohttp.ClientSession(
            headers={
                "Accept": "application/json",
                "x-api-key": self._api_key,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        if not self._session:
            self._session = aiohttp.ClientSession(
                headers={
                    "Accept": "application/json",
                    "x-api-key": self._api_key,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            )

    def _cache_key(self, endpoint: str, params: dict[str, Any]) -> str:
        raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _read_cache(self, key: str) -> Optional[Any]:
        cache_file = self._cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if time.time() - data.get("_cached_at", 0) > self._cache_ttl:
                cache_file.unlink(missing_ok=True)
                return None
            return data.get("payload")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Cache read error for {}: {}", key, e)
            return None

    def _write_cache(self, key: str, payload: Any) -> None:
        cache_file = self._cache_dir / f"{key}.json"
        try:
            cache_file.write_text(
                json.dumps({"_cached_at": time.time(), "payload": payload}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Cache write error for {}: {}", key, e)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        use_cache: bool = True,
        retries: int = _DEFAULT_RETRY_COUNT,
    ) -> Any:
        """Make an authenticated, rate-limited request with retries and caching."""
        params = params or {}
        self._ensure_session()

        if use_cache and method.upper() == "GET":
            cache_key = self._cache_key(endpoint, params)
            cached = self._read_cache(cache_key)
            if cached is not None:
                logger.debug("Cache hit for {} {}", endpoint, params)
                return cached

        url = f"{BASE_URL}{endpoint}"

        for attempt in range(1, retries + 1):
            async with self._semaphore:
                try:
                    async with self._session.request(
                        method, url, params=params,
                    ) as resp:
                        if resp.status == 429:
                            retry_after = float(
                                resp.headers.get("Retry-After", _DEFAULT_RETRY_DELAY * attempt)
                            )
                            logger.warning(
                                "Rate limited on {} (attempt {}/{}), waiting {:.1f}s",
                                endpoint, attempt, retries, retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        if resp.status in (500, 502, 503, 504):
                            logger.warning(
                                "Server error {} on {} (attempt {}/{})",
                                resp.status, endpoint, attempt, retries,
                            )
                            if attempt < retries:
                                await asyncio.sleep(_DEFAULT_RETRY_DELAY * attempt)
                                continue

                        resp.raise_for_status()
                        data = await resp.json()

                        if use_cache and method.upper() == "GET":
                            self._write_cache(cache_key, data)

                        return data

                except aiohttp.ClientResponseError as e:
                    logger.error(
                        "J-Quants API error on {} (attempt {}/{}): {} {}",
                        endpoint, attempt, retries, e.status, e.message,
                    )
                    if attempt == retries:
                        raise
                    await asyncio.sleep(_DEFAULT_RETRY_DELAY * attempt)

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(
                        "J-Quants connection error on {} (attempt {}/{}): {}",
                        endpoint, attempt, retries, e,
                    )
                    if attempt == retries:
                        raise
                    await asyncio.sleep(_DEFAULT_RETRY_DELAY * attempt)

        raise RuntimeError(f"J-Quants request failed after {retries} retries: {endpoint}")

    # ------------------------------------------------------------------
    # Public API methods (V2 endpoints)
    # ------------------------------------------------------------------

    async def get_listed_info(
        self,
        ticker: Optional[str] = None,
        report_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """上場銘柄一覧を取得する。 V2: /equities/master"""
        params: dict[str, Any] = {}
        if ticker:
            params["code"] = ticker
        if report_date:
            params["date"] = report_date

        logger.info("Fetching listed info (ticker={}, date={})", ticker, report_date)
        data = await self._request("GET", "/equities/master", params=params)
        results = data.get("data", data.get("master", data.get("info", [])))
        logger.info("Retrieved {} listed stock entries", len(results))
        return results

    async def get_prices_daily(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> list[dict[str, Any]]:
        """日足の株価データを取得する。 V2: /equities/bars/daily"""
        params = {
            "code": ticker,
            "from": from_date,
            "to": to_date,
        }
        logger.info("Fetching daily prices: {} from {} to {}", ticker, from_date, to_date)
        data = await self._request("GET", "/equities/bars/daily", params=params)
        results = data.get("data", data.get("bars_daily", data.get("daily_quotes", [])))
        logger.info("Retrieved {} daily price records for {}", len(results), ticker)
        return results

    async def get_prices_daily_bulk(
        self,
        date_str: str,
    ) -> list[dict[str, Any]]:
        """指定日の全銘柄日足を一括取得する。V2: /equities/bars/daily?date=YYYY-MM-DD

        1回のAPIコールでその日の全銘柄データを取得できるため、
        個別銘柄ごとのリクエストより大幅に効率的。
        """
        params = {"date": date_str}
        logger.info("Fetching bulk daily prices for date={}", date_str)
        data = await self._request("GET", "/equities/bars/daily", params=params)
        results = data.get("data", data.get("bars_daily", data.get("daily_quotes", [])))
        logger.info("Retrieved {} records for bulk daily (date={})", len(results), date_str)
        return results

    async def get_prices_daily_range_bulk(
        self,
        from_date: str,
        to_date: str,
    ) -> list[dict[str, Any]]:
        """期間指定で全銘柄日足を一括取得する。日ごとにリクエストしてマージ。"""
        from datetime import datetime, timedelta
        start = datetime.strptime(from_date, "%Y-%m-%d").date()
        end = datetime.strptime(to_date, "%Y-%m-%d").date()
        all_records: list[dict[str, Any]] = []
        d = start
        while d <= end:
            records = await self.get_prices_daily_bulk(d.isoformat())
            all_records.extend(records)
            d += timedelta(days=1)
        logger.info("Bulk range fetch: {} total records ({} to {})", len(all_records), from_date, to_date)
        return all_records

    async def get_prices_intraday(
        self,
        ticker: str,
        date_str: str,
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """分足の株価データを取得する。 V2: /equities/bars/daily/am, /pm

        Args:
            use_cache: Falseにするとキャッシュを無視して最新データを取得
        """
        params = {
            "code": ticker,
            "date": date_str,
        }
        logger.info("Fetching intraday prices: {} on {}", ticker, date_str)

        am_quotes: list = []
        try:
            data = await self._request("GET", "/equities/bars/daily/am", params=params, use_cache=use_cache)
            am_quotes = data.get("data", data.get("bars_daily_am", data.get("prices_am", [])))
        except aiohttp.ClientResponseError as e:
            if e.status in (400, 403, 404):
                logger.warning("AM intraday data not available for {} on {}", ticker, date_str)
            else:
                raise

        pm_quotes: list = []
        try:
            data = await self._request("GET", "/equities/bars/daily/pm", params=params, use_cache=use_cache)
            pm_quotes = data.get("data", data.get("bars_daily_pm", data.get("prices_pm", [])))
        except aiohttp.ClientResponseError as e:
            if e.status in (400, 403, 404):
                logger.warning("PM intraday data not available for {} on {}", ticker, date_str)
            else:
                raise

        results = am_quotes + pm_quotes
        logger.info("Retrieved {} intraday records for {} on {}", len(results), ticker, date_str)
        return results

    async def get_financial_statements(
        self,
        ticker: str,
        report_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """財務情報を取得する。 V2: /fins/summary"""
        params: dict[str, Any] = {"code": ticker}
        if report_date:
            params["date"] = report_date

        logger.info("Fetching financial statements for {}", ticker)
        data = await self._request("GET", "/fins/summary", params=params)
        results = data.get("data", data.get("summary", data.get("statements", [])))
        logger.info("Retrieved {} financial records for {}", len(results), ticker)
        return results

    async def get_trading_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """営業日カレンダーを取得する。 V2: /markets/trading_calendar"""
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        logger.info("Fetching trading calendar (from={}, to={})", from_date, to_date)
        try:
            data = await self._request("GET", "/markets/trading_calendar", params=params)
            results = data.get("data", data.get("trading_calendar", []))
        except aiohttp.ClientResponseError as e:
            if e.status in (403, 404):
                logger.warning("Trading calendar endpoint error (status={}). Using empty list.", e.status)
                results = []
            else:
                raise
        logger.info("Retrieved {} calendar entries", len(results))
        return results
