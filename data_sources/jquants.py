"""
J-Quants API client for Japanese stock market data.

Provides access to:
- Listed stock information (上場銘柄一覧)
- Daily OHLCV prices (日足)
- Intraday minute-level prices (分足)
- Financial statements (財務情報)
- Trading calendar (営業日カレンダー)

API Documentation: https://jpx.gitbook.io/j-quants-api
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiohttp
from loguru import logger

BASE_URL = "https://api.jquants.com/v1"

# Rate limit: 12 requests per second as documented
_DEFAULT_MAX_CONCURRENT = 10
_DEFAULT_RETRY_COUNT = 3
_DEFAULT_RETRY_DELAY = 1.0  # seconds
_CACHE_DIR = Path("data/cache")


@dataclass
class TokenInfo:
    """Stores authentication token state."""

    id_token: str = ""
    refresh_token: str = ""
    id_token_expires_at: float = 0.0  # unix timestamp


class JQuantsClient:
    """
    Async client for the J-Quants API.

    Usage:
        async with JQuantsClient(refresh_token="...") as client:
            prices = await client.get_prices_daily("7203", "2024-01-01", "2024-01-31")
    """

    def __init__(
        self,
        refresh_token: Optional[str] = None,
        mail_address: Optional[str] = None,
        password: Optional[str] = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        cache_dir: Optional[Path] = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self._refresh_token = refresh_token or os.getenv("KABUAI_JQUANTS_REFRESH_TOKEN", "") or os.getenv("JQUANTS_REFRESH_TOKEN", "")
        self._mail_address = mail_address or os.getenv("KABUAI_JQUANTS_MAIL", "") or os.getenv("JQUANTS_MAIL_ADDRESS", "")
        self._password = password or os.getenv("KABUAI_JQUANTS_PASSWORD", "") or os.getenv("JQUANTS_PASSWORD", "")
        self._token = TokenInfo(refresh_token=self._refresh_token)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache_dir = cache_dir or _CACHE_DIR
        self._cache_ttl = cache_ttl_seconds

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> JQuantsClient:
        self._session = aiohttp.ClientSession(
            headers={"Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    async def authenticate(self) -> str:
        """
        Obtain or refresh the ID token.

        Flow:
        1. If a valid (non-expired) ID token exists, return it.
        2. If a refresh token is available, call /token/auth_refresh.
        3. Otherwise, call /token/auth_user with email/password to get a
           refresh token first, then obtain an ID token.

        Returns:
            The current ID token string.
        """
        # Still valid?
        if self._token.id_token and time.time() < self._token.id_token_expires_at - 60:
            return self._token.id_token

        if not self._session:
            self._session = aiohttp.ClientSession(
                headers={"Accept": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30),
            )

        # If we don't have a refresh token, get one via email/password
        if not self._token.refresh_token:
            if not self._mail_address or not self._password:
                raise RuntimeError(
                    "J-Quants: No refresh token and no email/password configured. "
                    "Set JQUANTS_REFRESH_TOKEN or JQUANTS_MAIL_ADDRESS + JQUANTS_PASSWORD."
                )
            logger.info("J-Quants: Authenticating with email/password")
            async with self._session.post(
                f"{BASE_URL}/token/auth_user",
                json={
                    "mailaddress": self._mail_address,
                    "password": self._password,
                },
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()
                self._token.refresh_token = body["refreshToken"]
                logger.info("J-Quants: Obtained refresh token")

        # Exchange refresh token for ID token
        logger.info("J-Quants: Refreshing ID token")
        async with self._session.post(
            f"{BASE_URL}/token/auth_refresh",
            params={"refreshtoken": self._token.refresh_token},
        ) as resp:
            resp.raise_for_status()
            body = await resp.json()
            self._token.id_token = body["idToken"]
            # ID tokens are typically valid for 24 hours
            self._token.id_token_expires_at = time.time() + 86400
            logger.info("J-Quants: ID token refreshed successfully")

        return self._token.id_token

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token.id_token}"}

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
        """
        Make an authenticated, rate-limited request with retries and caching.
        """
        params = params or {}

        # Check cache
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
                    # Ensure token is valid
                    await self.authenticate()

                    async with self._session.request(
                        method,
                        url,
                        params=params,
                        headers=self._auth_headers(),
                    ) as resp:
                        if resp.status == 429:
                            retry_after = float(resp.headers.get("Retry-After", _DEFAULT_RETRY_DELAY * attempt))
                            logger.warning(
                                "J-Quants rate limited on {} (attempt {}/{}), waiting {:.1f}s",
                                endpoint, attempt, retries, retry_after,
                            )
                            await asyncio.sleep(retry_after)
                            continue

                        resp.raise_for_status()
                        data = await resp.json()

                        # Cache successful GET responses
                        if use_cache and method.upper() == "GET":
                            self._write_cache(cache_key, data)

                        return data

                except aiohttp.ClientResponseError as e:
                    if e.status in (401, 403):
                        logger.warning("J-Quants auth error on {}, re-authenticating", endpoint)
                        self._token.id_token = ""
                        self._token.id_token_expires_at = 0.0
                        if attempt < retries:
                            await asyncio.sleep(_DEFAULT_RETRY_DELAY)
                            continue
                    logger.error(
                        "J-Quants API error on {} (attempt {}/{}): {}",
                        endpoint, attempt, retries, e,
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
    # Public API methods
    # ------------------------------------------------------------------

    async def get_listed_info(
        self,
        ticker: Optional[str] = None,
        report_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        上場銘柄一覧を取得する。

        Args:
            ticker: Filter by specific ticker code (e.g. "7203").
            report_date: Date string "YYYY-MM-DD". Defaults to today.

        Returns:
            List of dicts with keys like Code, CompanyName, Sector33Code, etc.
        """
        params: dict[str, Any] = {}
        if ticker:
            params["code"] = ticker
        if report_date:
            params["date"] = report_date

        logger.info("Fetching listed info (ticker={}, date={})", ticker, report_date)
        data = await self._request("GET", "/listed/info", params=params)
        results = data.get("info", [])
        logger.info("Retrieved {} listed stock entries", len(results))
        return results

    async def get_prices_daily(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> list[dict[str, Any]]:
        """
        日足の株価データを取得する。

        Args:
            ticker: Stock code (e.g. "7203").
            from_date: Start date "YYYY-MM-DD".
            to_date: End date "YYYY-MM-DD".

        Returns:
            List of dicts with keys: Date, Code, Open, High, Low, Close,
            Volume, TurnoverValue, AdjustmentOpen, AdjustmentHigh,
            AdjustmentLow, AdjustmentClose, AdjustmentVolume.
        """
        params = {
            "code": ticker,
            "from": from_date,
            "to": to_date,
        }
        logger.info("Fetching daily prices: {} from {} to {}", ticker, from_date, to_date)
        data = await self._request("GET", "/prices/daily_quotes", params=params)
        results = data.get("daily_quotes", [])
        logger.info("Retrieved {} daily price records for {}", len(results), ticker)
        return results

    async def get_prices_intraday(
        self,
        ticker: str,
        date_str: str,
    ) -> list[dict[str, Any]]:
        """
        分足（1分足）の株価データを取得する。

        Note: J-Quants API availability of intraday data depends on
        the subscription plan. This method uses the morning/afternoon
        session endpoints.

        Args:
            ticker: Stock code (e.g. "7203").
            date_str: Date "YYYY-MM-DD".

        Returns:
            List of dicts with minute-level OHLCV data.
        """
        params = {
            "code": ticker,
            "date": date_str,
        }
        logger.info("Fetching intraday prices: {} on {}", ticker, date_str)

        # Intraday data may not be available on all plans; attempt the request
        # and return an empty list on 400/404 errors
        try:
            data = await self._request(
                "GET",
                "/prices/prices_am",
                params=params,
                use_cache=True,
            )
            am_quotes = data.get("prices_am", [])
        except aiohttp.ClientResponseError as e:
            if e.status in (400, 404):
                logger.warning("AM intraday data not available for {} on {}", ticker, date_str)
                am_quotes = []
            else:
                raise

        try:
            data = await self._request(
                "GET",
                "/prices/prices_pm",
                params=params,
                use_cache=True,
            )
            pm_quotes = data.get("prices_pm", [])
        except aiohttp.ClientResponseError as e:
            if e.status in (400, 404):
                logger.warning("PM intraday data not available for {} on {}", ticker, date_str)
                pm_quotes = []
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
        """
        財務情報を取得する。

        Args:
            ticker: Stock code (e.g. "7203").
            report_date: Optional date filter "YYYY-MM-DD".

        Returns:
            List of financial statement dicts.
        """
        params: dict[str, Any] = {"code": ticker}
        if report_date:
            params["date"] = report_date

        logger.info("Fetching financial statements for {}", ticker)
        data = await self._request("GET", "/fins/statements", params=params)
        results = data.get("statements", [])
        logger.info("Retrieved {} financial statement records for {}", len(results), ticker)
        return results

    async def get_trading_calendar(
        self,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        営業日カレンダーを取得する。

        Args:
            from_date: Start date "YYYY-MM-DD".
            to_date: End date "YYYY-MM-DD".

        Returns:
            List of dicts with Date and HolidayDivision keys.
            HolidayDivision: "1" = business day, "0" = holiday.
        """
        params: dict[str, Any] = {}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        logger.info("Fetching trading calendar (from={}, to={})", from_date, to_date)
        data = await self._request("GET", "/markets/trading_calendar", params=params)
        results = data.get("trading_calendar", [])
        logger.info("Retrieved {} calendar entries", len(results))
        return results
