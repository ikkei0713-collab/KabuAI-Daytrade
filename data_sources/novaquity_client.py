"""
Novaquity API client for event intelligence and corporate graph data.

Provides event data, company features, and propagation candidates
from the Novaquity platform.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

import aiohttp
from loguru import logger

_DEFAULT_TIMEOUT = 10
_DEFAULT_RETRIES = 2
_DEFAULT_BACKOFF = 1.0
_CACHE_TTL_SECONDS = 300  # 5 minutes


class NovaquityClient:
    """
    Async client for the Novaquity API.

    Usage:
        async with NovaquityClient() as client:
            events = await client.get_events("72030")
            features = await client.get_company_features("72030")
            propagation = await client.get_propagation("72030")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("NOVAQUITY_API_KEY", "")
        self._base_url = (
            base_url
            or os.getenv("NOVAQUITY_BASE_URL", "https://api.novaquity.com/v1")
        ).rstrip("/")
        self._enabled = (
            enabled
            if enabled is not None
            else os.getenv("NOVAQUITY_ENABLED", "true").lower() in ("true", "1", "yes")
        )
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, tuple[float, Any]] = {}  # key -> (expires_at, data)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> NovaquityClient:
        if self._enabled:
            self._session = aiohttp.ClientSession(
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                timeout=aiohttp.ClientTimeout(total=_DEFAULT_TIMEOUT),
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def get_events(self, ticker: str) -> list[dict]:
        """指定銘柄のイベント情報を取得する。"""
        data = await self._request("GET", f"/events/{ticker}")
        results = data.get("events", data.get("data", []))
        logger.info("Retrieved {} events for {}", len(results), ticker)
        return results

    async def get_company_features(self, ticker: str) -> dict:
        """指定銘柄の企業特徴量を取得する。"""
        data = await self._request("GET", f"/companies/{ticker}/features")
        logger.info("Retrieved company features for {}", ticker)
        return data

    async def get_propagation(self, ticker: str) -> list[dict]:
        """指定銘柄の企業グラフ伝播候補を取得する。"""
        data = await self._request("GET", f"/propagation/{ticker}")
        results = data.get("candidates", data.get("data", []))
        logger.info("Retrieved {} propagation candidates for {}", len(results), ticker)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Return cached value if present and not expired."""
        entry = self._cache.get(cache_key)
        if entry is None:
            return None
        expires_at, data = entry
        if time.time() > expires_at:
            del self._cache[cache_key]
            return None
        return data

    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Store value in cache with TTL."""
        self._cache[cache_key] = (time.time() + _CACHE_TTL_SECONDS, data)

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        retries: int = _DEFAULT_RETRIES,
    ) -> Any:
        """Make an authenticated request with retry, backoff, and caching."""
        if not self._enabled:
            logger.debug("Novaquity disabled, returning empty result for {}", endpoint)
            return {}

        if not self._session:
            logger.warning("Novaquity session not initialized")
            return {}

        # Check in-memory cache
        cache_key = f"{method}:{endpoint}:{params}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            logger.debug("Cache hit for {}", endpoint)
            return cached

        url = f"{self._base_url}{endpoint}"

        for attempt in range(1, retries + 1):
            try:
                async with self._session.request(method, url, params=params) as resp:
                    if resp.status in (500, 502, 503, 504, 429):
                        delay = _DEFAULT_BACKOFF * attempt
                        logger.warning(
                            "Novaquity {} on {} (attempt {}/{}), retrying in {:.1f}s",
                            resp.status, endpoint, attempt, retries, delay,
                        )
                        if attempt < retries:
                            await asyncio.sleep(delay)
                            continue
                        # Last attempt, return empty
                        logger.error(
                            "Novaquity {} on {} after {} retries",
                            resp.status, endpoint, retries,
                        )
                        return {}

                    if resp.status >= 400:
                        logger.error(
                            "Novaquity API error {} on {}", resp.status, endpoint,
                        )
                        return {}

                    data = await resp.json()
                    self._set_cache(cache_key, data)
                    return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                delay = _DEFAULT_BACKOFF * attempt
                logger.warning(
                    "Novaquity connection error on {} (attempt {}/{}): {}",
                    endpoint, attempt, retries, e,
                )
                if attempt < retries:
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Novaquity request failed after {} retries: {} - {}",
                    retries, endpoint, e,
                )
                return {}

        return {}
