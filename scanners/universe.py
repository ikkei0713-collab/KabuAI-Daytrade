"""
Universe scanner for selecting tradeable Japanese stocks.

Filters the TSE listed universe down to a tradeable watchlist
based on liquidity, price range, market cap, and pre-market activity.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any, Optional

from loguru import logger

from core.config import settings
from data_sources.jquants import JQuantsClient


# ---------------------------------------------------------------------------
# Constants for Japanese market filtering
# ---------------------------------------------------------------------------

MIN_AVG_DAILY_VOLUME = 500_000          # 50万株/日 以上
MIN_PRICE_YEN = 200                     # 最低株価
MAX_PRICE_YEN = 50_000                  # 最高株価
MIN_LISTING_DAYS = 30                   # 新規上場30日以内は除外
LOT_SIZE = 100                          # 単元株数

# Market cap tiers (yen)
MARKET_CAP_MIN = 10_000_000_000         # 100億円以上
MARKET_CAP_MAX = 10_000_000_000_000     # 10兆円以下（流動性確保 & ボラ確保）

# Pre-market / opportunity ranking
GAP_THRESHOLD_PCT = 1.5                 # 1.5%以上のギャップ
VOLUME_SPIKE_RATIO = 1.5               # 平均比1.5倍以上の出来高


class UniverseScanner:
    """
    Scans the TSE universe and filters down to tradeable day-trade candidates.

    Usage::

        async with JQuantsClient(refresh_token="...") as client:
            scanner = UniverseScanner(client)
            tickers = await scanner.scan_universe()
    """

    def __init__(
        self,
        jquants_client: JQuantsClient,
        lookback_days: int = 20,
    ) -> None:
        self._client = jquants_client
        self._lookback_days = lookback_days
        self._listed_cache: list[dict[str, Any]] = []
        self._price_cache: dict[str, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scan_universe(self, ref_date: Optional[date] = None) -> list[str]:
        """
        Return tickers that pass all base filters.

        Filters applied in order:
        1. Listed on TSE Prime / Standard / Growth
        2. Average daily volume > MIN_AVG_DAILY_VOLUME
        3. Price within [MIN_PRICE_YEN, MAX_PRICE_YEN]
        4. Market cap within [MARKET_CAP_MIN, MARKET_CAP_MAX]
        5. Listed for at least MIN_LISTING_DAYS
        """
        ref = ref_date or date.today()
        logger.info("UniverseScanner: scanning universe for {}", ref)

        # Step 1: fetch all listed stocks
        listed = await self._fetch_listed_info(ref)
        logger.info("UniverseScanner: {} total listed stocks", len(listed))

        # Step 2: basic static filters
        candidates = self._apply_static_filters(listed, ref)
        logger.info(
            "UniverseScanner: {} candidates after static filters", len(candidates),
        )

        # Step 3: fetch recent prices and apply dynamic filters
        tickers = [c["Code"] for c in candidates]
        passed = await self._apply_dynamic_filters(tickers, ref)
        logger.info(
            "UniverseScanner: {} tickers passed all filters", len(passed),
        )
        return passed

    async def get_today_watchlist(
        self,
        base_tickers: Optional[list[str]] = None,
        ref_date: Optional[date] = None,
    ) -> list[str]:
        """
        From the base universe (or provided tickers), filter further based
        on pre-market activity such as gap-ups/downs and volume spikes.

        Returns a ranked subset suitable for today's trading.
        """
        ref = ref_date or date.today()
        if base_tickers is None:
            base_tickers = await self.scan_universe(ref)

        logger.info(
            "UniverseScanner: building watchlist from {} base tickers", len(base_tickers),
        )

        watchlist: list[tuple[str, float]] = []

        for ticker in base_tickers:
            prices = await self._get_recent_prices(ticker, ref)
            if len(prices) < 2:
                continue

            latest = prices[-1]
            prev = prices[-2]
            prev_close = float(prev.get("AdjustmentClose", prev.get("Close", 0)))
            today_open = float(latest.get("AdjustmentOpen", latest.get("Open", 0)))

            if prev_close == 0:
                continue

            # Gap percentage
            gap_pct = ((today_open - prev_close) / prev_close) * 100.0

            # Volume vs average
            volumes = [float(p.get("AdjustmentVolume", p.get("Volume", 0))) for p in prices[:-1]]
            avg_vol = sum(volumes) / len(volumes) if volumes else 1.0
            today_vol = float(latest.get("AdjustmentVolume", latest.get("Volume", 0)))
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0.0

            # Score: absolute gap + volume spike
            score = abs(gap_pct) * 0.6 + min(vol_ratio, 5.0) * 0.4

            if abs(gap_pct) >= GAP_THRESHOLD_PCT or vol_ratio >= VOLUME_SPIKE_RATIO:
                watchlist.append((ticker, score))

        # Sort descending by score
        watchlist.sort(key=lambda x: x[1], reverse=True)
        result = [t for t, _ in watchlist[:30]]  # Top 30
        logger.info("UniverseScanner: watchlist has {} tickers", len(result))
        return result

    def rank_by_opportunity(
        self,
        tickers: list[str],
        features: dict[str, dict[str, float]],
    ) -> list[str]:
        """
        Rank tickers by trading opportunity potential using pre-computed features.

        Features dict: {ticker: {feature_name: value, ...}}

        Scoring factors:
        - volatility (ATR / price)
        - volume ratio vs average
        - absolute gap percentage
        - RSI extremes (< 30 or > 70)
        """
        scored: list[tuple[str, float]] = []

        for ticker in tickers:
            feat = features.get(ticker, {})
            score = 0.0

            # Volatility component
            atr_pct = feat.get("atr_pct", 0.0)
            score += min(atr_pct * 10.0, 3.0)  # cap at 3

            # Volume spike
            vol_ratio = feat.get("volume_ratio", 1.0)
            score += min(vol_ratio - 1.0, 3.0) if vol_ratio > 1.0 else 0.0

            # Gap
            gap_pct = abs(feat.get("gap_pct", 0.0))
            score += min(gap_pct, 3.0)

            # RSI extremes
            rsi = feat.get("rsi", 50.0)
            if rsi < 30:
                score += (30 - rsi) * 0.1
            elif rsi > 70:
                score += (rsi - 70) * 0.1

            scored.append((ticker, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _fetch_listed_info(self, ref: date) -> list[dict[str, Any]]:
        """Fetch and cache listed stock info."""
        if not self._listed_cache:
            self._listed_cache = await self._client.get_listed_info(
                report_date=ref.isoformat(),
            )
        return self._listed_cache

    def _apply_static_filters(
        self, listed: list[dict[str, Any]], ref: date,
    ) -> list[dict[str, Any]]:
        """Apply filters that don't require price data."""
        results: list[dict[str, Any]] = []
        cutoff_date = ref - timedelta(days=MIN_LISTING_DAYS)

        for stock in listed:
            # Must be on TSE (MarketCode starts with appropriate prefix)
            market_code = stock.get("MarketCode", "")
            if market_code not in ("0111", "0112", "0113", "0114"):
                # 0111=Prime, 0112=Standard, 0113=Growth, 0114=JASDAQ (legacy)
                continue

            # Check listing date
            listing_date_str = stock.get("Date", "")
            if listing_date_str:
                try:
                    listing_date = datetime.strptime(listing_date_str, "%Y-%m-%d").date()
                    if listing_date > cutoff_date:
                        continue
                except ValueError:
                    pass

            # Must be a normal stock (not ETF, REIT, etc.)
            security_code = stock.get("Code", "")
            if not security_code or len(security_code) < 4:
                continue

            results.append(stock)

        return results

    async def _apply_dynamic_filters(
        self, tickers: list[str], ref: date,
    ) -> list[str]:
        """Apply filters that require price data (volume, price range, market cap)."""
        passed: list[str] = []

        # Process in batches to respect rate limits
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            tasks = [self._check_dynamic_filters(t, ref) for t in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for ticker, result in zip(batch, results):
                if isinstance(result, Exception):
                    logger.warning(
                        "UniverseScanner: error checking {}: {}", ticker, result,
                    )
                    continue
                if result:
                    passed.append(ticker)

        return passed

    async def _check_dynamic_filters(self, ticker: str, ref: date) -> bool:
        """Check dynamic filters for a single ticker."""
        prices = await self._get_recent_prices(ticker, ref)

        if len(prices) < 5:
            return False

        # Average daily volume
        volumes = [
            float(p.get("AdjustmentVolume", p.get("Volume", 0)))
            for p in prices
        ]
        avg_volume = sum(volumes) / len(volumes)
        if avg_volume < MIN_AVG_DAILY_VOLUME:
            return False

        # Latest close price
        latest = prices[-1]
        close_price = float(latest.get("AdjustmentClose", latest.get("Close", 0)))
        if close_price < MIN_PRICE_YEN or close_price > MAX_PRICE_YEN:
            return False

        # Rough market cap estimate (close * volume is a proxy; real = close * shares outstanding)
        # For now, we filter by price and volume as proxy
        # Actual market cap would need shares outstanding from listed info
        position_value = close_price * LOT_SIZE
        if position_value < 20_000:  # 2万円未満は除外
            return False

        return True

    async def _get_recent_prices(
        self, ticker: str, ref: date,
    ) -> list[dict[str, Any]]:
        """Fetch and cache recent daily prices for a ticker."""
        if ticker not in self._price_cache:
            from_date = (ref - timedelta(days=self._lookback_days + 10)).isoformat()
            to_date = ref.isoformat()
            try:
                prices = await self._client.get_prices_daily(ticker, from_date, to_date)
                self._price_cache[ticker] = prices
            except Exception as e:
                logger.warning("Failed to fetch prices for {}: {}", ticker, e)
                self._price_cache[ticker] = []
        return self._price_cache[ticker]
