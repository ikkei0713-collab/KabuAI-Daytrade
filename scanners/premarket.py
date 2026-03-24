"""
Pre-market scanner for identifying day-trade candidates.

Scans for gap-ups/downs, TDnet events, and unusual volume
before the TSE opening bell at 09:00 JST.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

from loguru import logger

from core.config import settings
from data_sources.jquants import JQuantsClient


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

GAP_UP_THRESHOLD_PCT = 2.0      # 2% gap up
GAP_DOWN_THRESHOLD_PCT = -2.0   # 2% gap down
VOLUME_SPIKE_RATIO = 2.0        # 平均の2倍以上
EVENT_LOOKBACK_DAYS = 3         # TDnet events within 3 days
MAX_WATCHLIST_SIZE = 50          # 有料プラン: 広範囲スクリーニング


@dataclass
class ScanResult:
    """A single scan result with ticker, reason, and feature snapshot."""
    ticker: str
    reason: str
    features: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


class PreMarketScanner:
    """
    Scans for pre-market signals before the TSE opens.

    Identifies stocks with:
    - Significant overnight gaps (gap-up / gap-down)
    - Recent TDnet corporate events (earnings, guidance, M&A)
    - Unusual pre-market volume spikes

    Usage::

        async with JQuantsClient(refresh_token="...") as client:
            scanner = PreMarketScanner(client)
            watchlist = await scanner.generate_watchlist(tickers, ref_date)
    """

    def __init__(
        self,
        jquants_client: JQuantsClient,
        lookback_days: int = 20,
        price_cache: Optional[dict[str, list[dict[str, Any]]]] = None,
    ) -> None:
        self._client = jquants_client
        self._lookback_days = lookback_days
        self._price_cache: dict[str, list[dict[str, Any]]] = price_cache or {}
        # 個別API呼び出し用: 同時2リクエスト + sleep で429回避
        self._api_semaphore = asyncio.Semaphore(2)

    # ------------------------------------------------------------------
    # Public scanning methods
    # ------------------------------------------------------------------

    async def scan_gaps(
        self,
        tickers: list[str],
        ref_date: Optional[date] = None,
    ) -> list[ScanResult]:
        """
        Find tickers with significant overnight gaps.

        A gap is measured as:
            gap_pct = (today_open - prev_close) / prev_close * 100

        Returns list of ScanResult for tickers with |gap| >= threshold.
        """
        ref = ref_date or date.today()
        logger.info("PreMarketScanner: scanning gaps for {} tickers", len(tickers))
        results: list[ScanResult] = []

        for ticker in tickers:
            prices = await self._get_recent_prices(ticker, ref)
            if len(prices) < 2:
                continue

            latest = prices[-1]
            prev = prices[-2]

            prev_close = float(prev.get("AdjustmentClose", prev.get("Close", 0)))
            today_open = float(latest.get("AdjustmentOpen", latest.get("Open", 0)))
            today_high = float(latest.get("AdjustmentHigh", latest.get("High", 0)))
            today_low = float(latest.get("AdjustmentLow", latest.get("Low", 0)))
            today_close = float(latest.get("AdjustmentClose", latest.get("Close", 0)))
            today_volume = float(latest.get("AdjustmentVolume", latest.get("Volume", 0)))

            if prev_close == 0:
                continue

            gap_pct = ((today_open - prev_close) / prev_close) * 100.0

            if gap_pct >= GAP_UP_THRESHOLD_PCT:
                direction = "gap_up"
                reason = f"ギャップアップ +{gap_pct:.1f}% (前日終値: {prev_close:.0f} -> 本日始値: {today_open:.0f})"
            elif gap_pct <= GAP_DOWN_THRESHOLD_PCT:
                direction = "gap_down"
                reason = f"ギャップダウン {gap_pct:.1f}% (前日終値: {prev_close:.0f} -> 本日始値: {today_open:.0f})"
            else:
                continue

            # Calculate intraday range
            intraday_range = (today_high - today_low) / prev_close * 100 if prev_close else 0

            features = {
                "gap_pct": gap_pct,
                "prev_close": prev_close,
                "today_open": today_open,
                "today_high": today_high,
                "today_low": today_low,
                "today_close": today_close,
                "today_volume": today_volume,
                "intraday_range_pct": intraday_range,
                "direction": 1.0 if direction == "gap_up" else -1.0,
            }

            score = abs(gap_pct) / 10.0  # Normalize roughly to 0-1 range
            results.append(ScanResult(
                ticker=ticker, reason=reason, features=features, score=min(score, 1.0),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("PreMarketScanner: found {} gap signals", len(results))
        return results

    async def scan_events(
        self,
        tickers: list[str],
        ref_date: Optional[date] = None,
    ) -> list[ScanResult]:
        """
        Find tickers with recent TDnet corporate events.

        Checks for:
        - Earnings announcements (決算発表)
        - Guidance revisions (業績修正)
        - Dividend changes (配当変更)
        - Share buybacks (自社株買い)
        - M&A / restructuring

        Note: Uses J-Quants financial statements endpoint as proxy.
        Real TDnet integration would use a dedicated feed.
        """
        ref = ref_date or date.today()
        logger.info("PreMarketScanner: scanning events for {} tickers", len(tickers))
        results: list[ScanResult] = []

        for ticker in tickers:
            try:
                async with self._api_semaphore:
                    statements = await self._client.get_financial_statements(
                        ticker, report_date=ref.isoformat(),
                    )
                    await asyncio.sleep(0.5)  # 財務API throttle
            except Exception as e:
                logger.debug("Failed to fetch statements for {}: {}", ticker, e)
                continue

            if not statements:
                continue

            # Check for recent filings
            for stmt in statements:
                disclosure_date_str = stmt.get("DisclosedDate", "")
                if not disclosure_date_str:
                    continue

                try:
                    disclosure_date = datetime.strptime(
                        disclosure_date_str, "%Y-%m-%d",
                    ).date()
                except ValueError:
                    continue

                days_since = (ref - disclosure_date).days
                if days_since < 0 or days_since > EVENT_LOOKBACK_DAYS:
                    continue

                # Determine event type
                type_of_document = stmt.get("TypeOfDocument", "")
                event_type = self._classify_event(type_of_document)

                if event_type == "unknown":
                    continue

                # Extract key financial metrics for features
                net_sales = float(stmt.get("NetSales", 0) or 0)
                operating_income = float(stmt.get("OperatingIncome", 0) or 0)
                forecast_net_sales = float(stmt.get("ForecastNetSales", 0) or 0)
                forecast_operating = float(stmt.get("ForecastOperatingIncome", 0) or 0)

                reason = (
                    f"TDnet: {event_type} (開示日: {disclosure_date_str}, "
                    f"書類: {type_of_document})"
                )

                features = {
                    "event_type": hash(event_type) % 100 / 100.0,
                    "days_since_event": float(days_since),
                    "net_sales": net_sales,
                    "operating_income": operating_income,
                    "forecast_net_sales": forecast_net_sales,
                    "forecast_operating_income": forecast_operating,
                }

                # Events closer to today score higher
                score = max(0.0, 1.0 - days_since * 0.2)
                results.append(ScanResult(
                    ticker=ticker, reason=reason, features=features, score=score,
                ))
                break  # One event per ticker is enough

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("PreMarketScanner: found {} event signals", len(results))
        return results

    async def scan_volume(
        self,
        tickers: list[str],
        ref_date: Optional[date] = None,
    ) -> list[ScanResult]:
        """
        Find tickers with unusual volume compared to their average.

        Volume ratio = today's volume / N-day average volume.
        Flags tickers where ratio >= VOLUME_SPIKE_RATIO.
        """
        ref = ref_date or date.today()
        logger.info("PreMarketScanner: scanning volume for {} tickers", len(tickers))
        results: list[ScanResult] = []

        for ticker in tickers:
            prices = await self._get_recent_prices(ticker, ref)
            if len(prices) < 5:
                continue

            # Calculate average volume (excluding today)
            historical = prices[:-1]
            volumes = [
                float(p.get("AdjustmentVolume", p.get("Volume", 0)))
                for p in historical
            ]
            avg_volume = sum(volumes) / len(volumes) if volumes else 1.0

            # Today's volume
            latest = prices[-1]
            today_volume = float(latest.get("AdjustmentVolume", latest.get("Volume", 0)))

            if avg_volume <= 0:
                continue

            vol_ratio = today_volume / avg_volume

            if vol_ratio < VOLUME_SPIKE_RATIO:
                continue

            today_close = float(latest.get("AdjustmentClose", latest.get("Close", 0)))
            today_open = float(latest.get("AdjustmentOpen", latest.get("Open", 0)))
            price_change_pct = (
                ((today_close - today_open) / today_open * 100) if today_open else 0
            )

            reason = (
                f"出来高急増 {vol_ratio:.1f}倍 "
                f"(本日: {today_volume:,.0f} / 平均: {avg_volume:,.0f})"
            )

            features = {
                "volume_ratio": vol_ratio,
                "today_volume": today_volume,
                "avg_volume": avg_volume,
                "today_close": today_close,
                "today_open": today_open,
                "price_change_pct": price_change_pct,
            }

            score = min(vol_ratio / 5.0, 1.0)  # Normalize: 5x = score 1.0
            results.append(ScanResult(
                ticker=ticker, reason=reason, features=features, score=score,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("PreMarketScanner: found {} volume signals", len(results))
        return results

    async def generate_watchlist(
        self,
        tickers: list[str],
        ref_date: Optional[date] = None,
    ) -> list[ScanResult]:
        """
        Generate a combined, ranked watchlist from all scan types.

        Runs gap, event, and volume scans in parallel, then merges and
        deduplicates results, ranking by combined score.
        """
        ref = ref_date or date.today()
        logger.info(
            "PreMarketScanner: generating combined watchlist from {} tickers", len(tickers),
        )

        # gap・volume はキャッシュ済みデータで全銘柄スキャン可
        # events は個別APIのため上位100銘柄に限定して429回避
        gap_results = await self.scan_gaps(tickers, ref)
        volume_results = await self.scan_volume(tickers, ref)  # price cache hit
        event_tickers = tickers[:100]  # 財務API呼び出しを上位100に限定
        event_results = await self.scan_events(event_tickers, ref)

        # Merge results by ticker, combining scores and reasons
        ticker_map: dict[str, ScanResult] = {}

        for result in gap_results:
            if result.ticker not in ticker_map:
                ticker_map[result.ticker] = ScanResult(
                    ticker=result.ticker,
                    reason=result.reason,
                    features=dict(result.features),
                    score=result.score,
                )
            else:
                existing = ticker_map[result.ticker]
                existing.score += result.score * 0.4
                existing.reason += f" | {result.reason}"
                existing.features.update(result.features)

        for result in event_results:
            if result.ticker not in ticker_map:
                ticker_map[result.ticker] = ScanResult(
                    ticker=result.ticker,
                    reason=result.reason,
                    features=dict(result.features),
                    score=result.score,
                )
            else:
                existing = ticker_map[result.ticker]
                existing.score += result.score * 0.35
                existing.reason += f" | {result.reason}"
                existing.features.update(result.features)

        for result in volume_results:
            if result.ticker not in ticker_map:
                ticker_map[result.ticker] = ScanResult(
                    ticker=result.ticker,
                    reason=result.reason,
                    features=dict(result.features),
                    score=result.score,
                )
            else:
                existing = ticker_map[result.ticker]
                existing.score += result.score * 0.25
                existing.reason += f" | {result.reason}"
                existing.features.update(result.features)

        # Sort by combined score and take top N
        watchlist = sorted(ticker_map.values(), key=lambda r: r.score, reverse=True)
        watchlist = watchlist[:MAX_WATCHLIST_SIZE]

        logger.info(
            "PreMarketScanner: final watchlist has {} tickers (gap={}, event={}, vol={})",
            len(watchlist), len(gap_results), len(event_results), len(volume_results),
        )
        return watchlist

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_recent_prices(
        self, ticker: str, ref: date,
    ) -> list[dict[str, Any]]:
        """Fetch and cache recent daily prices (rate-limit aware)."""
        if ticker not in self._price_cache:
            from_date = (ref - timedelta(days=self._lookback_days + 10)).isoformat()
            to_date = ref.isoformat()
            async with self._api_semaphore:
                try:
                    prices = await self._client.get_prices_daily(ticker, from_date, to_date)
                    self._price_cache[ticker] = prices
                except Exception as e:
                    logger.warning("Failed to fetch prices for {}: {}", ticker, e)
                    self._price_cache[ticker] = []
        return self._price_cache[ticker]

    @staticmethod
    def _classify_event(type_of_document: str) -> str:
        """Classify a TDnet document type into a trading-relevant category."""
        doc_lower = type_of_document.lower()

        if any(kw in doc_lower for kw in ("決算短信", "quarterly", "annual", "earnings")):
            return "earnings"
        if any(kw in doc_lower for kw in ("業績予想", "forecast", "guidance", "修正")):
            return "guidance_revision"
        if any(kw in doc_lower for kw in ("配当", "dividend")):
            return "dividend"
        if any(kw in doc_lower for kw in ("自己株式", "buyback", "treasury")):
            return "buyback"
        if any(kw in doc_lower for kw in ("合併", "統合", "merger", "acquisition", "m&a")):
            return "manda"
        if any(kw in doc_lower for kw in ("株式分割", "split")):
            return "stock_split"
        if any(kw in doc_lower for kw in ("公開買付", "tob", "tender")):
            return "tender_offer"

        return "unknown"
