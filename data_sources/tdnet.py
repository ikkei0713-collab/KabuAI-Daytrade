"""
TDnet (Timely Disclosure network) scraper for corporate disclosures.

TDnet is operated by the Tokyo Stock Exchange and publishes
timely disclosures (適時開示) from listed companies.

URL: https://www.release.tdnet.info/inbs/
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional

import aiohttp
from loguru import logger

TDNET_BASE_URL = "https://www.release.tdnet.info/inbs/"
TDNET_LISTING_URL = f"{TDNET_BASE_URL}I_list_001_0.html"

# Patterns for extracting disclosure rows from TDnet HTML
_ROW_PATTERN = re.compile(
    r'<tr[^>]*>.*?</tr>',
    re.DOTALL,
)
_LINK_PATTERN = re.compile(
    r'<a\s+href="([^"]+)"[^>]*>([^<]+)</a>',
    re.DOTALL,
)
_TICKER_PATTERN = re.compile(r'(\d{4})')
_TIME_PATTERN = re.compile(r'(\d{1,2}:\d{2})')
_TD_PATTERN = re.compile(r'<td[^>]*>(.*?)</td>', re.DOTALL)


class DisclosureType(str, Enum):
    """Classification of disclosure types."""

    EARNINGS = "決算"
    UPWARD_REVISION = "上方修正"
    DOWNWARD_REVISION = "下方修正"
    BUYBACK = "自社株買い"
    DIVIDEND = "配当"
    STOCK_SPLIT = "株式分割"
    NEW_LISTING = "新規上場"
    DELISTING = "上場廃止"
    MERGER = "合併"
    TOB = "公開買付"
    BUSINESS_ALLIANCE = "業務提携"
    OTHER = "その他"


@dataclass
class DisclosureInfo:
    """Structured representation of a single TDnet disclosure."""

    ticker: str
    title: str
    disclosure_type: DisclosureType
    timestamp: datetime
    url: str
    is_material: bool = False
    company_name: str = ""
    raw_text: str = ""


# Keywords for classifying disclosures
_CLASSIFICATION_RULES: list[tuple[list[str], DisclosureType]] = [
    (["上方修正", "業績予想の修正（上方）", "増額"], DisclosureType.UPWARD_REVISION),
    (["下方修正", "業績予想の修正（下方）", "減額"], DisclosureType.DOWNWARD_REVISION),
    (["自己株式", "自社株買", "自己株取得"], DisclosureType.BUYBACK),
    (["配当", "剰余金の配当"], DisclosureType.DIVIDEND),
    (["株式分割", "株式併合"], DisclosureType.STOCK_SPLIT),
    (["決算短信", "四半期報告", "有価証券報告", "決算"], DisclosureType.EARNINGS),
    (["新規上場"], DisclosureType.NEW_LISTING),
    (["上場廃止"], DisclosureType.DELISTING),
    (["合併"], DisclosureType.MERGER),
    (["公開買付"], DisclosureType.TOB),
    (["業務提携", "資本提携"], DisclosureType.BUSINESS_ALLIANCE),
]

# Material event types that typically move stock prices
_MATERIAL_TYPES = {
    DisclosureType.UPWARD_REVISION,
    DisclosureType.DOWNWARD_REVISION,
    DisclosureType.BUYBACK,
    DisclosureType.EARNINGS,
    DisclosureType.MERGER,
    DisclosureType.TOB,
}

# Keywords that increase the likelihood of a disclosure being material
_MATERIAL_KEYWORDS = [
    "特別利益",
    "特別損失",
    "業績予想の修正",
    "大幅",
    "過去最高",
    "ストップ高",
    "ストップ安",
    "MBO",
    "TOB",
    "債務超過",
    "民事再生",
]


class TDnetClient:
    """
    Async TDnet scraper for fetching and parsing corporate disclosures.

    Usage:
        async with TDnetClient() as client:
            disclosures = await client.fetch_today_disclosures()
            material = client.filter_material_events(disclosures)
    """

    def __init__(self, timeout: int = 30) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def __aenter__(self) -> TDnetClient:
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ja,en;q=0.9",
            },
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def fetch_today_disclosures(self) -> list[DisclosureInfo]:
        """
        本日の適時開示一覧を取得する。

        Fetches the TDnet disclosure listing page and parses all entries
        for the current trading day.

        Returns:
            List of DisclosureInfo objects.
        """
        if not self._session:
            raise RuntimeError("TDnetClient must be used as async context manager")

        logger.info("Fetching today's TDnet disclosures from {}", TDNET_LISTING_URL)

        try:
            async with self._session.get(TDNET_LISTING_URL) as resp:
                resp.raise_for_status()
                html = await resp.text(encoding="utf-8", errors="replace")
        except aiohttp.ClientError as e:
            logger.error("Failed to fetch TDnet listing page: {}", e)
            return []

        disclosures = self.parse_disclosure(html)
        logger.info("Parsed {} disclosures from TDnet", len(disclosures))
        return disclosures

    def parse_disclosure(self, html: str) -> list[DisclosureInfo]:
        """
        Parse the TDnet disclosure listing HTML page.

        Extracts disclosure entries from the HTML table, parsing out
        ticker codes, titles, timestamps, and PDF/XBRL links.

        Args:
            html: Raw HTML string from TDnet listing page.

        Returns:
            List of DisclosureInfo objects.
        """
        disclosures: list[DisclosureInfo] = []
        today = date.today()

        # TDnet pages use a table-based layout; each disclosure is a <tr>
        rows = _ROW_PATTERN.findall(html)

        for row in rows:
            try:
                cells = _TD_PATTERN.findall(row)
                if len(cells) < 4:
                    continue

                # Extract time from the first cell
                time_match = _TIME_PATTERN.search(cells[0])
                if not time_match:
                    continue
                time_str = time_match.group(1)

                # Extract ticker from the second cell
                ticker_match = _TICKER_PATTERN.search(cells[1])
                if not ticker_match:
                    continue
                ticker = ticker_match.group(1)

                # Extract company name (clean HTML tags)
                company_name = re.sub(r'<[^>]+>', '', cells[1]).strip()
                company_name = company_name.replace(ticker, '').strip()

                # Extract title and link from the third cell
                link_match = _LINK_PATTERN.search(cells[2])
                if link_match:
                    relative_url = link_match.group(1)
                    title = link_match.group(2).strip()
                    url = f"{TDNET_BASE_URL}{relative_url}" if not relative_url.startswith("http") else relative_url
                else:
                    title = re.sub(r'<[^>]+>', '', cells[2]).strip()
                    url = ""

                if not title:
                    continue

                # Parse timestamp
                try:
                    hour, minute = time_str.split(":")
                    timestamp = datetime(
                        today.year, today.month, today.day,
                        int(hour), int(minute),
                    )
                except (ValueError, IndexError):
                    timestamp = datetime.now()

                # Classify
                disclosure_type = self.classify_disclosure(title)

                disclosure = DisclosureInfo(
                    ticker=ticker,
                    title=title,
                    disclosure_type=disclosure_type,
                    timestamp=timestamp,
                    url=url,
                    company_name=company_name,
                    raw_text=row,
                )

                # Determine materiality
                disclosure.is_material = self._is_material(disclosure)
                disclosures.append(disclosure)

            except Exception as e:
                logger.debug("Failed to parse TDnet row: {}", e)
                continue

        return disclosures

    @staticmethod
    def classify_disclosure(text: str) -> DisclosureType:
        """
        Classify disclosure text into a DisclosureType.

        Uses keyword matching against a prioritized list of classification
        rules. Returns DisclosureType.OTHER if no match is found.

        Args:
            text: Disclosure title or body text.

        Returns:
            The classified DisclosureType.
        """
        for keywords, dtype in _CLASSIFICATION_RULES:
            for keyword in keywords:
                if keyword in text:
                    return dtype
        return DisclosureType.OTHER

    def filter_material_events(
        self,
        disclosures: list[DisclosureInfo],
    ) -> list[DisclosureInfo]:
        """
        Filter disclosures to only material (stock-moving) events.

        An event is considered material if:
        1. Its type is in _MATERIAL_TYPES, OR
        2. Its title contains any of _MATERIAL_KEYWORDS.

        Args:
            disclosures: List of DisclosureInfo objects.

        Returns:
            Filtered list containing only material events.
        """
        material = [d for d in disclosures if d.is_material]
        logger.info(
            "Filtered {} material events from {} total disclosures",
            len(material), len(disclosures),
        )
        return material

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_material(disclosure: DisclosureInfo) -> bool:
        """Determine if a disclosure is material (likely to move prices)."""
        # Check by type
        if disclosure.disclosure_type in _MATERIAL_TYPES:
            return True

        # Check by keyword
        for keyword in _MATERIAL_KEYWORDS:
            if keyword in disclosure.title:
                return True

        return False
