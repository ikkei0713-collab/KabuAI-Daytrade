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

# Patterns for extracting disclosure rows from TDnet HTML
_ROW_PATTERN = re.compile(r'<tr>\s*<td[^>]*kjTime.*?</tr>', re.DOTALL)
_TIME_RE = re.compile(r'kjTime[^>]*>(\d{1,2}:\d{2})')
_CODE_RE = re.compile(r'kjCode[^>]*>(\w{4,5})')
_NAME_RE = re.compile(r'kjName[^>]*>([^<]+)')
_TITLE_LINK_RE = re.compile(r'kjTitle[^>]*>.*?<a\s+href="([^"]+)"[^>]*>([^<]+)</a>', re.DOTALL)
_TITLE_TEXT_RE = re.compile(r'kjTitle[^>]*>(.*?)</td>', re.DOTALL)


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

    async def fetch_today_disclosures(self, target_date: Optional[date] = None) -> list[DisclosureInfo]:
        """本日（または指定日）の適時開示一覧を取得する。"""
        if not self._session:
            raise RuntimeError("TDnetClient must be used as async context manager")

        d = target_date or date.today()
        date_str = d.strftime("%Y%m%d")
        url = f"{TDNET_BASE_URL}I_list_001_{date_str}.html"
        logger.info("Fetching TDnet disclosures from {}", url)

        try:
            async with self._session.get(url) as resp:
                if resp.status == 404:
                    logger.info("No TDnet page for {} (holiday?)", d)
                    return []
                resp.raise_for_status()
                html = await resp.text(encoding="utf-8", errors="replace")
        except aiohttp.ClientError as e:
            logger.error("Failed to fetch TDnet listing page: {}", e)
            return []

        disclosures = self.parse_disclosure(html, d)
        logger.info("Parsed {} disclosures from TDnet for {}", len(disclosures), d)
        return disclosures

    def parse_disclosure(self, html: str, target_date: Optional[date] = None) -> list[DisclosureInfo]:
        """TDnet HTMLページをパースして開示一覧を返す。"""
        disclosures: list[DisclosureInfo] = []
        d = target_date or date.today()

        rows = _ROW_PATTERN.findall(html)
        if not rows:
            # フォールバック: 全trを取得
            rows = re.findall(r'<tr>(.*?)</tr>', html, re.DOTALL)

        for row in rows:
            try:
                time_m = _TIME_RE.search(row)
                code_m = _CODE_RE.search(row)
                name_m = _NAME_RE.search(row)
                title_link_m = _TITLE_LINK_RE.search(row)

                if not code_m:
                    continue

                ticker = code_m.group(1).strip()
                company_name = name_m.group(1).strip() if name_m else ""

                if title_link_m:
                    relative_url = title_link_m.group(1).strip()
                    title = title_link_m.group(2).strip()
                    url = f"{TDNET_BASE_URL}{relative_url}" if not relative_url.startswith("http") else relative_url
                else:
                    title_text_m = _TITLE_TEXT_RE.search(row)
                    title = re.sub(r'<[^>]+>', '', title_text_m.group(1)).strip() if title_text_m else ""
                    url = ""

                if not title:
                    continue

                # Timestamp
                if time_m:
                    h, m = time_m.group(1).split(":")
                    timestamp = datetime(d.year, d.month, d.day, int(h), int(m))
                else:
                    timestamp = datetime(d.year, d.month, d.day, 15, 0)

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
