"""Yahoo Japan ファイナンスのランキングをスクレイピングして動的ユニバース構築用データを取得"""

import asyncio
import re
from typing import List

import aiohttp
from loguru import logger

_HEADERS = {"User-Agent": "Mozilla/5.0 (KabuAI/1.0)"}

_RANKINGS = {
    "volume": "https://finance.yahoo.co.jp/stocks/ranking/volume?market=all",
    "up": "https://finance.yahoo.co.jp/stocks/ranking/up?market=all",
    "down": "https://finance.yahoo.co.jp/stocks/ranking/down?market=all",
    "turnover": "https://finance.yahoo.co.jp/stocks/ranking/tradingValueHigh?market=all",
}

_CODE_RE = re.compile(r"/quote/(\d{4}|\d{3}[A-Z])\.T")


async def _fetch(session: aiohttp.ClientSession, url: str) -> List[str]:
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                return []
            html = await r.text()
            return list(dict.fromkeys(_CODE_RE.findall(html)))
    except Exception as e:
        logger.warning(f"Yahoo ranking fetch failed: {url} {e}")
        return []


async def fetch_all_rankings(limit_per_category: int = 50) -> dict:
    """4カテゴリのランキングを並列取得。返り値: {category: [code4,...]}"""
    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        tasks = {name: _fetch(session, url) for name, url in _RANKINGS.items()}
        results = await asyncio.gather(*tasks.values())
    return {name: codes[:limit_per_category] for name, codes in zip(tasks.keys(), results)}


async def fetch_dynamic_universe(limit_per_category: int = 50) -> List[str]:
    """全カテゴリを統合してユニーク銘柄リストを返す (4桁または3桁+英字)"""
    rankings = await fetch_all_rankings(limit_per_category)
    seen = set()
    universe = []
    for cat in ("turnover", "volume", "up", "down"):
        for code in rankings.get(cat, []):
            if code not in seen:
                seen.add(code)
                universe.append(code)
    logger.info(f"Yahoo動的ユニバース: {len(universe)}銘柄 ({len(rankings.get('turnover',[]))}/{len(rankings.get('volume',[]))}/{len(rankings.get('up',[]))}/{len(rankings.get('down',[]))})")
    return universe


if __name__ == "__main__":
    async def main():
        u = await fetch_dynamic_universe()
        print(f"Total: {len(u)}")
        print(u[:30])
    asyncio.run(main())
