"""追加データ収集: 空売り残高 / 信用取引残高 / 指数価格 / 決算データ / ユニバーススキャン"""

import asyncio
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

JST = ZoneInfo("Asia/Tokyo")
TODAY = date.today()


async def fetch_short_selling():
    """空売り残高データ"""
    logger.info("[ShortSelling] 取得開始...")
    try:
        from data_sources.jquants import JQuantsClient
        jq = JQuantsClient()
        data = await jq.get_short_selling(TODAY.strftime("%Y-%m-%d"))
        logger.info(f"[ShortSelling] {len(data)}件取得")
        # 空売り比率が高いものを抽出
        top = sorted(data, key=lambda x: float(x.get("SellShortQuantity", 0) or 0), reverse=True)[:20]
        return {"count": len(data), "top20": top}
    except Exception as e:
        logger.error(f"[ShortSelling] エラー: {e}")
        return {"error": str(e)}


async def fetch_margin_trading():
    """信用取引残高"""
    logger.info("[MarginTrading] 取得開始...")
    try:
        from data_sources.jquants import JQuantsClient
        jq = JQuantsClient()
        data = await jq.get_margin_trading(TODAY.strftime("%Y-%m-%d"))
        logger.info(f"[MarginTrading] {len(data)}件取得")
        return {"count": len(data), "sample": data[:10] if data else []}
    except Exception as e:
        logger.error(f"[MarginTrading] エラー: {e}")
        return {"error": str(e)}


async def fetch_index_prices():
    """指数価格 (日経225, TOPIX)"""
    logger.info("[Index] 指数取得開始...")
    try:
        from data_sources.jquants import JQuantsClient
        jq = JQuantsClient()
        from_date = (TODAY - timedelta(days=30)).strftime("%Y-%m-%d")
        to_date = TODAY.strftime("%Y-%m-%d")
        data = await jq.get_index_prices(from_date, to_date)
        logger.info(f"[Index] {len(data)}件取得")
        # 直近5日分
        recent = data[-10:] if len(data) > 10 else data
        return {"count": len(data), "recent": recent}
    except Exception as e:
        logger.error(f"[Index] エラー: {e}")
        return {"error": str(e)}


async def fetch_financials():
    """決算データ (監視銘柄)"""
    logger.info("[Financials] 決算データ取得開始...")
    try:
        from data_sources.jquants import JQuantsClient
        from run_backtest_learn import CANDIDATE_CODES
        jq = JQuantsClient()
        results = {}
        for code in CANDIDATE_CODES[:10]:
            try:
                data = await jq.get_financial_statements(code)
                if data:
                    latest = data[-1] if isinstance(data, list) else data
                    results[code] = latest
                    logger.info(f"  {code}: 取得OK")
            except Exception as e:
                logger.warning(f"  {code}: {e}")
        logger.info(f"[Financials] {len(results)}銘柄取得完了")
        return {"count": len(results), "data": results}
    except Exception as e:
        logger.error(f"[Financials] エラー: {e}")
        return {"error": str(e)}


async def scan_universe():
    """ユニバーススキャン"""
    logger.info("[Universe] スキャン開始...")
    try:
        from scanners.universe import UniverseScanner
        from data_sources.jquants import JQuantsClient
        async with JQuantsClient() as client:
            scanner = UniverseScanner(client)
            watchlist = await scanner.get_today_watchlist()
        logger.info(f"[Universe] {len(watchlist)}銘柄を検出")
        return {
            "count": len(watchlist),
            "top20": [
                {"code": getattr(s, "code", getattr(s, "ticker", str(s))),
                 "score": getattr(s, "score", getattr(s, "total_score", 0))}
                for s in (watchlist[:20] if hasattr(watchlist, '__getitem__') else [])
            ]
        }
    except Exception as e:
        logger.error(f"[Universe] エラー: {e}")
        return {"error": str(e)}


async def fetch_trading_calendar():
    """取引カレンダー"""
    logger.info("[Calendar] 取得開始...")
    try:
        from data_sources.jquants import JQuantsClient
        jq = JQuantsClient()
        cal = await jq.get_trading_calendar()
        logger.info(f"[Calendar] {len(cal)}件取得")
        # 今週分
        today_str = TODAY.strftime("%Y-%m-%d")
        week_end = (TODAY + timedelta(days=7)).strftime("%Y-%m-%d")
        this_week = [c for c in cal if today_str <= c.get("Date", "") <= week_end]
        return {"total": len(cal), "this_week": this_week}
    except Exception as e:
        logger.error(f"[Calendar] エラー: {e}")
        return {"error": str(e)}


async def main():
    from datetime import datetime
    logger.info("=" * 60)
    logger.info(f"追加データ収集 {TODAY} {datetime.now(JST).strftime('%H:%M:%S')}")
    logger.info("=" * 60)

    results = await asyncio.gather(
        fetch_short_selling(),
        fetch_margin_trading(),
        fetch_index_prices(),
        fetch_financials(),
        scan_universe(),
        fetch_trading_calendar(),
        return_exceptions=True,
    )

    labels = ["short_selling", "margin_trading", "index_prices",
              "financials", "universe", "trading_calendar"]
    output = {"date": str(TODAY), "timestamp": datetime.now(JST).isoformat()}

    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            output[label] = {"error": str(result)}
            logger.error(f"[{label}] 例外: {result}")
        else:
            output[label] = result

    out_path = Path("knowledge/extra_data.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"結果を {out_path} に保存")

    # サマリー
    logger.info("=" * 60)
    for label in labels:
        d = output.get(label, {})
        if "error" in d:
            logger.warning(f"  {label}: エラー - {d['error'][:60]}")
        elif "count" in d:
            logger.info(f"  {label}: {d['count']}件")
        elif "total" in d:
            logger.info(f"  {label}: {d['total']}件")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
