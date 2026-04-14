"""
夜間・寄り前マーケット情報収集

市場が閉まっている間の情報を網羅的に収集:
1. 日経225先物 (夜間取引)
2. 米国主要指数 (S&P500, NASDAQ, DOW)
3. 為替 (USD/JPY, EUR/JPY)
4. VIX (恐怖指数)
5. 原油・金
6. アジア先物 (上海, 香港)
7. セクターETF詳細
8. 経済指標カレンダー的な動き
"""

import asyncio
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv
load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

JST = ZoneInfo("Asia/Tokyo")
TODAY = date.today()

# Yahoo Finance chart API
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

OVERNIGHT_SYMBOLS = {
    # 日経先物
    "NKD=F":   "日経225先物",
    # 米国主要指数
    "^GSPC":   "S&P500",
    "^IXIC":   "NASDAQ",
    "^DJI":    "ダウ平均",
    "^RUT":    "Russell2000",
    # 恐怖指数
    "^VIX":    "VIX",
    # 為替
    "JPY=X":   "USD/JPY",
    "EURJPY=X":"EUR/JPY",
    "CNYJPY=X":"CNY/JPY",
    # コモディティ
    "CL=F":    "原油WTI",
    "GC=F":    "金",
    # アジア
    "^HSI":    "香港ハンセン",
    "000001.SS":"上海総合",
    # 欧州
    "^STOXX50E":"ユーロSTOXX50",
    # 米国セクターETF (前日騰落)
    "XLK":     "テクノロジー",
    "XLF":     "金融",
    "XLE":     "エネルギー",
    "XLV":     "ヘルスケア",
    "XLI":     "資本財",
    "XLY":     "一般消費財",
    "XLP":     "生活必需品",
    "XLRE":    "不動産",
    "XLU":     "公共事業",
    "XLC":     "通信",
    "XLB":     "素材",
    # 半導体 (日本テック株に直結)
    "SMH":     "半導体ETF",
    "SOXX":    "半導体指数",
}


async def fetch_quote(session: aiohttp.ClientSession, symbol: str, name: str) -> dict:
    """Yahoo Financeから直近の価格情報を取得"""
    try:
        params = {"range": "5d", "interval": "1d", "includePrePost": "true"}
        headers = {"User-Agent": "Mozilla/5.0"}
        async with session.get(
            YAHOO_CHART_URL.format(symbol=symbol),
            params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                return {"symbol": symbol, "name": name, "error": f"HTTP {resp.status}"}
            data = await resp.json()

        result_data = data.get("chart", {}).get("result", [])
        if not result_data:
            return {"symbol": symbol, "name": name, "error": "no data"}

        meta = result_data[0].get("meta", {})
        indicators = result_data[0].get("indicators", {}).get("quote", [{}])[0]

        closes = [c for c in (indicators.get("close") or []) if c is not None]
        if len(closes) < 2:
            return {
                "symbol": symbol,
                "name": name,
                "price": meta.get("regularMarketPrice", 0),
                "change_pct": 0,
            }

        current = closes[-1]
        previous = closes[-2]
        change_pct = ((current - previous) / previous * 100) if previous else 0

        return {
            "symbol": symbol,
            "name": name,
            "price": round(current, 2),
            "prev_close": round(previous, 2),
            "change_pct": round(change_pct, 2),
        }
    except Exception as e:
        return {"symbol": symbol, "name": name, "error": str(e)[:60]}


async def main():
    logger.info("=" * 60)
    logger.info(f"夜間マーケット情報収集 {TODAY} {datetime.now(JST).strftime('%H:%M:%S')}")
    logger.info("=" * 60)

    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_quote(session, sym, name)
            for sym, name in OVERNIGHT_SYMBOLS.items()
        ]
        results = await asyncio.gather(*tasks)

    # カテゴリ分類
    categories = {
        "日経先物": [],
        "米国指数": [],
        "恐怖指数": [],
        "為替": [],
        "コモディティ": [],
        "アジア": [],
        "欧州": [],
        "米国セクター": [],
        "半導体": [],
    }

    cat_map = {
        "NKD=F": "日経先物",
        "^GSPC": "米国指数", "^IXIC": "米国指数", "^DJI": "米国指数", "^RUT": "米国指数",
        "^VIX": "恐怖指数",
        "JPY=X": "為替", "EURJPY=X": "為替", "CNYJPY=X": "為替",
        "CL=F": "コモディティ", "GC=F": "コモディティ",
        "^HSI": "アジア", "000001.SS": "アジア",
        "^STOXX50E": "欧州",
        "SMH": "半導体", "SOXX": "半導体",
    }

    for r in results:
        sym = r["symbol"]
        cat = cat_map.get(sym, "米国セクター")
        categories[cat].append(r)

    # サマリー出力
    logger.info("")
    for cat_name, items in categories.items():
        if not items:
            continue
        logger.info(f"【{cat_name}】")
        for item in items:
            if "error" in item:
                logger.warning(f"  {item['name']}: エラー ({item['error'][:40]})")
            else:
                arrow = "↑" if item["change_pct"] > 0 else "↓" if item["change_pct"] < 0 else "→"
                logger.info(
                    f"  {item['name']:12s} {item['price']:>12,.2f}  "
                    f"{arrow} {item['change_pct']:+.2f}%"
                )
        logger.info("")

    # 市場判断
    sp500 = next((r for r in results if r["symbol"] == "^GSPC" and "error" not in r), None)
    vix = next((r for r in results if r["symbol"] == "^VIX" and "error" not in r), None)
    nk_fut = next((r for r in results if r["symbol"] == "NKD=F" and "error" not in r), None)
    usdjpy = next((r for r in results if r["symbol"] == "JPY=X" and "error" not in r), None)
    smh = next((r for r in results if r["symbol"] == "SMH" and "error" not in r), None)

    signals = []
    if sp500 and sp500["change_pct"] > 1.0:
        signals.append("米国大幅高 → 日本株追い風")
    elif sp500 and sp500["change_pct"] < -1.0:
        signals.append("米国大幅安 → 日本株逆風")

    if vix and vix["price"] > 25:
        signals.append(f"VIX={vix['price']:.1f} → リスクオフ警戒")
    elif vix and vix["price"] < 15:
        signals.append(f"VIX={vix['price']:.1f} → 低ボラ環境")

    if nk_fut and nk_fut["change_pct"] > 0.5:
        signals.append(f"日経先物 +{nk_fut['change_pct']:.1f}% → ギャップアップ期待")
    elif nk_fut and nk_fut["change_pct"] < -0.5:
        signals.append(f"日経先物 {nk_fut['change_pct']:.1f}% → ギャップダウン警戒")

    if usdjpy and usdjpy["change_pct"] > 0.5:
        signals.append(f"円安 {usdjpy['change_pct']:+.1f}% → 輸出株有利")
    elif usdjpy and usdjpy["change_pct"] < -0.5:
        signals.append(f"円高 {usdjpy['change_pct']:+.1f}% → 内需株有利")

    if smh and smh["change_pct"] > 1.5:
        signals.append(f"半導体 +{smh['change_pct']:.1f}% → テック株強い")
    elif smh and smh["change_pct"] < -1.5:
        signals.append(f"半導体 {smh['change_pct']:.1f}% → テック株弱い")

    logger.info("=" * 60)
    logger.info("【市場シグナル】")
    if signals:
        for s in signals:
            logger.info(f"  → {s}")
    else:
        logger.info("  → 特段のシグナルなし (通常環境)")
    logger.info("=" * 60)

    # JSON保存
    output = {
        "date": str(TODAY),
        "timestamp": datetime.now(JST).isoformat(),
        "quotes": {r["symbol"]: r for r in results},
        "categories": {k: v for k, v in categories.items()},
        "signals": signals,
        "summary": {
            "sp500_chg": sp500["change_pct"] if sp500 else None,
            "vix": vix["price"] if vix else None,
            "nk_futures_chg": nk_fut["change_pct"] if nk_fut else None,
            "usdjpy_chg": usdjpy["change_pct"] if usdjpy else None,
            "smh_chg": smh["change_pct"] if smh else None,
        },
    }
    out_path = Path("knowledge/overnight_scan.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"結果を {out_path} に保存")


if __name__ == "__main__":
    asyncio.run(main())
