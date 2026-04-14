"""
寄り前フル準備スクリプト
- TDnet適時開示スキャン
- PreMarketScanner (ギャップ/出来高/イベント)
- セクターバイアス (米国→日本)
- マーケットレジーム判定
- StockSelector で最終候補選定
全て並列で走らせ、結果をknowledgeに保存
"""

import asyncio
import json
import sys
from datetime import date, datetime
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


async def scan_tdnet():
    """TDnet適時開示スキャン"""
    logger.info("[TDnet] 適時開示スキャン開始...")
    try:
        from data_sources.tdnet import TDnetClient
        async with TDnetClient() as client:
            disclosures = await client.fetch_today_disclosures()
        logger.info(f"[TDnet] {len(disclosures)}件の開示を取得")

        # LLM分析（同期メソッド）
        if disclosures:
            from tools.disclosure_analyzer import DisclosureAnalyzer
            analyzer = DisclosureAnalyzer()
            analyzed = []
            for d in disclosures[:20]:
                try:
                    result = analyzer.analyze(d.title, d.company_name or d.ticker)
                    analyzed.append({
                        "ticker": d.ticker,
                        "title": d.title,
                        "type": d.disclosure_type.value,
                        "direction": result.direction,
                        "magnitude": result.magnitude,
                        "category": result.category,
                    })
                except Exception as e:
                    logger.warning(f"[TDnet] 分析エラー {d.ticker}: {e}")
            logger.info(f"[TDnet] {len(analyzed)}件を分析完了")
            return {"disclosures": len(disclosures), "analyzed": analyzed}

        return {"disclosures": 0, "analyzed": []}
    except Exception as e:
        logger.error(f"[TDnet] エラー: {e}")
        return {"error": str(e)}


async def scan_premarket():
    """PreMarketScanner実行"""
    logger.info("[PreMarket] スキャン開始...")
    try:
        from data_sources.jquants import JQuantsClient
        from scanners.premarket import PreMarketScanner
        from run_backtest_learn import CANDIDATE_CODES

        jquants = JQuantsClient()
        # まず上場銘柄一覧から候補を取得
        try:
            info = await jquants.get_listed_info()
            all_codes = [s.get("Code", "") for s in info if s.get("Code")]
            # 優先銘柄 + 全体から上位
            tickers = list(dict.fromkeys(CANDIDATE_CODES + all_codes[:100]))
        except Exception:
            tickers = CANDIDATE_CODES

        logger.info(f"[PreMarket] {len(tickers)}銘柄をスキャン対象に設定")
        scanner = PreMarketScanner(jquants_client=jquants)
        results = await scanner.generate_watchlist(tickers=tickers, ref_date=TODAY)
        logger.info(f"[PreMarket] {len(results)}銘柄を検出")
        return {
            "count": len(results),
            "candidates": [
                {"ticker": r.ticker, "reason": r.reason, "score": r.score}
                for r in results[:30]
            ]
        }
    except Exception as e:
        logger.error(f"[PreMarket] エラー: {e}")
        return {"error": str(e)}


async def calc_sector_bias():
    """米国→日本セクターバイアス"""
    logger.info("[SectorBias] 計算開始...")
    try:
        from tools.sector_bias import SectorBiasCalculator
        calc = SectorBiasCalculator()
        result = await calc.calculate()
        logger.info(f"[SectorBias] 完了: risk_off={getattr(result, 'risk_off', False)}")
        return {
            "risk_off": getattr(result, "risk_off", False),
            "biases": {k: round(v, 3) for k, v in getattr(result, "sector_biases", {}).items()},
            "spy_return": round(getattr(result, "spy_return", 0), 4),
        }
    except Exception as e:
        logger.error(f"[SectorBias] エラー: {e}")
        return {"error": str(e)}


async def detect_regime():
    """マーケットレジーム判定"""
    logger.info("[Regime] レジーム判定開始...")
    try:
        from tools.market_regime import RegimeDetector
        from data_sources.jquants import JQuantsClient
        from datetime import timedelta
        import pandas as pd

        jquants = JQuantsClient()
        to_date = TODAY.strftime("%Y-%m-%d")
        from_date = (TODAY - timedelta(days=90)).strftime("%Y-%m-%d")
        # N225 ETF: 1321
        records = await jquants.get_prices_daily("1321", from_date, to_date)
        if records and len(records) > 20:
            df = pd.DataFrame(records)
            # J-Quants V2 カラム名: O,H,L,C,Vo → Open,High,Low,Close,Volume
            df = df.rename(columns={
                "O": "Open", "H": "High", "L": "Low", "C": "Close",
                "Vo": "Volume", "Va": "TurnoverValue",
            })
            detector = RegimeDetector()
            result = detector.detect(df)
            logger.info(f"[Regime] {result.regime} (conf={result.confidence:.2f})")
            return {
                "regime": result.regime,
                "confidence": result.confidence,
                "position_scale": result.position_scale,
                "strategy_weights": result.strategy_weights,
            }
        else:
            logger.warning(f"[Regime] データ不足 ({len(records) if records else 0}件)")
            return {"error": "insufficient data"}
    except Exception as e:
        logger.error(f"[Regime] エラー: {e}")
        return {"error": str(e)}


async def main():
    logger.info("=" * 60)
    logger.info(f"寄り前フル準備 {TODAY} {datetime.now(JST).strftime('%H:%M:%S')}")
    logger.info("=" * 60)

    # 全て並列実行
    results = await asyncio.gather(
        scan_tdnet(),
        scan_premarket(),
        calc_sector_bias(),
        detect_regime(),
        return_exceptions=True,
    )

    labels = ["tdnet", "premarket", "sector_bias", "regime"]
    output = {"date": str(TODAY), "timestamp": datetime.now(JST).isoformat()}

    for label, result in zip(labels, results):
        if isinstance(result, Exception):
            output[label] = {"error": str(result)}
            logger.error(f"[{label}] 例外: {result}")
        else:
            output[label] = result

    # 保存
    out_path = Path("knowledge/premarket_scan.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info(f"結果を {out_path} に保存")

    # サマリー出力
    logger.info("=" * 60)
    logger.info("サマリー:")
    if "premarket" in output and "count" in output["premarket"]:
        logger.info(f"  銘柄候補: {output['premarket']['count']}銘柄")
    if "tdnet" in output and "disclosures" in output["tdnet"]:
        logger.info(f"  適時開示: {output['tdnet']['disclosures']}件")
    if "sector_bias" in output and "spy_return" in output["sector_bias"]:
        logger.info(f"  SPY前日: {output['sector_bias']['spy_return']:+.2%}")
        logger.info(f"  リスクオフ: {output['sector_bias'].get('risk_off', '?')}")
    if "regime" in output and "regime" in output["regime"]:
        logger.info(f"  レジーム: {output['regime']['regime']} (conf={output['regime'].get('confidence', 0):.2f})")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
