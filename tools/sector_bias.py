"""
米国セクター → 日本セクターバイアス (lead-lag signal)

前日の米国セクターETFの騰落率から、翌営業日の日本株個別銘柄に対して
「属する業種が追い風か逆風か」を数値化する。

この機能は:
- 単独で売買しない
- watchlist スコアの加点/減点 (+/-0.10 程度) に使う
- vwap_reclaim の confidence 補助 (+/-0.05 程度) に使う

制約:
- PCA の厳密再現はしない (軽量 lead-lag score)
- 米国データは Yahoo Finance Chart API から取得 (外部依存なし)
- US→JP セクターマッピングは手動テーブル (保守しやすい)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Config defaults (overridable from core.config or run_paper.py)
# ---------------------------------------------------------------------------

SECTOR_BIAS_WEIGHT_WATCHLIST = 0.10     # watchlist combined score への加算重み
SECTOR_BIAS_WEIGHT_CONFIDENCE = 0.05    # vwap_reclaim confidence への加算重み
SECTOR_BIAS_CLIP = 1.0                  # bias score の上下限
RISK_OFF_DAMPING = 0.5                  # VIX高 / SPY大幅安時の全体減衰

# ---------------------------------------------------------------------------
# 米国セクターETFティッカー
# ---------------------------------------------------------------------------

US_SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# SPY: 市場全体 / ^VIX は直接取りにくいので SPY の騰落で代用
US_MARKET_TICKERS = ["SPY"]

# ---------------------------------------------------------------------------
# 米国セクター → 日本33業種マッピング
# キー: 米国セクターETFティッカー
# 値:   対応する東証33業種コード (S33) のリスト
#
# 東証33業種コード参照:
# 0050水産, 1050鉱業, 2050建設, 3050食料品, 3100繊維, 3150パルプ,
# 3200化学, 3250医薬品, 3300石油, 3350ゴム, 3400ガラス, 3450鉄鋼,
# 3500非鉄, 3550金属, 3600機械, 3650電気機器, 3700輸送用機器,
# 3750精密, 3800その他製品, 4050電気ガス, 5050陸運, 5100海運,
# 5150空運, 5200倉庫, 5250情報通信, 6050卸売, 6100小売,
# 7050銀行, 7100証券, 7150保険, 7200その他金融, 8050不動産,
# 9050サービス
# ---------------------------------------------------------------------------

US_TO_JP_SECTOR_MAP: dict[str, list[str]] = {
    "XLK": ["3650", "5250", "3750"],          # Tech → 電気機器, 情報通信, 精密
    "XLF": ["7050", "7100", "7150", "7200"],   # Financials → 銀行, 証券, 保険, その他金融
    "XLE": ["1050", "3300"],                    # Energy → 鉱業, 石油
    "XLV": ["3250"],                            # Healthcare → 医薬品
    "XLY": ["6100", "9050", "5150"],           # Consumer Disc → 小売, サービス, 空運
    "XLP": ["3050", "6050"],                    # Consumer Staples → 食料品, 卸売
    "XLI": ["3600", "3700", "2050"],           # Industrials → 機械, 輸送用機器, 建設
    "XLB": ["3200", "3450", "3500", "3400"],   # Materials → 化学, 鉄鋼, 非鉄, ガラス
    "XLU": ["4050"],                            # Utilities → 電気ガス
    "XLRE": ["8050"],                           # Real Estate → 不動産
    "XLC": ["5250"],                            # Communication → 情報通信
}

# 逆引き: 日本33業種コード → 対応する米国ETFリスト
_JP_TO_US_MAP: dict[str, list[str]] = {}
for _etf, _sectors in US_TO_JP_SECTOR_MAP.items():
    for _s in _sectors:
        _JP_TO_US_MAP.setdefault(_s, []).append(_etf)


# ---------------------------------------------------------------------------
# データ取得 (Yahoo Finance Chart API, no yfinance dependency)
# ---------------------------------------------------------------------------

_YF_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


async def _fetch_prev_day_return(
    session: aiohttp.ClientSession, symbol: str,
) -> float | None:
    """Yahoo Finance Chart API で前営業日の騰落率を取得。

    range=5d, interval=1d で直近5営業日のデータを取得し、
    最後の2営業日からリターンを計算。
    """
    url = _YF_CHART_URL.format(symbol=symbol)
    params = {"range": "5d", "interval": "1d", "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.debug(f"[sector_bias] {symbol}: HTTP {resp.status}")
                return None
            data = await resp.json()
            result = data.get("chart", {}).get("result", [])
            if not result:
                return None
            closes = result[0].get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
            if not closes or len(closes) < 2:
                # fallback to regular close
                closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
            if not closes or len(closes) < 2:
                return None
            # 最後の2つの有効な終値からリターン計算
            valid = [c for c in closes if c is not None]
            if len(valid) < 2:
                return None
            prev, last = valid[-2], valid[-1]
            if prev <= 0:
                return None
            return (last - prev) / prev
    except Exception as e:
        logger.debug(f"[sector_bias] {symbol} fetch error: {e}")
        return None


# ---------------------------------------------------------------------------
# SectorBiasResult
# ---------------------------------------------------------------------------

@dataclass
class SectorBiasResult:
    """当日のセクターバイアス計算結果"""
    source_date: str                            # 米国データの日付
    spy_return: float = 0.0                     # SPY 前日リターン
    risk_off: bool = False                      # risk-off 判定
    us_returns: dict[str, float] = field(default_factory=dict)   # ETF→return
    jp_sector_bias: dict[str, float] = field(default_factory=dict)  # S33→bias
    fetch_success: bool = False

    def get_bias_for_sector(self, s33_code: str) -> float:
        """個別銘柄の S33 コードに対する bias score を返す"""
        return self.jp_sector_bias.get(s33_code, 0.0)

    def get_bias_label(self, s33_code: str) -> str:
        """bullish / neutral / bearish"""
        b = self.get_bias_for_sector(s33_code)
        if b > 0.3:
            return "bullish"
        elif b < -0.3:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict:
        return {
            "source_date": self.source_date,
            "spy_return": round(self.spy_return, 4),
            "risk_off": self.risk_off,
            "us_returns": {k: round(v, 4) for k, v in self.us_returns.items()},
            "jp_sector_bias": {k: round(v, 3) for k, v in self.jp_sector_bias.items()},
            "fetch_success": self.fetch_success,
        }


# ---------------------------------------------------------------------------
# メイン計算クラス
# ---------------------------------------------------------------------------

class SectorBiasCalculator:
    """米国セクターETF → 日本セクター bias score を計算。

    Usage::

        calc = SectorBiasCalculator()
        result = await calc.calculate()
        bias = result.get_bias_for_sector("3650")  # 電気機器
    """

    def __init__(
        self,
        watchlist_weight: float = SECTOR_BIAS_WEIGHT_WATCHLIST,
        confidence_weight: float = SECTOR_BIAS_WEIGHT_CONFIDENCE,
        clip: float = SECTOR_BIAS_CLIP,
        risk_off_damping: float = RISK_OFF_DAMPING,
    ):
        self.watchlist_weight = watchlist_weight
        self.confidence_weight = confidence_weight
        self.clip = clip
        self.risk_off_damping = risk_off_damping
        self._cache: SectorBiasResult | None = None
        self._cache_date: str = ""

    async def calculate(self, force: bool = False) -> SectorBiasResult:
        """前日の米国セクターETFデータを取得し、日本セクターbias scoreを計算。

        結果はキャッシュされ、同一日内は再取得しない。
        """
        today = date.today().isoformat()
        if not force and self._cache and self._cache_date == today:
            return self._cache

        result = SectorBiasResult(source_date=today)

        async with aiohttp.ClientSession() as session:
            # 1. 全ETF + SPY のリターンを取得
            all_symbols = list(US_SECTOR_ETFS.keys()) + US_MARKET_TICKERS
            returns: dict[str, float] = {}

            for symbol in all_symbols:
                ret = await _fetch_prev_day_return(session, symbol)
                if ret is not None:
                    returns[symbol] = ret

        if not returns:
            logger.warning("[sector_bias] 米国データ取得失敗: 全ティッカーエラー")
            self._cache = result
            self._cache_date = today
            return result

        result.fetch_success = True
        result.us_returns = {k: v for k, v in returns.items() if k in US_SECTOR_ETFS}
        result.spy_return = returns.get("SPY", 0.0)

        # 2. Risk-off 判定: SPY が -1.5% 以上下落
        if result.spy_return < -0.015:
            result.risk_off = True
            logger.info(f"[sector_bias] Risk-off detected: SPY={result.spy_return:+.2%}")

        # 3. 米国セクターリターンを z-score 化
        sector_returns = [v for k, v in returns.items() if k in US_SECTOR_ETFS]
        if len(sector_returns) >= 3:
            mean_r = np.mean(sector_returns)
            std_r = np.std(sector_returns)
            if std_r > 0.0001:
                z_scores = {k: (v - mean_r) / std_r for k, v in returns.items() if k in US_SECTOR_ETFS}
            else:
                z_scores = {k: 0.0 for k in US_SECTOR_ETFS if k in returns}
        else:
            z_scores = {k: 0.0 for k in US_SECTOR_ETFS if k in returns}

        # 4. 日本33業種への伝播
        jp_bias: dict[str, float] = {}
        for jp_s33, us_etfs in _JP_TO_US_MAP.items():
            scores = [z_scores[etf] for etf in us_etfs if etf in z_scores]
            if scores:
                raw = float(np.mean(scores))
                # risk-off 時は正のバイアスを減衰
                if result.risk_off and raw > 0:
                    raw *= self.risk_off_damping
                jp_bias[jp_s33] = max(-self.clip, min(self.clip, raw))

        result.jp_sector_bias = jp_bias

        # ログ
        bullish = [(k, v) for k, v in jp_bias.items() if v > 0.3]
        bearish = [(k, v) for k, v in jp_bias.items() if v < -0.3]
        if bullish or bearish:
            logger.info(
                f"[sector_bias] SPY={result.spy_return:+.2%} "
                f"bullish={len(bullish)} bearish={len(bearish)} risk_off={result.risk_off}"
            )

        # キャッシュ & ファイル保存
        self._cache = result
        self._cache_date = today

        # knowledge への保存
        try:
            out_path = Path("knowledge/sector_bias.json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"[sector_bias] JSON保存失敗: {e}")

        return result

    def get_watchlist_adjustment(self, sector_bias: SectorBiasResult, s33_code: str) -> float:
        """watchlist combined score への加算値を返す。

        Returns:
            -0.10 ~ +0.10 程度の加算値
        """
        bias = sector_bias.get_bias_for_sector(s33_code)
        return round(bias * self.watchlist_weight, 4)

    def get_confidence_adjustment(self, sector_bias: SectorBiasResult, s33_code: str) -> float:
        """vwap_reclaim confidence への加算値を返す。

        Returns:
            -0.05 ~ +0.05 程度の加算値
        """
        bias = sector_bias.get_bias_for_sector(s33_code)
        return round(bias * self.confidence_weight, 4)
