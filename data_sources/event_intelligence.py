"""
Event Intelligence — 自前実装版

外部APIを使わず、TDnet + J-Quants + 日足データから
イベント重要度・波及スコア・エビデンスを自前計算する。

Novaquity APIのロジックを再現:
- event_importance_score: TDnetの開示タイプ+キーワードから算出
- event_freshness_score: 開示時刻からの経過時間で算出
- propagation_score: 同業種の値動き相関から波及効果を推定
- company_feature_score: 出来高・ATR・時価総額代理から算出
- traceability_score: ソースURL・エビデンス数から算出
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from data_sources.tdnet import DisclosureInfo, DisclosureType


# ------------------------------------------------------------------
# Importance mapping
# ------------------------------------------------------------------

_EVENT_TYPE_IMPORTANCE: dict[str, float] = {
    "上方修正": 0.9,
    "業績予想の修正": 0.85,
    "下方修正": 0.9,
    "自社株買い": 0.7,
    "自社株買": 0.7,
    "自己株式": 0.7,
    "決算短信": 0.6,
    "決算": 0.6,
    "配当": 0.5,
    "増配": 0.65,
    "株式分割": 0.5,
    "合併": 0.8,
    "公開買付": 0.8,
    "TOB": 0.8,
    "MBO": 0.85,
    "業務提携": 0.6,
    "資本提携": 0.65,
    "新規上場": 0.5,
    "上場廃止": 0.7,
    "債務超過": 0.9,
    "民事再生": 0.95,
    "特別利益": 0.6,
    "特別損失": 0.65,
    "過去最高": 0.75,
    "大幅": 0.7,
    "ストップ高": 0.8,
    "ストップ安": 0.8,
    "新製品": 0.5,
    "新サービス": 0.5,
}

_DISCLOSURE_TYPE_IMPORTANCE: dict[DisclosureType, float] = {
    DisclosureType.UPWARD_REVISION: 0.9,
    DisclosureType.DOWNWARD_REVISION: 0.9,
    DisclosureType.BUYBACK: 0.7,
    DisclosureType.EARNINGS: 0.6,
    DisclosureType.DIVIDEND: 0.5,
    DisclosureType.STOCK_SPLIT: 0.5,
    DisclosureType.MERGER: 0.8,
    DisclosureType.TOB: 0.8,
    DisclosureType.BUSINESS_ALLIANCE: 0.6,
    DisclosureType.NEW_LISTING: 0.5,
    DisclosureType.DELISTING: 0.7,
    DisclosureType.OTHER: 0.3,
}

# 業種グループ（波及スコア計算用）
# 同じグループ内の銘柄はイベント波及しやすい
_SECTOR_GROUPS: dict[str, list[str]] = {
    "通信": ["9432", "9434", "9433", "4689", "3778"],
    "自動車": ["7201", "7203", "7211", "7267", "7269"],
    "電機": ["6740", "6758", "6861", "6902", "6981"],
    "銀行": ["8306", "8316", "8411", "8308", "7186"],
    "化学": ["4005", "4183", "4063", "4188", "4021"],
    "鉄鋼": ["5401", "5411", "3315", "5406", "5423"],
    "商社": ["8002", "8031", "8058", "8053", "8015"],
    "不動産": ["8801", "8802", "8830", "3289", "8804"],
    "IT": ["4689", "3656", "9984", "4755", "3769"],
}


@dataclass
class EventIntelligence:
    """統合イベントインテリジェンス"""
    symbol: str
    source: str  # "tdnet" | "internal"
    event_type: str
    event_timestamp: datetime

    event_importance_score: float = 0.0
    event_freshness_score: float = 0.0
    company_feature_score: float = 0.0
    propagation_score: float = 0.0

    propagation_source_symbol: str = ""
    propagation_path_type: str = ""

    evidence_count: int = 0
    evidence_summary: str = ""
    traceability_score: float = 0.0

    raw_payload_ref: str = ""


# ------------------------------------------------------------------
# Score calculation
# ------------------------------------------------------------------

def _calc_freshness_score(event_time: datetime) -> float:
    """経過時間から鮮度スコアを算出"""
    now = datetime.now(tz=event_time.tzinfo if event_time.tzinfo else None)
    elapsed_hours = (now - event_time).total_seconds() / 3600.0
    if elapsed_hours < 0:
        return 1.0
    if elapsed_hours <= 1:
        return 1.0
    if elapsed_hours <= 3:
        return 0.9
    if elapsed_hours <= 6:
        return 0.7
    if elapsed_hours <= 12:
        return 0.5
    if elapsed_hours <= 24:
        return 0.3
    return 0.1


def _calc_importance_from_text(text: str) -> float:
    """テキストからキーワードマッチで重要度を算出"""
    best = 0.3
    for keyword, score in _EVENT_TYPE_IMPORTANCE.items():
        if keyword in text:
            best = max(best, score)
    return best


def _calc_company_feature_score(
    df: Optional[pd.DataFrame] = None,
    price: float = 0,
) -> float:
    """出来高・ボラティリティ・価格帯から企業特徴スコアを算出"""
    if df is None or df.empty:
        return 0.0

    score = 0.0

    # 出来高が平均より多い → 注目度が高い
    if "volume" in df.columns:
        vol = df["volume"].astype(float)
        if len(vol) >= 20:
            vol_ratio = float(vol.iloc[-1] / vol.tail(20).mean()) if vol.tail(20).mean() > 0 else 1.0
            if vol_ratio > 2.0:
                score += 0.3
            elif vol_ratio > 1.5:
                score += 0.2
            elif vol_ratio > 1.0:
                score += 0.1

    # ATR%が適度（トレードしやすい）
    if "close" in df.columns and "high" in df.columns and "low" in df.columns:
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        if len(close) >= 14:
            tr = (high - low).tail(14)
            atr_pct = float(tr.mean() / close.iloc[-1] * 100) if close.iloc[-1] > 0 else 0
            if 1.0 <= atr_pct <= 5.0:
                score += 0.3  # 適度なボラ
            elif atr_pct > 5.0:
                score += 0.1  # 高ボラ（リスク高め）

    # 価格帯（低位株は動きやすい）
    if 50 < price <= 500:
        score += 0.2
    elif 500 < price <= 2000:
        score += 0.1

    return min(score, 1.0)


def _calc_propagation_score(
    ticker: str,
    all_events: dict[str, "DisclosureInfo"] = None,
) -> tuple[float, str, str]:
    """同業種のイベント波及スコアを計算

    Returns:
        (propagation_score, source_symbol, path_type)
    """
    if not all_events:
        return 0.0, "", ""

    # この銘柄が属するセクターを特定
    ticker_4 = ticker[:4]
    my_sectors = []
    for sector_name, members in _SECTOR_GROUPS.items():
        if ticker_4 in members:
            my_sectors.append((sector_name, members))

    if not my_sectors:
        return 0.0, "", ""

    # 同セクターの他銘柄にイベントがあるか
    best_score = 0.0
    best_source = ""
    best_path = ""

    for sector_name, members in my_sectors:
        for member in members:
            if member == ticker_4:
                continue
            event = all_events.get(member)
            if event:
                # イベントの重要度に応じて波及スコアを計算
                importance = _DISCLOSURE_TYPE_IMPORTANCE.get(
                    event.disclosure_type, 0.3
                )
                freshness = _calc_freshness_score(event.timestamp)
                prop_score = importance * freshness * 0.7  # 波及は元の70%
                if prop_score > best_score:
                    best_score = prop_score
                    best_source = member
                    best_path = f"同業種({sector_name})"

    return min(best_score, 1.0), best_source, best_path


# ------------------------------------------------------------------
# Transformer
# ------------------------------------------------------------------

def from_tdnet_disclosure(
    disclosure: DisclosureInfo,
    df: Optional[pd.DataFrame] = None,
    price: float = 0,
    all_events: dict[str, "DisclosureInfo"] = None,
) -> EventIntelligence:
    """TDnet開示情報からEventIntelligenceを生成（自前計算）"""
    importance = _DISCLOSURE_TYPE_IMPORTANCE.get(disclosure.disclosure_type, 0.3)
    importance = max(importance, _calc_importance_from_text(disclosure.title))

    freshness = _calc_freshness_score(disclosure.timestamp)
    company_feature = _calc_company_feature_score(df, price)
    prop_score, prop_source, prop_path = _calc_propagation_score(
        disclosure.ticker, all_events
    )

    traceability = 0.8 if disclosure.url else 0.5

    intel = EventIntelligence(
        symbol=disclosure.ticker,
        source="tdnet",
        event_type=disclosure.disclosure_type.value,
        event_timestamp=disclosure.timestamp,
        event_importance_score=importance,
        event_freshness_score=freshness,
        company_feature_score=company_feature,
        propagation_score=prop_score,
        propagation_source_symbol=prop_source,
        propagation_path_type=prop_path,
        evidence_count=1,
        evidence_summary=disclosure.title,
        traceability_score=traceability,
        raw_payload_ref=disclosure.url,
    )

    logger.debug(
        "EventIntel: {} [{}] importance={:.2f} fresh={:.2f} prop={:.2f} company={:.2f}",
        disclosure.ticker, disclosure.disclosure_type.value,
        importance, freshness, prop_score, company_feature,
    )
    return intel


def from_price_action(
    ticker: str,
    df: pd.DataFrame,
    price: float = 0,
    all_events: dict[str, "DisclosureInfo"] = None,
) -> Optional[EventIntelligence]:
    """TDnet開示がない銘柄でも、価格アクションから疑似イベントを検出

    急騰（+3%以上）+ 出来高急増（2倍以上）= 何かが起きている
    """
    if df is None or len(df) < 5:
        return None

    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    if len(close) < 2 or close.iloc[-2] <= 0:
        return None

    # 日次リターン
    daily_ret = (close.iloc[-1] / close.iloc[-2] - 1) * 100
    # 出来高比率
    vol_avg = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
    vol_ratio = float(volume.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0

    # 急騰 + 出来高急増 = 疑似イベント
    if abs(daily_ret) >= 3.0 and vol_ratio >= 2.0:
        direction = "急騰" if daily_ret > 0 else "急落"
        importance = min(abs(daily_ret) / 10.0, 0.8)  # 10%で0.8
        company_feature = _calc_company_feature_score(df, price)
        prop_score, prop_source, prop_path = _calc_propagation_score(
            ticker, all_events
        )

        return EventIntelligence(
            symbol=ticker[:4],
            source="price_action",
            event_type=f"{direction}({daily_ret:+.1f}% vol{vol_ratio:.1f}x)",
            event_timestamp=datetime.now(),
            event_importance_score=importance,
            event_freshness_score=1.0,  # 直近のアクション
            company_feature_score=company_feature,
            propagation_score=prop_score,
            propagation_source_symbol=prop_source,
            propagation_path_type=prop_path,
            evidence_count=1,
            evidence_summary=f"{direction} {abs(daily_ret):.1f}% 出来高{vol_ratio:.1f}倍",
            traceability_score=0.4,  # 価格アクションのみなのでやや低め
            raw_payload_ref="",
        )

    return None
