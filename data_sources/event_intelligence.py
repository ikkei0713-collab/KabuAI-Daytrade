"""
Event intelligence schema and transformer.

Provides a unified EventIntelligence dataclass that normalizes events
from multiple sources (Novaquity API, TDnet disclosures) into a
common schema with computed scores for importance, freshness,
propagation, and traceability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from data_sources.tdnet import DisclosureInfo, DisclosureType


# ------------------------------------------------------------------
# Importance mapping by event type keyword (Japanese)
# ------------------------------------------------------------------

_EVENT_TYPE_IMPORTANCE: dict[str, float] = {
    "上方修正": 0.9,
    "下方修正": 0.9,
    "自社株買い": 0.7,
    "自社株買": 0.7,
    "自己株式": 0.7,
    "決算": 0.6,
    "決算短信": 0.6,
    "配当": 0.5,
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


@dataclass
class EventIntelligence:
    """Unified event intelligence record from any data source."""

    symbol: str
    source: str  # "novaquity" | "tdnet" | etc.
    event_type: str
    event_timestamp: datetime

    # Computed scores (0-1)
    event_importance_score: float = 0.0
    event_freshness_score: float = 0.0
    company_feature_score: float = 0.0
    propagation_score: float = 0.0

    # Propagation info
    propagation_source_symbol: str = ""
    propagation_path_type: str = ""

    # Evidence
    evidence_count: int = 0
    evidence_summary: str = ""
    traceability_score: float = 0.0

    # Reference to raw payload
    raw_payload_ref: str = ""


# ------------------------------------------------------------------
# Score calculation helpers
# ------------------------------------------------------------------

def _calc_freshness_score(event_time: datetime) -> float:
    """
    Calculate freshness score based on time elapsed since event.

    0-6h   -> 1.0
    6-12h  -> 0.7
    12-24h -> 0.4
    >24h   -> 0.1
    """
    now = datetime.now(tz=event_time.tzinfo if event_time.tzinfo else None)
    elapsed_hours = (now - event_time).total_seconds() / 3600.0

    if elapsed_hours < 0:
        # Future event — treat as maximally fresh
        return 1.0
    if elapsed_hours <= 6:
        return 1.0
    if elapsed_hours <= 12:
        return 0.7
    if elapsed_hours <= 24:
        return 0.4
    return 0.1


def _calc_importance_from_text(text: str) -> float:
    """Return importance score by matching keywords in text."""
    best = 0.3  # default for unknown types
    for keyword, score in _EVENT_TYPE_IMPORTANCE.items():
        if keyword in text:
            best = max(best, score)
    return best


# ------------------------------------------------------------------
# Transformer functions
# ------------------------------------------------------------------

def from_novaquity_event(raw: dict) -> EventIntelligence:
    """
    Transform a Novaquity API raw event dict into EventIntelligence.

    Expected raw keys (best-effort; missing keys handled gracefully):
        symbol, event_type, timestamp, importance, company_feature_score,
        propagation_score, propagation_source, propagation_path,
        evidence_count, evidence_summary, raw_id
    """
    symbol = raw.get("symbol", raw.get("ticker", ""))
    event_type = raw.get("event_type", raw.get("type", "unknown"))

    # Parse timestamp
    ts_raw = raw.get("timestamp", raw.get("event_timestamp", ""))
    try:
        event_time = datetime.fromisoformat(str(ts_raw)) if ts_raw else datetime.now()
    except (ValueError, TypeError):
        logger.warning("Could not parse Novaquity timestamp '{}', using now", ts_raw)
        event_time = datetime.now()

    # Importance: prefer API-provided, fall back to keyword matching
    importance = float(raw.get("importance", 0.0))
    if importance <= 0:
        importance = _calc_importance_from_text(event_type)

    freshness = _calc_freshness_score(event_time)

    # Propagation
    prop_score = float(raw.get("propagation_score", 0.0))
    prop_source = raw.get("propagation_source", raw.get("propagation_source_symbol", ""))
    prop_path = raw.get("propagation_path", raw.get("propagation_path_type", ""))

    # Evidence
    evidence_count = int(raw.get("evidence_count", 0))
    evidence_summary = raw.get("evidence_summary", "")

    # Traceability: higher if more evidence and explicit source
    traceability = min(1.0, evidence_count * 0.2) if evidence_count > 0 else 0.0
    if raw.get("raw_id") or raw.get("source_url"):
        traceability = max(traceability, 0.5)

    intel = EventIntelligence(
        symbol=symbol,
        source="novaquity",
        event_type=event_type,
        event_timestamp=event_time,
        event_importance_score=importance,
        event_freshness_score=freshness,
        company_feature_score=float(raw.get("company_feature_score", 0.0)),
        propagation_score=prop_score,
        propagation_source_symbol=str(prop_source),
        propagation_path_type=str(prop_path),
        evidence_count=evidence_count,
        evidence_summary=evidence_summary,
        traceability_score=traceability,
        raw_payload_ref=raw.get("raw_id", raw.get("id", "")),
    )

    logger.debug(
        "Novaquity event -> {} type={} importance={:.2f} freshness={:.2f}",
        symbol, event_type, importance, freshness,
    )
    return intel


def from_tdnet_disclosure(disclosure: DisclosureInfo) -> EventIntelligence:
    """
    Transform an existing TDnet DisclosureInfo into EventIntelligence.
    """
    importance = _DISCLOSURE_TYPE_IMPORTANCE.get(
        disclosure.disclosure_type, 0.3,
    )
    # Boost importance if material keywords found in title
    importance = max(importance, _calc_importance_from_text(disclosure.title))

    freshness = _calc_freshness_score(disclosure.timestamp)

    # TDnet disclosures have good traceability (official source with URL)
    traceability = 0.8 if disclosure.url else 0.5

    intel = EventIntelligence(
        symbol=disclosure.ticker,
        source="tdnet",
        event_type=disclosure.disclosure_type.value,
        event_timestamp=disclosure.timestamp,
        event_importance_score=importance,
        event_freshness_score=freshness,
        company_feature_score=0.0,  # not available from TDnet
        propagation_score=0.0,  # not available from TDnet
        evidence_count=1,
        evidence_summary=disclosure.title,
        traceability_score=traceability,
        raw_payload_ref=disclosure.url,
    )

    logger.debug(
        "TDnet disclosure -> {} type={} importance={:.2f} freshness={:.2f}",
        disclosure.ticker, disclosure.disclosure_type.value, importance, freshness,
    )
    return intel
