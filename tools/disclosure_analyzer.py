"""
TDnet 適時開示の分析 (キーワードルールベース)

開示タイトルからキーワードマッチで分類・インパクト判定する。
外部APIへの依存なし。

分析内容:
- direction: positive / negative / neutral
- magnitude: 0.0-1.0 (株価インパクトの大きさ)
- category: guidance_up / guidance_down / buyback / earnings_beat / earnings_miss / ...
- reasoning: 判断理由 (1行)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger


@dataclass
class DisclosureAnalysis:
    """開示分析結果"""
    direction: str = "neutral"    # positive / negative / neutral
    magnitude: float = 0.0        # 0.0-1.0
    category: str = "other"       # guidance_up, buyback, earnings_beat, etc.
    reasoning: str = ""           # 判断理由
    raw_title: str = ""
    analyzed: bool = False        # 分析済みか


# (keywords, direction, magnitude, category, reasoning) — 先にマッチしたものが優先
_RULES: list[tuple[list[str], str, float, str, str]] = [
    # === 超高インパクト ===
    (["MBO"], "positive", 0.9, "tob", "MBOはプレミアム付き買付"),
    (["公開買付"], "positive", 0.8, "tob", "TOBはプレミアム期待"),
    (["上場廃止"], "negative", 0.8, "delisting", "上場廃止で流動性喪失"),

    # === 高インパクト ===
    (["上方修正", "業績予想の修正（上方）", "増額修正"], "positive", 0.6, "guidance_up", "業績上方修正"),
    (["下方修正", "業績予想の修正（下方）", "減額修正"], "negative", 0.6, "guidance_down", "業績下方修正"),
    (["特別利益"], "positive", 0.5, "special_profit", "特別利益計上"),
    (["特別損失"], "negative", 0.5, "special_loss", "特別損失計上"),

    # === 中インパクト ===
    (["自己株式の取得", "自社株買"], "positive", 0.5, "buyback", "自社株買い需給改善"),
    (["増配"], "positive", 0.4, "dividend_up", "増配で株主還元強化"),
    (["復配"], "positive", 0.4, "dividend_up", "復配は業績回復サイン"),
    (["減配"], "negative", 0.5, "dividend_down", "減配は業績悪化サイン"),
    (["無配"], "negative", 0.5, "dividend_down", "無配転落"),
    (["株式分割"], "positive", 0.3, "stock_split", "株式分割で流動性向上"),
    (["株式併合"], "neutral", 0.2, "reverse_split", "株式併合"),
    (["合併", "株式交換", "経営統合"], "neutral", 0.5, "merger", "M&A — 条件次第"),
    (["業務提携", "資本提携", "資本業務提携"], "positive", 0.3, "alliance", "提携でシナジー期待"),
    (["新株予約権", "第三者割当"], "negative", 0.4, "dilution", "希薄化懸念"),

    # === 業績予想修正（方向不明） ===
    (["業績予想の修正", "業績予想修正"], "neutral", 0.4, "guidance_revision", "業績予想修正 — 方向は本文次第"),

    # === 決算関連 ===
    (["決算短信"], "neutral", 0.3, "earnings", "決算発表 — 内容次第"),
    (["四半期報告書", "有価証券報告書"], "neutral", 0.1, "filing", "定例開示"),

    # === 低インパクト ===
    (["役員", "人事"], "neutral", 0.1, "personnel", "人事異動"),
    (["株主総会", "継続会"], "neutral", 0.1, "agm", "株主総会関連"),
    (["会計監査人", "公認会計士"], "neutral", 0.1, "auditor", "監査人変更"),
    (["訂正"], "neutral", 0.15, "correction", "訂正開示"),
    (["ＥＴＦ", "ETF"], "neutral", 0.05, "etf", "ETF定例開示"),
    (["分配金見込"], "neutral", 0.05, "etf_dist", "ETF分配金通知"),
]

# 修正系キーワードで magnitude を増幅
_MAGNITUDE_BOOSTERS = [
    (re.compile(r"(大幅|著しい|急|過去最高)"), 0.15),
    (re.compile(r"(通期|連結)"), 0.05),
]


class DisclosureAnalyzer:
    """ルールベースの TDnet 開示分析"""

    def __init__(self, model: str = ""):
        # model引数は後方互換のため残すが使わない
        pass

    def analyze(self, title: str, company_name: str = "") -> DisclosureAnalysis:
        """開示タイトルを分析して direction / magnitude / category を返す。"""
        result = DisclosureAnalysis(raw_title=title)

        for keywords, direction, magnitude, category, reasoning in _RULES:
            if any(kw in title for kw in keywords):
                result.direction = direction
                result.magnitude = magnitude
                result.category = category
                result.reasoning = reasoning
                result.analyzed = True

                # ブースター適用
                for pattern, boost in _MAGNITUDE_BOOSTERS:
                    if pattern.search(title):
                        result.magnitude = min(1.0, result.magnitude + boost)

                logger.info(
                    f"[disclosure] {title[:50]}... → "
                    f"{result.direction} mag={result.magnitude:.2f} cat={result.category}"
                )
                return result

        # どのルールにもマッチしない
        result.direction = "neutral"
        result.magnitude = 0.1
        result.category = "other"
        result.reasoning = "該当ルールなし"
        result.analyzed = True
        return result

    def analyze_batch(self, disclosures: list[dict]) -> list[DisclosureAnalysis]:
        """複数の開示を一括分析。"""
        return [
            self.analyze(d.get("title", ""), d.get("company_name", ""))
            for d in disclosures
        ]
