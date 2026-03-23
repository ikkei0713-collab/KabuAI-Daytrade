"""
TDnet 適時開示の LLM 分析 (OpenAI gpt-4o-mini)

開示タイトルからキーワードマッチで分類していた処理を、
LLM で精度向上する。

分析内容:
- direction: positive / negative / neutral
- magnitude: 0.0-1.0 (株価インパクトの大きさ)
- category: guidance_up / guidance_down / buyback / earnings_beat / earnings_miss / ...
- reasoning: 判断理由 (1行)

コスト: ~¥10/月 (1日10件 × gpt-4o-mini)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class DisclosureAnalysis:
    """LLM による開示分析結果"""
    direction: str = "neutral"    # positive / negative / neutral
    magnitude: float = 0.0        # 0.0-1.0
    category: str = "other"       # guidance_up, buyback, earnings_beat, etc.
    reasoning: str = ""           # 判断理由
    raw_title: str = ""
    analyzed: bool = False        # LLM 分析済みか


# System prompt for disclosure analysis
_SYSTEM_PROMPT = """あなたは日本株の適時開示（TDnet）を分析する専門家です。
開示タイトルから、株価への影響を判定してください。

以下のJSON形式で回答してください。それ以外のテキストは不要です。
{
  "direction": "positive" or "negative" or "neutral",
  "magnitude": 0.0〜1.0の数値（株価インパクトの大きさ。0=影響なし、1.0=ストップ高/安レベル）,
  "category": カテゴリ文字列,
  "reasoning": "判断理由（1行）"
}

カテゴリ一覧:
- guidance_up: 業績上方修正
- guidance_down: 業績下方修正
- earnings_beat: 決算好調（予想上回り）
- earnings_miss: 決算不振（予想下回り）
- buyback: 自社株買い
- dividend_up: 増配
- dividend_down: 減配
- stock_split: 株式分割
- merger: 合併・統合
- tob: 公開買付
- alliance: 業務提携・資本提携
- new_product: 新製品・新サービス
- restructuring: リストラ・事業売却
- scandal: 不祥事・行政処分
- other: 上記以外

magnitude の目安:
- 0.1: 軽微な開示（定例報告等）
- 0.3: 小規模な修正（数%の上方修正等）
- 0.5: 中規模（10-20%の修正、自社株買い等）
- 0.7: 大規模（30%以上の修正、大型M&A等）
- 0.9: 極めて大きい（過去最高益、MBO、TOB等）
- 1.0: ストップ高/安が予想されるレベル"""


class DisclosureAnalyzer:
    """OpenAI gpt-4o-mini を使った TDnet 開示分析"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def analyze(self, title: str, company_name: str = "") -> DisclosureAnalysis:
        """開示タイトルを分析して direction / magnitude / category を返す。

        API エラー時はフォールバック（キーワードマッチ）で返す。
        """
        result = DisclosureAnalysis(raw_title=title)

        if not os.getenv("OPENAI_API_KEY"):
            logger.debug("[disclosure_analyzer] OPENAI_API_KEY not set, using fallback")
            return self._fallback(title, result)

        user_msg = f"開示タイトル: {title}"
        if company_name:
            user_msg += f"\n企業名: {company_name}"

        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=200,
                temperature=0.0,
            )
            content = resp.choices[0].message.content.strip()

            # JSON パース
            # ``` で囲まれていたら除去
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            data = json.loads(content)
            result.direction = data.get("direction", "neutral")
            result.magnitude = max(0.0, min(1.0, float(data.get("magnitude", 0.0))))
            result.category = data.get("category", "other")
            result.reasoning = data.get("reasoning", "")
            result.analyzed = True

            logger.info(
                f"[disclosure_analyzer] {title[:40]}... → "
                f"{result.direction} mag={result.magnitude:.2f} cat={result.category}"
            )

        except json.JSONDecodeError as e:
            logger.warning(f"[disclosure_analyzer] JSON parse error: {e}, using fallback")
            return self._fallback(title, result)
        except Exception as e:
            logger.warning(f"[disclosure_analyzer] API error: {e}, using fallback")
            return self._fallback(title, result)

        return result

    def analyze_batch(self, disclosures: list[dict]) -> list[DisclosureAnalysis]:
        """複数の開示を一括分析。

        Args:
            disclosures: [{"title": "...", "company_name": "..."}, ...]
        """
        results = []
        for d in disclosures:
            result = self.analyze(d.get("title", ""), d.get("company_name", ""))
            results.append(result)
        return results

    @staticmethod
    def _fallback(title: str, result: DisclosureAnalysis) -> DisclosureAnalysis:
        """LLM が使えない場合のキーワードマッチフォールバック"""
        title_lower = title.lower()

        if any(kw in title for kw in ["上方修正", "増額"]):
            result.direction = "positive"
            result.magnitude = 0.5
            result.category = "guidance_up"
        elif any(kw in title for kw in ["下方修正", "減額"]):
            result.direction = "negative"
            result.magnitude = 0.5
            result.category = "guidance_down"
        elif any(kw in title for kw in ["自己株式", "自社株買"]):
            result.direction = "positive"
            result.magnitude = 0.4
            result.category = "buyback"
        elif any(kw in title for kw in ["増配", "配当増"]):
            result.direction = "positive"
            result.magnitude = 0.3
            result.category = "dividend_up"
        elif any(kw in title for kw in ["減配", "無配"]):
            result.direction = "negative"
            result.magnitude = 0.4
            result.category = "dividend_down"
        elif any(kw in title for kw in ["株式分割"]):
            result.direction = "positive"
            result.magnitude = 0.3
            result.category = "stock_split"
        elif any(kw in title for kw in ["MBO", "公開買付"]):
            result.direction = "positive"
            result.magnitude = 0.8
            result.category = "tob"
        elif any(kw in title for kw in ["決算短信"]):
            result.direction = "neutral"
            result.magnitude = 0.3
            result.category = "earnings_beat"  # 方向不明
        else:
            result.direction = "neutral"
            result.magnitude = 0.1
            result.category = "other"

        result.reasoning = "keyword fallback"
        return result
