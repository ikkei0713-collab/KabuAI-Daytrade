"""
銘柄コード → 銘柄名 マッピング

J-Quantsの上場銘柄マスタからキャッシュ。
表示時に「トヨタ自動車(7203)」形式に変換する。
"""

import json
from pathlib import Path
from loguru import logger

CACHE_PATH = Path(__file__).parent.parent / "data" / "cache" / "ticker_names.json"


def _load_cache() -> dict[str, str]:
    """キャッシュから銘柄名マップを読み込む"""
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_cache(mapping: dict[str, str]) -> None:
    """キャッシュに保存"""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )


# モジュールレベルのキャッシュ
_TICKER_MAP: dict[str, str] = _load_cache()


def update_from_jquants(master_list: list[dict]) -> None:
    """J-Quantsのマスタデータから銘柄名マップを更新"""
    global _TICKER_MAP
    for item in master_list:
        code = item.get("Code", "")
        name = item.get("CoName", "") or item.get("CompanyName", "")
        if code and name:
            _TICKER_MAP[code] = name.strip()
    _save_cache(_TICKER_MAP)
    logger.info(f"銘柄名マップ更新: {len(_TICKER_MAP)}件")


def get_name(code: str) -> str:
    """コードから銘柄名を返す（なければコードのまま）"""
    return _TICKER_MAP.get(code, "")


def format_ticker(code: str) -> str:
    """
    銘柄コードを「銘柄名(コード)」形式に変換

    例: "72030" → "トヨタ自動車(7203)"
    """
    # J-Quantsのコードは5桁（末尾にチェックディジット0）、表示は4桁
    display_code = code[:-1] if len(code) == 5 and code[-1] == "0" else code
    name = _TICKER_MAP.get(code, "")
    if name:
        return f"{name}({display_code})"
    return display_code


def format_ticker_column(codes) -> list[str]:
    """DataFrameのコード列を一括変換"""
    return [format_ticker(c) for c in codes]
