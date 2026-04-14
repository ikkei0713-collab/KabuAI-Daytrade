"""
朝の準備 — 全スキャンを一括実行

寄り前に実行して全データを収集:
1. 夜間マーケットスキャン (海外指数/先物/為替/VIX/セクター/半導体)
2. TDnet適時開示スキャン + LLM分析
3. セクターバイアス (米国→日本)
4. マーケットレジーム判定
5. 決算データ更新

結果は全て knowledge/ に保存され、実取引ボットが起動時に自動読み込み。
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")

PYTHON = str(Path(__file__).parent / ".venv/bin/python3.13")


async def run_script(name: str, script: str):
    """サブプロセスでスクリプトを実行"""
    logger.info(f"[{name}] 開始...")
    start = time.time()
    try:
        proc = await asyncio.create_subprocess_exec(
            PYTHON, script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={"KABUAI_ALLOW_LIVE_TRADING": "false", "PATH": "/usr/bin:/bin:/usr/local/bin"},
            cwd=str(Path(__file__).parent),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        elapsed = time.time() - start

        # 最後の数行だけ表示
        lines = stdout.decode("utf-8", errors="replace").strip().split("\n")
        for line in lines[-5:]:
            logger.info(f"  [{name}] {line}")
        logger.info(f"[{name}] 完了 ({elapsed:.0f}秒)")
    except asyncio.TimeoutError:
        logger.warning(f"[{name}] タイムアウト (300秒)")
    except Exception as e:
        logger.error(f"[{name}] エラー: {e}")


async def main():
    logger.info("=" * 60)
    logger.info("朝の準備開始 — 全データ収集")
    logger.info("=" * 60)

    # Phase 1: 夜間データ + 寄り前スキャンを並列実行
    await asyncio.gather(
        run_script("夜間マーケット", "run_overnight_scan.py"),
        run_script("寄り前スキャン", "run_premarket_all.py"),
        run_script("追加データ", "run_extra_data.py"),
    )

    logger.info("=" * 60)
    logger.info("朝の準備完了 — 全データ knowledge/ に保存済み")
    logger.info("=" * 60)

    # サマリー表示
    import json
    for fname, label in [
        ("knowledge/overnight_scan.json", "夜間"),
        ("knowledge/premarket_scan.json", "寄り前"),
        ("knowledge/extra_data.json", "追加"),
    ]:
        p = Path(fname)
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            logger.info(f"  {label}: {p.name} ({p.stat().st_size:,} bytes)")
        else:
            logger.warning(f"  {label}: 未生成")


if __name__ == "__main__":
    asyncio.run(main())
