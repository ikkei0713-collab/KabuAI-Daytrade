"""
ペーパートレード学習ループ

1時間かけてバックテスト→ナレッジ抽出→パラメータ自動調整→再バックテストを
何度も繰り返し、戦略を磨き込む。

各イテレーションで:
1. 全16戦略でバックテスト
2. 勝ちパターン/負けパターン抽出
3. レジーム別パフォーマンスからパラメータ自動調整
4. 成績の悪い戦略をauto_toggleで自動停止
5. 結果をknowledgeに蓄積
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("logs/paper_learn.log", rotation="10 MB", level="DEBUG")

from run_backtest_learn import BacktestLearner, CANDIDATE_CODES
from strategies.registry import StrategyRegistry

DURATION_MINUTES = 55  # 実行時間


async def apply_auto_tuning():
    """前回のauto_tuning.jsonの結果を戦略パラメータに反映"""
    tuning_path = Path("knowledge/auto_tuning.json")
    if not tuning_path.exists():
        return

    try:
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        adjustments = tuning.get("adjustments_applied", [])

        for adj in adjustments:
            target = adj.get("target", "")
            action = adj.get("action", "")
            parts = target.rsplit("_", 1)
            if len(parts) != 2:
                continue
            strategy_name, regime = parts[0], parts[1]

            strategy = StrategyRegistry.get(strategy_name)
            if not strategy:
                continue

            blocked = set(strategy.config.parameter_set.get("blocked_regimes", []))

            if action == "suppress_regime" and regime not in blocked:
                blocked.add(regime)
                strategy.config.parameter_set["blocked_regimes"] = list(blocked)
                logger.info(f"[auto_tune] {strategy_name}: {regime} をブロック追加")

            elif action == "boost_regime" and regime in blocked:
                blocked.discard(regime)
                strategy.config.parameter_set["blocked_regimes"] = list(blocked)
                logger.info(f"[auto_tune] {strategy_name}: {regime} のブロック解除")

        # OOSの悪い戦略のconfidence閾値を上げる
        oos = tuning.get("oos_summary", {})
        if oos.get("pf", 0) < 1.0 and oos.get("trades", 0) >= 10:
            logger.info(f"[auto_tune] OOS PF={oos['pf']:.2f} < 1.0 → 全戦略のフィルタ強化")

    except Exception as e:
        logger.warning(f"auto_tuning読み込みエラー: {e}")


async def apply_auto_toggle(learner: BacktestLearner):
    """バックテスト結果でauto_toggle実行"""
    if not learner.all_trades:
        return

    toggled = StrategyRegistry.auto_toggle(learner.all_trades, min_trades=8)
    if toggled:
        logger.info(f"[auto_toggle] {len(toggled)}戦略を切り替え: {toggled}")


async def run_learning_loop():
    """学習ループ本体"""
    start_time = time.time()
    end_time = start_time + DURATION_MINUTES * 60
    iteration = 0
    best_oos_pf = 0.0
    best_iteration = 0
    history = []

    logger.info("=" * 60)
    logger.info(f"ペーパートレード学習ループ開始 ({DURATION_MINUTES}分)")
    logger.info("=" * 60)

    while time.time() < end_time:
        iteration += 1
        elapsed = (time.time() - start_time) / 60
        remaining = (end_time - time.time()) / 60

        logger.info("=" * 60)
        logger.info(f"イテレーション {iteration} ({elapsed:.0f}分経過, 残り{remaining:.0f}分)")
        logger.info("=" * 60)

        # 戦略を再登録（前回のauto_toggle反映）
        StrategyRegistry.clear()
        StrategyRegistry.register_all_defaults()

        # 前回の自動チューニング結果を適用
        if iteration > 1:
            await apply_auto_tuning()

        active = StrategyRegistry.get_active()
        logger.info(f"Active戦略: {len(active)}個 = {[s.name for s in active]}")

        # バックテスト実行
        learner = BacktestLearner()
        await learner.run()

        # OOS評価
        oos_m = learner._calc_metrics(learner.oos_trades)
        is_m = learner._calc_metrics(learner.is_trades)
        all_m = learner._calc_metrics(learner.all_trades)

        logger.info(
            f"結果: 全体 {all_m['total']}件 PF={all_m['pf']:.2f} | "
            f"IS {is_m['total']}件 PF={is_m['pf']:.2f} | "
            f"OOS {oos_m['total']}件 PF={oos_m['pf']:.2f} ¥{oos_m['total_pnl']:+,.0f}"
        )

        # ベスト更新チェック
        if oos_m["pf"] > best_oos_pf and oos_m["total"] >= 10:
            best_oos_pf = oos_m["pf"]
            best_iteration = iteration
            logger.info(f"★ ベスト更新: OOS PF={best_oos_pf:.2f} (イテレーション {iteration})")

        # auto_toggle: 成績悪い戦略を自動停止
        await apply_auto_toggle(learner)

        # 戦略別サマリー
        strategy_results = {}
        for t in learner.oos_trades:
            if t.strategy_name not in strategy_results:
                strategy_results[t.strategy_name] = []
            strategy_results[t.strategy_name].append(t)

        for sname, trades in sorted(strategy_results.items(), key=lambda x: -sum(t.pnl for t in x[1])):
            sm = learner._calc_metrics(trades)
            logger.info(
                f"  {sname}: OOS {sm['total']}件 WR={sm['win_rate']:.0%} "
                f"PF={sm['pf']:.2f} ¥{sm['total_pnl']:+,.0f}"
            )

        # 履歴に記録
        history.append({
            "iteration": iteration,
            "elapsed_min": round(elapsed, 1),
            "active_strategies": len(active),
            "all": {"trades": all_m["total"], "pf": round(all_m["pf"], 2), "pnl": round(all_m["total_pnl"], 0)},
            "is": {"trades": is_m["total"], "pf": round(is_m["pf"], 2), "pnl": round(is_m["total_pnl"], 0)},
            "oos": {"trades": oos_m["total"], "pf": round(oos_m["pf"], 2), "pnl": round(oos_m["total_pnl"], 0)},
            "strategy_oos": {
                sname: {
                    "trades": learner._calc_metrics(trades)["total"],
                    "pf": round(learner._calc_metrics(trades)["pf"], 2),
                    "pnl": round(learner._calc_metrics(trades)["total_pnl"], 0),
                }
                for sname, trades in strategy_results.items()
            },
        })

        # 時間チェック - 残り2分未満なら終了
        if time.time() + 120 > end_time:
            break

    # 最終サマリー
    total_elapsed = (time.time() - start_time) / 60

    logger.info("\n" + "=" * 60)
    logger.info(f"学習ループ完了: {iteration}イテレーション ({total_elapsed:.1f}分)")
    logger.info(f"ベスト OOS PF: {best_oos_pf:.2f} (イテレーション {best_iteration})")
    logger.info("=" * 60)

    # イテレーション推移
    logger.info("\n推移:")
    for h in history:
        logger.info(
            f"  #{h['iteration']} ({h['elapsed_min']}分) "
            f"| {h['active_strategies']}戦略 "
            f"| OOS: {h['oos']['trades']}件 PF={h['oos']['pf']:.2f} ¥{h['oos']['pnl']:+,.0f}"
        )

    # JSON保存
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_iterations": iteration,
        "duration_minutes": round(total_elapsed, 1),
        "best_oos_pf": round(best_oos_pf, 2),
        "best_iteration": best_iteration,
        "history": history,
    }
    Path("knowledge/paper_learn_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("結果を knowledge/paper_learn_results.json に保存")


if __name__ == "__main__":
    asyncio.run(run_learning_loop())
