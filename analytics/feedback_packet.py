"""
Claude Code 向け改善フィードバックパッケージ生成

バックテスト / 学習ループ完了時に呼び出し、
JSON (機械可読) + Markdown (人間/LLM可読) + plots (可視化) を生成する。

論文準拠フィードバック設計 (2026-03-20):
  Prompt1 (基本テキスト): 全体/IS/OOS metrics + strategy breakdown
  Prompt2 (基本+追加テキスト): + feature statistics + IC/ICIR + net exposure
  Prompt3 (テキスト+プロット): + equity curve, drawdown, cumulative IC, net exposure plot

出力:
  knowledge/feedback_packet.json   — 全メトリクス + 異常情報 + 推奨ヒント
  knowledge/feedback_summary.md    — Claude Code へコピペしやすいサマリ
  knowledge/plots/*.png            — 主要チャート群
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

KNOWLEDGE_DIR = Path("knowledge")
PLOTS_DIR = KNOWLEDGE_DIR / "plots"
PACKET_PATH = KNOWLEDGE_DIR / "feedback_packet.json"
SUMMARY_PATH = KNOWLEDGE_DIR / "feedback_summary.md"

# ---------------------------------------------------------------------------
# Proxy / pseudo features that are NOT real intraday data
# ---------------------------------------------------------------------------

PROXY_FEATURES = {
    "time_below_vwap",       # 固定値 30
    "volume_at_reclaim",     # vol * 1.2
    "spread_percentile",     # 固定値 0.5
    "volume_first_5min",     # vol * 0.05
    "tick_direction",        # 推定値
    "bid_ask_ratio",         # 推定値
    "spread",                # ATR * 0.05
    "depth_imbalance",       # 推定値
    "volume_building",       # vol / vol_avg
    "price_compression",     # BB percentile
    "opening_range_high",    # daily high (not real OR)
    "opening_range_low",     # daily low (not real OR)
}


# ---------------------------------------------------------------------------
# Helper: safe metrics calc
# ---------------------------------------------------------------------------

def _calc(trades: list) -> dict:
    """Calculate standard metrics from a trade list.

    Includes fit3 (Arai 2013): -Loss + 0.01*Profit
    損失回避型適合度関数。損失を強く罰し過学習を抑制する。
    """
    if not trades:
        return {"total": 0, "wins": 0, "win_rate": 0, "pf": 0,
                "pnl": 0, "avg": 0, "max_win": 0, "max_loss": 0,
                "sharpe": 0, "fit3": 0}
    total = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    pnl = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in trades if t.pnl > 0)
    gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
    pnl_arr = np.array([t.pnl for t in trades])
    sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(252)) if pnl_arr.std() > 0 else 0
    # 論文 (Arai 2013) fit3: -Loss + 0.01 * Profit
    fit3 = -gl + 0.01 * gp
    return {
        "total": total,
        "wins": wins,
        "win_rate": round(wins / total, 3) if total else 0,
        "pf": round(gp / gl, 2) if gl > 0 else 0,
        "pnl": round(pnl, 0),
        "avg": round(pnl / total, 0) if total else 0,
        "max_win": round(max(t.pnl for t in trades), 0) if trades else 0,
        "max_loss": round(min(t.pnl for t in trades), 0) if trades else 0,
        "sharpe": round(sharpe, 2),
        "fit3": round(fit3, 0),
    }


def _max_drawdown(trades: list) -> tuple[float, float]:
    """Calculate max drawdown from trade PnLs. Returns (absolute, pct of peak)."""
    if not trades:
        return 0.0, 0.0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    for t in trades:
        equity += t.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
        if peak > 0:
            dd_pct = dd / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    return round(max_dd, 0), round(max_dd_pct, 3)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class FeedbackPacketGenerator:
    """Generate feedback_packet.json, feedback_summary.md, and plots."""

    def __init__(
        self,
        all_trades: list,
        is_trades: list,
        oos_trades: list,
        strategy_regime_trades: dict[tuple[str, str], list] | None = None,
    ):
        self.all_trades = all_trades
        self.is_trades = is_trades
        self.oos_trades = oos_trades
        self.strategy_regime_trades = strategy_regime_trades or {}

    def generate(self) -> dict:
        """Generate all outputs. Returns the packet dict."""
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        packet = self._build_packet()

        # Save JSON
        PACKET_PATH.write_text(
            json.dumps(packet, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info(f"[feedback] feedback_packet.json 生成完了 ({len(self.all_trades)}件)")

        # Save Markdown
        md = self._build_markdown(packet)
        SUMMARY_PATH.write_text(md, encoding="utf-8")
        logger.info("[feedback] feedback_summary.md 生成完了")

        # Generate plots
        self._generate_plots()
        logger.info("[feedback] plots 生成完了")

        return packet

    # ------------------------------------------------------------------
    # Packet builder
    # ------------------------------------------------------------------

    def _build_packet(self) -> dict:
        m_all = _calc(self.all_trades)
        m_is = _calc(self.is_trades)
        m_oos = _calc(self.oos_trades)
        dd_abs, dd_pct = _max_drawdown(self.all_trades)

        # Strategy breakdown
        strat_map: dict[str, list] = {}
        for t in self.all_trades:
            strat_map.setdefault(t.strategy_name, []).append(t)

        strategy_metrics = []
        for sname in sorted(strat_map):
            s_all = strat_map[sname]
            s_is = [t for t in self.is_trades if t.strategy_name == sname]
            s_oos = [t for t in self.oos_trades if t.strategy_name == sname]
            sm_is = _calc(s_is)
            sm_oos = _calc(s_oos)
            strategy_metrics.append({
                "name": sname,
                "all": _calc(s_all),
                "is": sm_is,
                "oos": sm_oos,
                "avg_holding_min": round(np.mean([t.holding_minutes for t in s_all]), 0) if s_all else 0,
                "is_oos_pf_gap": round(sm_is["pf"] - sm_oos["pf"], 2) if sm_oos["total"] >= 3 else None,
            })

        # Regime breakdown
        regime_metrics = []
        for (sname, regime), trades in sorted(self.strategy_regime_trades.items()):
            if len(trades) < 2:
                continue
            regime_metrics.append({
                "strategy": sname,
                "regime": regime,
                **_calc(trades),
            })

        # Symbol breakdown
        sym_map: dict[str, list] = {}
        for t in self.all_trades:
            sym_map.setdefault(t.ticker, []).append(t)
        symbol_metrics = [
            {"ticker": ticker, **_calc(trades)}
            for ticker, trades in sorted(sym_map.items(), key=lambda x: sum(t.pnl for t in x[1]), reverse=True)[:20]
        ]

        # Feature diagnostics
        feature_diag = self._feature_diagnostics()

        # Anomaly summary
        anomaly = self._anomaly_summary()

        # Recommendation hints
        hints = self._recommendation_hints(strategy_metrics, m_all, m_is, m_oos)

        # Confidence vs PnL data
        conf_pnl = []
        for t in self.all_trades:
            conf = t.features_at_entry.get("confidence", None)
            if conf is None:
                # Try to extract from entry_reason or signal snapshot
                pass
            conf_pnl.append({
                "confidence": conf,
                "pnl": t.pnl,
                "strategy": t.strategy_name,
            })

        # --- Prompt2 追加情報: Feature statistics ---
        feature_stats = self._feature_statistics()

        # --- Prompt2 追加情報: IC/ICIR (近似) ---
        ic_icir = self._ic_icir()

        # --- Prompt2 追加情報: Net exposure ---
        net_exposure = self._net_exposure()

        # Convergence feature analysis (v3.3)
        convergence_analysis = self._convergence_analysis()

        # Plot paths
        plot_paths = {
            "equity_curve": str(PLOTS_DIR / "equity_curve.png"),
            "drawdown_curve": str(PLOTS_DIR / "drawdown_curve.png"),
            "strategy_oos": str(PLOTS_DIR / "strategy_oos.png"),
            "regime_heatmap": str(PLOTS_DIR / "regime_heatmap.png"),
            "confidence_vs_pnl": str(PLOTS_DIR / "confidence_vs_pnl.png"),
            "feature_win_loss": str(PLOTS_DIR / "feature_win_loss_compare.png"),
            "cumulative_ic": str(PLOTS_DIR / "cumulative_ic.png"),
            "net_exposure": str(PLOTS_DIR / "net_exposure.png"),
        }

        return {
            "generated_at": datetime.now().isoformat(),
            "data_quality": feature_diag,
            "overall_metrics": {**m_all, "max_drawdown": dd_abs, "max_drawdown_pct": dd_pct},
            "in_sample_metrics": m_is,
            "out_of_sample_metrics": m_oos,
            "strategy_metrics": strategy_metrics,
            "regime_metrics": regime_metrics,
            "symbol_metrics": symbol_metrics,
            "feature_diagnostics": feature_diag,
            "feature_statistics": feature_stats,
            "ic_icir": ic_icir,
            "net_exposure": net_exposure,
            "convergence_analysis": convergence_analysis,
            "anomaly_summary": anomaly,
            "recommendation_hints": hints,
            "plot_paths": plot_paths,
        }

    # ------------------------------------------------------------------
    # Feature diagnostics
    # ------------------------------------------------------------------

    def _feature_diagnostics(self) -> dict:
        """Analyze proxy feature usage and data quality."""
        if not self.all_trades:
            return {"proxy_feature_rate": 0, "proxy_features_used": []}

        proxy_count = 0
        total_features = 0
        proxy_used = set()

        for t in self.all_trades:
            feats = t.features_at_entry or {}
            for key in feats:
                total_features += 1
                if key in PROXY_FEATURES:
                    proxy_count += 1
                    proxy_used.add(key)

        rate = proxy_count / total_features if total_features > 0 else 0

        return {
            "total_feature_observations": total_features,
            "proxy_feature_observations": proxy_count,
            "proxy_feature_rate": round(rate, 3),
            "proxy_features_used": sorted(proxy_used),
            "warning": "intraday proxy依存 > 30%" if rate > 0.3 else None,
        }

    # ------------------------------------------------------------------
    # Feature statistics (論文 Prompt2 基本情報)
    # ------------------------------------------------------------------

    def _feature_statistics(self) -> dict:
        """Per-feature descriptive statistics across all trades.

        論文の「基本情報」に相当: count, mean, std, min,
        1%, 5%, 50%, 95%, 99%, max, skew, kurtosis, missing_ratio, zero_ratio.
        """
        if not self.all_trades:
            return {}

        # Collect numeric features from all trades
        feature_vals: dict[str, list[float]] = {}
        total_trades = len(self.all_trades)
        missing_counts: dict[str, int] = {}

        # Key features to track (avoid internal/metadata keys)
        target_keys = {
            "atr", "vwap", "vwap_distance", "volume_ratio", "gap_pct",
            "rsi_14", "rsi_5", "ema_9", "ema_21", "sma_5", "sma_20",
            "macd", "macd_histogram", "bb_pct", "trend_strength",
            "volume_trend", "intraday_range", "roc_5", "roc_10",
            "time_below_vwap", "volume_at_reclaim", "selector_score",
            # Convergence features (v3.3)
            "ma_spread_pct", "ma_convergence_score", "ma_convergence_trend",
            "range_compression_score", "volatility_compression_score",
            "extension_from_ma_score", "price_ma_cluster_score",
            "pullback_to_ma_score",
        }

        for t in self.all_trades:
            feats = t.features_at_entry or {}
            for key in target_keys:
                val = feats.get(key)
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    missing_counts[key] = missing_counts.get(key, 0) + 1
                else:
                    try:
                        feature_vals.setdefault(key, []).append(float(val))
                    except (ValueError, TypeError):
                        missing_counts[key] = missing_counts.get(key, 0) + 1

        stats = {}
        from scipy import stats as sp_stats

        for key in sorted(target_keys):
            vals = feature_vals.get(key, [])
            n = len(vals)
            missing = missing_counts.get(key, 0) + (total_trades - n - missing_counts.get(key, 0))
            if n < 2:
                stats[key] = {"count": n, "missing_ratio": 1.0 if n == 0 else round(missing / total_trades, 3)}
                continue

            arr = np.array(vals)
            is_proxy = key in PROXY_FEATURES
            stats[key] = {
                "count": n,
                "mean": round(float(arr.mean()), 4),
                "std": round(float(arr.std()), 4),
                "min": round(float(arr.min()), 4),
                "p1": round(float(np.percentile(arr, 1)), 4),
                "p5": round(float(np.percentile(arr, 5)), 4),
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p95": round(float(np.percentile(arr, 95)), 4),
                "p99": round(float(np.percentile(arr, 99)), 4),
                "max": round(float(arr.max()), 4),
                "skew": round(float(sp_stats.skew(arr)), 4) if n >= 3 and arr.std() > 1e-10 else None,
                "kurtosis": round(float(sp_stats.kurtosis(arr)), 4) if n >= 4 and arr.std() > 1e-10 else None,
                "missing_ratio": round(missing / total_trades, 3) if total_trades > 0 else 0,
                "zero_ratio": round(float((arr == 0).sum() / n), 3),
                "is_proxy": is_proxy,
            }

        return stats

    # ------------------------------------------------------------------
    # IC / ICIR (論文 Prompt2 追加情報)
    # ------------------------------------------------------------------

    def _ic_icir(self) -> dict:
        """Approximate Information Coefficient (IC) and IC Information Ratio.

        IC ≈ Spearman(selector_score, realized_return) per trade.
        これは厳密な因子IC ではなく近似値。
        """
        scores = []
        returns = []
        for t in self.all_trades:
            feats = t.features_at_entry or {}
            score = feats.get("selector_score")
            if score is not None and t.entry_price > 0:
                scores.append(float(score))
                returns.append(t.pnl_pct if hasattr(t, 'pnl_pct') and t.pnl_pct else
                               t.pnl / (t.entry_price * 100) * 100)

        if len(scores) < 5:
            return {"ic": None, "icir": None, "note": "サンプル不足 (5件未満)"}

        from scipy.stats import spearmanr
        ic, p_value = spearmanr(scores, returns)

        # Rolling IC for ICIR (use rolling window of 5)
        rolling_ics = []
        window = min(5, len(scores) - 1)
        for i in range(window, len(scores)):
            chunk_s = scores[i - window:i]
            chunk_r = returns[i - window:i]
            if len(set(chunk_s)) > 1 and len(set(chunk_r)) > 1:
                r, _ = spearmanr(chunk_s, chunk_r)
                if not np.isnan(r):
                    rolling_ics.append(r)

        icir = None
        if rolling_ics and np.std(rolling_ics) > 0:
            icir = round(float(np.mean(rolling_ics) / np.std(rolling_ics)), 3)

        return {
            "ic": round(float(ic), 4) if not np.isnan(ic) else None,
            "ic_p_value": round(float(p_value), 4) if not np.isnan(p_value) else None,
            "icir": icir,
            "n_samples": len(scores),
            "rolling_ics": [round(float(x), 4) for x in rolling_ics],
            "note": "近似値: Spearman(selector_score, realized_return)",
        }

    # ------------------------------------------------------------------
    # Net exposure (論文 Prompt2 追加情報)
    # ------------------------------------------------------------------

    def _net_exposure(self) -> dict:
        """Calculate net exposure breakdown by direction, strategy, sector.

        Net exposure ≈ 方向/戦略/セクターの偏り度合い。
        厳密なドル建てexposureではなく、取引件数ベースの近似値。
        """
        if not self.all_trades:
            return {}

        # Direction exposure
        long_count = sum(1 for t in self.all_trades if t.direction == "long")
        short_count = sum(1 for t in self.all_trades if t.direction == "short")
        total = len(self.all_trades)
        direction_bias = (long_count - short_count) / total if total > 0 else 0

        # Strategy concentration
        strat_counts: dict[str, int] = {}
        for t in self.all_trades:
            strat_counts[t.strategy_name] = strat_counts.get(t.strategy_name, 0) + 1
        max_strat = max(strat_counts.values()) if strat_counts else 0
        concentration = max_strat / total if total > 0 else 0

        # Ticker concentration (top 3)
        ticker_counts: dict[str, int] = {}
        for t in self.all_trades:
            ticker_counts[t.ticker] = ticker_counts.get(t.ticker, 0) + 1
        top_tickers = sorted(ticker_counts.items(), key=lambda x: -x[1])[:3]

        return {
            "long_count": long_count,
            "short_count": short_count,
            "direction_bias": round(direction_bias, 3),
            "direction_bias_label": "long偏重" if direction_bias > 0.3 else
                                    "short偏重" if direction_bias < -0.3 else "中立",
            "strategy_concentration": round(concentration, 3),
            "strategy_counts": strat_counts,
            "top_tickers": [{"ticker": t, "count": c} for t, c in top_tickers],
            "note": "取引件数ベースの近似値 (金額ベースではない)",
        }

    # ------------------------------------------------------------------
    # Convergence analysis (v3.3)
    # ------------------------------------------------------------------

    def _convergence_analysis(self) -> dict:
        """Analyze convergence feature effectiveness across trades."""
        if not self.all_trades:
            return {}

        conv_keys = [
            "ma_spread_pct", "ma_convergence_score", "range_compression_score",
            "volatility_compression_score", "extension_from_ma_score",
            "post_cross_expansion_flag", "post_cross_consolidation_flag",
            "squeeze_breakout_ready",
        ]

        # Collect distributions
        distributions: dict[str, list[float]] = {k: [] for k in conv_keys[:5]}
        win_vals: dict[str, list[float]] = {k: [] for k in conv_keys[:5]}
        loss_vals: dict[str, list[float]] = {k: [] for k in conv_keys[:5]}

        expansion_trades = []
        convergence_trades = []
        post_cross_exp_trades = []
        post_cross_con_trades = []
        squeeze_trades = []

        for t in self.all_trades:
            feats = t.features_at_entry or {}
            is_win = t.pnl > 0

            for k in conv_keys[:5]:
                val = feats.get(k)
                if val is not None and isinstance(val, (int, float)) and not np.isnan(val):
                    distributions[k].append(float(val))
                    if is_win:
                        win_vals[k].append(float(val))
                    else:
                        loss_vals[k].append(float(val))

            # Classify trades
            ma_conv = feats.get("ma_convergence_score")
            if ma_conv is not None:
                if ma_conv >= 0.55:
                    convergence_trades.append(t)
                else:
                    expansion_trades.append(t)

            if feats.get("post_cross_expansion_flag"):
                post_cross_exp_trades.append(t)
            if feats.get("post_cross_consolidation_flag"):
                post_cross_con_trades.append(t)
            if feats.get("squeeze_breakout_ready"):
                squeeze_trades.append(t)

        # Summary stats per feature
        feature_summary = {}
        for k in conv_keys[:5]:
            vals = distributions[k]
            if vals:
                feature_summary[k] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "p25": round(float(np.percentile(vals, 25)), 4),
                    "p50": round(float(np.percentile(vals, 50)), 4),
                    "p75": round(float(np.percentile(vals, 75)), 4),
                    "win_mean": round(float(np.mean(win_vals[k])), 4) if win_vals[k] else None,
                    "loss_mean": round(float(np.mean(loss_vals[k])), 4) if loss_vals[k] else None,
                }

        def _group_metrics(trades):
            if not trades:
                return {"count": 0}
            wins = sum(1 for t in trades if t.pnl > 0)
            pnl = sum(t.pnl for t in trades)
            gp = sum(t.pnl for t in trades if t.pnl > 0)
            gl = abs(sum(t.pnl for t in trades if t.pnl <= 0))
            return {
                "count": len(trades),
                "win_rate": round(wins / len(trades), 3) if trades else 0,
                "pf": round(gp / gl, 2) if gl > 0 else 0,
                "avg_pnl": round(pnl / len(trades), 0) if trades else 0,
            }

        return {
            "convergence_feature_summary": feature_summary,
            "expansion_entry_metrics": _group_metrics(expansion_trades),
            "convergence_entry_metrics": _group_metrics(convergence_trades),
            "post_cross_expansion_metrics": _group_metrics(post_cross_exp_trades),
            "post_cross_consolidation_metrics": _group_metrics(post_cross_con_trades),
            "squeeze_breakout_metrics": _group_metrics(squeeze_trades),
        }

    # ------------------------------------------------------------------
    # Anomaly summary
    # ------------------------------------------------------------------

    def _anomaly_summary(self) -> dict:
        """Summarize anomalies and data issues."""
        exit_reasons: dict[str, int] = {}
        stop_loss_count = 0
        high_conf_losses = 0

        for t in self.all_trades:
            reason = t.exit_reason or "unknown"
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            if "ストップロス" in reason:
                stop_loss_count += 1

        # Consecutive losses
        max_consecutive_losses = 0
        current_streak = 0
        for t in self.all_trades:
            if t.pnl <= 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        return {
            "exit_reason_breakdown": dict(sorted(exit_reasons.items(), key=lambda x: -x[1])),
            "stop_loss_rate": round(stop_loss_count / len(self.all_trades), 3) if self.all_trades else 0,
            "max_consecutive_losses": max_consecutive_losses,
            "high_confidence_losses": high_conf_losses,
        }

    # ------------------------------------------------------------------
    # Recommendation hints
    # ------------------------------------------------------------------

    def _recommendation_hints(
        self, strategy_metrics: list, m_all: dict, m_is: dict, m_oos: dict,
    ) -> dict:
        weak = []
        strong = []
        overfitting = []

        for sm in strategy_metrics:
            sname = sm["name"]
            oos = sm["oos"]
            is_m = sm["is"]

            if oos["total"] >= 3 and oos["pf"] < 0.90:
                weak.append({"strategy": sname, "oos_pf": oos["pf"], "oos_trades": oos["total"]})
            if oos["total"] >= 5 and oos["pf"] > 1.20:
                strong.append({"strategy": sname, "oos_pf": oos["pf"], "oos_trades": oos["total"]})
            if sm["is_oos_pf_gap"] is not None and sm["is_oos_pf_gap"] > 0.40:
                overfitting.append({
                    "strategy": sname,
                    "is_pf": is_m["pf"],
                    "oos_pf": oos["pf"],
                    "gap": sm["is_oos_pf_gap"],
                })

        # Convergence filter effectiveness (v3.3)
        conv = self._convergence_analysis()
        conv_exp = conv.get("expansion_entry_metrics", {})
        conv_con = conv.get("convergence_entry_metrics", {})
        convergence_helped = False
        expansion_loss_rate = 0
        if conv_con.get("count", 0) >= 3 and conv_exp.get("count", 0) >= 3:
            if conv_con.get("pf", 0) > conv_exp.get("pf", 0):
                convergence_helped = True
            expansion_loss_rate = round(1.0 - conv_exp.get("win_rate", 0), 3)

        return {
            "weakest_strategies": weak,
            "strongest_strategies": strong,
            "overfitting_signals": overfitting,
            "convergence_filter_helped": convergence_helped,
            "expansion_chasing_loss_rate": expansion_loss_rate,
            "data_quality_warning": self._feature_diagnostics().get("warning"),
            "system_notes": [
                "本システムはintraday proxy (日足+擬似特徴量) で動作",
                "ORB/spread/orderbook系の評価信頼度は限定的",
                "OOS PF > IS PF の場合はサンプルサイズ不足の可能性",
                "収束系特徴量は日足から直接計算 (proxy ではない) だが intraday 精度は限定的",
            ],
        }

    # ------------------------------------------------------------------
    # Markdown builder
    # ------------------------------------------------------------------

    def _build_markdown(self, packet: dict) -> str:
        m = packet["overall_metrics"]
        m_is = packet["in_sample_metrics"]
        m_oos = packet["out_of_sample_metrics"]
        hints = packet["recommendation_hints"]
        anomaly = packet["anomaly_summary"]
        dq = packet["data_quality"]

        lines = [
            "# KabuAI フィードバックサマリー",
            f"生成日時: {packet['generated_at'][:19]}",
            "",
            "## 全体成績",
            f"| 指標 | 全体 | IS | OOS |",
            f"|------|------|----|----|",
            f"| トレード数 | {m['total']} | {m_is['total']} | {m_oos['total']} |",
            f"| 勝率 | {m['win_rate']:.1%} | {m_is['win_rate']:.1%} | {m_oos['win_rate']:.1%} |",
            f"| PF | {m['pf']:.2f} | {m_is['pf']:.2f} | {m_oos['pf']:.2f} |",
            f"| 損益 | {m['pnl']:+,.0f} | {m_is['pnl']:+,.0f} | {m_oos['pnl']:+,.0f} |",
            f"| MaxDD | {m['max_drawdown']:,.0f} ({m['max_drawdown_pct']:.1%}) | - | - |",
            "",
            "## 戦略別 OOS 成績",
        ]

        for sm in packet["strategy_metrics"]:
            oos = sm["oos"]
            flag = ""
            if sm["is_oos_pf_gap"] is not None and sm["is_oos_pf_gap"] > 0.40:
                flag = " **[過学習疑い]**"
            lines.append(
                f"- **{sm['name']}**: OOS {oos['total']}件 WR={oos['win_rate']:.0%} "
                f"PF={oos['pf']:.2f} {oos['pnl']:+,.0f}円{flag}"
            )

        lines += [
            "",
            "## 問題点トップ5",
        ]

        problems = []
        if hints["overfitting_signals"]:
            for o in hints["overfitting_signals"]:
                problems.append(f"過学習: {o['strategy']} IS PF={o['is_pf']:.2f} → OOS PF={o['oos_pf']:.2f} (gap={o['gap']:.2f})")
        if hints["weakest_strategies"]:
            for w in hints["weakest_strategies"]:
                problems.append(f"弱い戦略: {w['strategy']} OOS PF={w['oos_pf']:.2f}")
        if anomaly["stop_loss_rate"] > 0.4:
            problems.append(f"ストップロス率が高い: {anomaly['stop_loss_rate']:.0%}")
        if anomaly["max_consecutive_losses"] >= 4:
            problems.append(f"最大連敗: {anomaly['max_consecutive_losses']}連敗")
        if dq.get("warning"):
            problems.append(f"データ品質: {dq['warning']}")
        if not problems:
            problems.append("重大な問題は検出されていません")

        for i, p in enumerate(problems[:5], 1):
            lines.append(f"{i}. {p}")

        lines += [
            "",
            "## 強い戦略",
        ]
        if hints["strongest_strategies"]:
            for s in hints["strongest_strategies"]:
                lines.append(f"- {s['strategy']}: OOS PF={s['oos_pf']:.2f} ({s['oos_trades']}件)")
        else:
            lines.append("- OOS PF > 1.20 かつ 5件以上の戦略なし")

        lines += [
            "",
            "## データ品質",
            f"- 擬似特徴量使用率: {dq['proxy_feature_rate']:.1%}",
            f"- 使用中の擬似特徴量: {', '.join(dq['proxy_features_used'][:5])}",
            "",
            "## 注意事項",
        ]
        for note in hints["system_notes"]:
            lines.append(f"- {note}")

        # IC/ICIR
        ic_data = packet.get("ic_icir", {})
        if ic_data.get("ic") is not None:
            lines += [
                "",
                "## IC/ICIR (近似)",
                f"- IC (Spearman): {ic_data['ic']:.4f} (p={ic_data.get('ic_p_value', '?')})",
                f"- ICIR: {ic_data.get('icir', 'N/A')}",
                f"- サンプル数: {ic_data.get('n_samples', 0)}",
                f"- 注: {ic_data.get('note', '')}",
            ]

        # Net exposure
        ne = packet.get("net_exposure", {})
        if ne:
            lines += [
                "",
                "## Net Exposure",
                f"- 方向: Long {ne.get('long_count', 0)} / Short {ne.get('short_count', 0)} "
                f"(バイアス: {ne.get('direction_bias_label', '?')})",
                f"- 戦略集中度: {ne.get('strategy_concentration', 0):.0%}",
            ]

        # Convergence analysis (v3.3)
        conv = packet.get("convergence_analysis", {})
        if conv:
            lines += [
                "",
                "## 収束フィルタ分析 (v3.3)",
            ]
            conv_m = conv.get("convergence_entry_metrics", {})
            exp_m = conv.get("expansion_entry_metrics", {})
            if conv_m.get("count", 0) > 0:
                lines.append(
                    f"- 収束後エントリー: {conv_m['count']}件 WR={conv_m.get('win_rate', 0):.0%} "
                    f"PF={conv_m.get('pf', 0):.2f} avg={conv_m.get('avg_pnl', 0):+,.0f}円"
                )
            if exp_m.get("count", 0) > 0:
                lines.append(
                    f"- 拡散エントリー: {exp_m['count']}件 WR={exp_m.get('win_rate', 0):.0%} "
                    f"PF={exp_m.get('pf', 0):.2f} avg={exp_m.get('avg_pnl', 0):+,.0f}円"
                )
            pce = conv.get("post_cross_expansion_metrics", {})
            pcc = conv.get("post_cross_consolidation_metrics", {})
            if pce.get("count", 0) > 0:
                lines.append(f"- GC/DC直後拡散: {pce['count']}件 WR={pce.get('win_rate', 0):.0%}")
            if pcc.get("count", 0) > 0:
                lines.append(f"- GC/DC後収束: {pcc['count']}件 WR={pcc.get('win_rate', 0):.0%}")
            if hints.get("convergence_filter_helped"):
                lines.append("- **収束フィルタは有効** (収束後 PF > 拡散 PF)")
            if hints.get("expansion_chasing_loss_rate", 0) > 0.5:
                lines.append(f"- 拡散飛び乗り損失率: {hints['expansion_chasing_loss_rate']:.0%}")

        lines += [
            "",
            "## 次に改善すべき観点",
            "1. OOS PF が低い戦略のパラメータ見直し or 停止",
            "2. 過学習が疑われる戦略の条件厳格化",
            "3. 擬似特徴量依存の低減 (分足データ導入検討)",
            "4. ストップロス幅の再検討",
            "5. 銘柄選定フィルタの見直し",
            "6. 収束フィルタ閾値の調整 (MAX_MA_SPREAD_PCT, MIN_CONVERGENCE_SCORE 等)",
        ]

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _generate_plots(self):
        """Generate all feedback plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.rcParams["font.size"] = 10
        except ImportError:
            logger.warning("[feedback] matplotlib未インストール: plot生成スキップ")
            return

        self._plot_equity_curve(plt)
        self._plot_drawdown(plt)
        self._plot_strategy_oos(plt)
        self._plot_regime_heatmap(plt)
        self._plot_confidence_vs_pnl(plt)
        self._plot_feature_compare(plt)
        self._plot_cumulative_ic(plt)
        self._plot_net_exposure(plt)

    def _plot_equity_curve(self, plt):
        if not self.all_trades:
            return
        cumulative = np.cumsum([t.pnl for t in self.all_trades])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cumulative, linewidth=1.5, color="#00d4aa")
        ax.fill_between(range(len(cumulative)), cumulative, alpha=0.15, color="#00d4aa")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("Equity Curve (cumulative PnL)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("PnL (JPY)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "equity_curve.png", dpi=120)
        plt.close(fig)

    def _plot_drawdown(self, plt):
        if not self.all_trades:
            return
        equity = np.cumsum([t.pnl for t in self.all_trades])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(range(len(dd)), -dd, color="#ff4444", alpha=0.5)
        ax.set_title("Drawdown Curve")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Drawdown (JPY)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "drawdown_curve.png", dpi=120)
        plt.close(fig)

    def _plot_strategy_oos(self, plt):
        strat_map: dict[str, list] = {}
        for t in self.oos_trades:
            strat_map.setdefault(t.strategy_name, []).append(t)
        if not strat_map:
            return
        names = sorted(strat_map.keys())
        pfs = [_calc(strat_map[n])["pf"] for n in names]
        colors = ["#00d4aa" if pf >= 1.0 else "#ff4444" for pf in pfs]
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(names, pfs, color=colors)
        ax.axvline(x=1.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_title("Strategy OOS Profit Factor")
        ax.set_xlabel("PF")
        for bar, pf in zip(bars, pfs):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{pf:.2f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "strategy_oos.png", dpi=120)
        plt.close(fig)

    def _plot_regime_heatmap(self, plt):
        if not self.strategy_regime_trades:
            return
        data: dict[str, dict[str, float]] = {}
        for (sname, regime), trades in self.strategy_regime_trades.items():
            if len(trades) < 2:
                continue
            data.setdefault(sname, {})[regime] = _calc(trades)["pf"]
        if not data:
            return
        strategies = sorted(data.keys())
        regimes = sorted({r for d in data.values() for r in d})
        matrix = []
        for s in strategies:
            row = [data.get(s, {}).get(r, 0) for r in regimes]
            matrix.append(row)
        fig, ax = plt.subplots(figsize=(8, max(3, len(strategies) * 0.5 + 1)))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)
        ax.set_xticks(range(len(regimes)))
        ax.set_xticklabels(regimes, rotation=45, ha="right")
        ax.set_yticks(range(len(strategies)))
        ax.set_yticklabels(strategies)
        for i in range(len(strategies)):
            for j in range(len(regimes)):
                val = matrix[i][j]
                if val > 0:
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("Strategy x Regime PF Heatmap")
        fig.colorbar(im, ax=ax, shrink=0.8, label="PF")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "regime_heatmap.png", dpi=120)
        plt.close(fig)

    def _plot_confidence_vs_pnl(self, plt):
        """Scatter: selector_score vs PnL (confidence is not stored directly)."""
        scores = []
        pnls = []
        for t in self.all_trades:
            feats = t.features_at_entry or {}
            score = feats.get("selector_score", feats.get("_selector_score"))
            if score is not None:
                scores.append(float(score))
                pnls.append(t.pnl)
        if len(scores) < 3:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#00d4aa" if p > 0 else "#ff4444" for p in pnls]
        ax.scatter(scores, pnls, c=colors, alpha=0.6, s=40)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("Selector Score vs PnL")
        ax.set_xlabel("Selector Score")
        ax.set_ylabel("PnL (JPY)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "confidence_vs_pnl.png", dpi=120)
        plt.close(fig)

    def _plot_feature_compare(self, plt):
        """Compare key feature values between winning and losing trades."""
        compare_keys = ["atr", "volume_ratio", "gap_pct", "vwap_distance", "trend_strength"]
        wins_data: dict[str, list] = {k: [] for k in compare_keys}
        losses_data: dict[str, list] = {k: [] for k in compare_keys}

        for t in self.all_trades:
            feats = t.features_at_entry or {}
            target = wins_data if t.pnl > 0 else losses_data
            for k in compare_keys:
                v = feats.get(k)
                if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
                    target[k].append(float(v))

        # Only plot keys that have data
        valid_keys = [k for k in compare_keys if wins_data[k] and losses_data[k]]
        if not valid_keys:
            return

        fig, axes = plt.subplots(1, len(valid_keys), figsize=(3 * len(valid_keys), 4))
        if len(valid_keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, valid_keys):
            w_med = np.median(wins_data[k])
            l_med = np.median(losses_data[k])
            ax.bar(["Win", "Lose"], [w_med, l_med], color=["#00d4aa", "#ff4444"], alpha=0.7)
            ax.set_title(k, fontsize=9)
            ax.set_ylabel("Median")
        fig.suptitle("Feature: Win vs Lose (Median)", fontsize=11)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "feature_win_loss_compare.png", dpi=120)
        plt.close(fig)

    def _plot_cumulative_ic(self, plt):
        """Cumulative IC plot (論文 Prompt3 図)."""
        ic_data = self._ic_icir()
        rolling_ics = ic_data.get("rolling_ics", [])
        if len(rolling_ics) < 3:
            return
        cumulative = np.cumsum(rolling_ics)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(cumulative, linewidth=1.5, color="#4488ff")
        ax.fill_between(range(len(cumulative)), cumulative, alpha=0.15, color="#4488ff")
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("Cumulative IC (selector_score vs return)")
        ax.set_xlabel("Window #")
        ax.set_ylabel("Cumulative IC")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "cumulative_ic.png", dpi=120)
        plt.close(fig)

    def _plot_net_exposure(self, plt):
        """Net exposure per trade (論文 Prompt3 図).

        +1 = long, -1 = short. Cumulative shows directional bias over time.
        """
        if not self.all_trades:
            return
        directions = [1 if t.direction == "long" else -1 for t in self.all_trades]
        cumulative = np.cumsum(directions)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(range(len(directions)), directions, color=["#00d4aa" if d > 0 else "#ff4444" for d in directions],
               alpha=0.6, width=0.8)
        ax2 = ax.twinx()
        ax2.plot(cumulative, linewidth=1.5, color="#ffaa00", label="Cumulative")
        ax2.set_ylabel("Cumulative Direction", color="#ffaa00")
        ax.set_title("Net Exposure (Direction Bias)")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Direction (+1=Long, -1=Short)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "net_exposure.png", dpi=120)
        plt.close(fig)
