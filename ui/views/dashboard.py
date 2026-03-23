"""ダッシュボード -- リアルデータ表示"""

import json
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ticker_map import format_ticker

DB_PATH = Path.home() / "dev" / "KabuAI-Daytrade" / "db" / "kabuai.db"
SUMMARY_PATH = Path.home() / "dev" / "KabuAI-Daytrade" / "knowledge" / "backtest_summary.json"
LOG_PATH = Path.home() / "dev" / "KabuAI-Daytrade" / "logs" / "paper_stdout.log"


def _get_db():
    """SQLite connection (sync for Streamlit)"""
    if not DB_PATH.exists():
        return None
    return sqlite3.connect(str(DB_PATH))


def _is_paper_running() -> bool:
    """Check if run_paper.py is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_paper.py"], capture_output=True, text=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def _is_backtest_running() -> bool:
    """Check if run_backtest_learn.py is running"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_backtest_learn.py"], capture_output=True, text=True, timeout=3
        )
        return result.returncode == 0
    except Exception:
        return False


def _load_trades(conn, days=30) -> pd.DataFrame:
    """Load recent trades from DB"""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    query = """
        SELECT ticker, strategy_name, direction, entry_price, exit_price,
               pnl, pnl_pct, holding_minutes, entry_reason, exit_reason,
               entry_time, exit_time, market_condition
        FROM trades
        WHERE entry_time > ?
        ORDER BY entry_time DESC
        LIMIT 200
    """
    try:
        df = pd.read_sql_query(query, conn, params=(cutoff,))
        return df
    except Exception:
        return pd.DataFrame()


def _load_strategy_performance(conn) -> pd.DataFrame:
    """Load strategy performance from DB"""
    query = """
        SELECT strategy_name, total_trades, wins, losses, win_rate,
               profit_factor, avg_pnl, avg_holding_minutes
        FROM strategy_performance
        ORDER BY total_trades DESC
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame()


def _load_knowledge(conn, limit=10) -> list[dict]:
    """Load latest knowledge entries"""
    query = """
        SELECT category, content, confidence, date
        FROM knowledge
        ORDER BY date DESC, id DESC
        LIMIT ?
    """
    try:
        cursor = conn.execute(query, (limit,))
        rows = cursor.fetchall()
        return [{"category": r[0], "content": r[1], "confidence": r[2], "date": r[3]} for r in rows]
    except Exception:
        return []


def _load_backtest_summary() -> dict | None:
    """Load latest backtest summary"""
    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text())
        except Exception:
            pass
    return None


def _load_paper_state() -> dict | None:
    """Load paper trading state (watchlist, regime, etc.)"""
    state_path = Path.home() / "dev" / "KabuAI-Daytrade" / "knowledge" / "paper_state.json"
    if state_path.exists():
        try:
            return json.loads(state_path.read_text())
        except Exception:
            pass
    return None


def _tail_log(n=15) -> list[str]:
    """Get last N lines of paper trading log"""
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text().strip().split("\n")
        return lines[-n:]
    except Exception:
        return []


def _equity_curve(trades_df: pd.DataFrame) -> go.Figure:
    """Build equity curve from trades"""
    fig = go.Figure()
    if trades_df.empty:
        fig.add_annotation(text="トレードデータなし", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        df = trades_df.sort_values("entry_time")
        df["cum_pnl"] = df["pnl"].cumsum()
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df["cum_pnl"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#00d4aa", width=2),
            fillcolor="rgba(0,212,170,0.1)",
        ))
    fig.update_layout(
        title="累計損益",
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1a2e",
        height=300,
        margin=dict(l=40, r=20, t=40, b=30),
        yaxis_title="円",
        xaxis_title="トレード#",
    )
    return fig


def render():
    # --- Header with status ---
    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown("## 📊 ダッシュボード")
    with col_status:
        paper_on = _is_paper_running()
        bt_on = _is_backtest_running()
        if paper_on:
            st.success("🟢 ペーパートレード稼働中")
        else:
            st.warning("⚪ ペーパートレード停止中")
        if bt_on:
            st.info("🔄 バックテスト実行中")

    conn = _get_db()
    if not conn:
        st.error("データベース未初期化。`make setup` を実行してください。")
        return

    try:
        trades_df = _load_trades(conn)
        perf_df = _load_strategy_performance(conn)
        knowledge = _load_knowledge(conn)
        summary = _load_backtest_summary()

        # --- KPI Row ---
        total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0
        total_trades = len(trades_df)
        win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0
        gp = trades_df[trades_df["pnl"] > 0]["pnl"].sum() if not trades_df.empty else 0
        gl = abs(trades_df[trades_df["pnl"] <= 0]["pnl"].sum()) if not trades_df.empty else 0
        pf = gp / gl if gl > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        sign = "+" if total_pnl >= 0 else ""
        c1.metric("累計損益", f"¥{sign}{total_pnl:,.0f}")
        c2.metric("勝率", f"{win_rate:.1%}")
        c3.metric("PF", f"{pf:.2f}")
        c4.metric("トレード数", f"{total_trades}")

        # --- Equity Curve ---
        fig = _equity_curve(trades_df)
        st.plotly_chart(fig, use_container_width=True)

        # --- Two columns: Trades + Strategy Performance ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 直近トレード")
            if not trades_df.empty:
                display_df = trades_df[["ticker", "strategy_name", "direction", "pnl", "pnl_pct", "exit_reason", "market_condition"]].head(20).copy()
                display_df["ticker"] = display_df["ticker"].apply(format_ticker)
                display_df.columns = ["銘柄", "戦略", "方向", "損益", "損益%", "決済理由", "レジーム"]
                st.dataframe(display_df, hide_index=True, use_container_width=True)
            else:
                st.info("トレードなし。バックテストまたはペーパートレードを実行してください。")

        with col2:
            st.markdown("#### 戦略別成績")
            if not perf_df.empty:
                display_perf = perf_df.copy()
                display_perf.columns = ["戦略", "件数", "勝", "負", "勝率", "PF", "平均損益", "平均保有(分)"]
                display_perf["勝率"] = display_perf["勝率"].apply(lambda x: f"{x:.0%}")
                display_perf["平均損益"] = display_perf["平均損益"].apply(lambda x: f"¥{x:+,.0f}")
                st.dataframe(display_perf, hide_index=True, use_container_width=True)
            else:
                st.info("戦略データなし")

        # --- Convergence Filter Status (v3.3) ---
        feedback_packet_path = Path.home() / "dev" / "KabuAI-Daytrade" / "knowledge" / "feedback_packet.json"
        if feedback_packet_path.exists():
            try:
                fb_packet = json.loads(feedback_packet_path.read_text())
                conv = fb_packet.get("convergence_analysis", {})
                if conv:
                    with st.expander("収束フィルタ分析 (v3.3)"):
                        cc1, cc2, cc3 = st.columns(3)
                        conv_m = conv.get("convergence_entry_metrics", {})
                        exp_m = conv.get("expansion_entry_metrics", {})
                        with cc1:
                            st.metric("収束後エントリー",
                                      f"{conv_m.get('count', 0)}件",
                                      f"WR {conv_m.get('win_rate', 0):.0%}")
                        with cc2:
                            st.metric("拡散エントリー",
                                      f"{exp_m.get('count', 0)}件",
                                      f"WR {exp_m.get('win_rate', 0):.0%}")
                        with cc3:
                            hints = fb_packet.get("recommendation_hints", {})
                            helped = hints.get("convergence_filter_helped", False)
                            st.metric("フィルタ有効性",
                                      "有効" if helped else "未確定",
                                      f"拡散損失率 {hints.get('expansion_chasing_loss_rate', 0):.0%}")
            except Exception:
                pass

        # --- Backtest Summary ---
        if summary:
            st.markdown("#### 最新バックテスト結果")
            bc1, bc2, bc3, bc4 = st.columns(4)
            bc1.metric("トレード数", summary.get("total_trades", 0))
            bc2.metric("勝率", f"{summary.get('win_rate', 0):.1%}")
            bc3.metric("PF", f"{summary.get('profit_factor', 0):.2f}")
            pnl_key = "total_pnl_cost_adjusted" if "total_pnl_cost_adjusted" in summary else "total_pnl"
            bc4.metric("損益(コスト後)", f"¥{summary.get(pnl_key, 0):+,.0f}")

            # IS vs OOS
            if "in_sample" in summary and "out_of_sample" in summary:
                is_d = summary["in_sample"]
                oos_d = summary["out_of_sample"]
                ic1, ic2 = st.columns(2)
                with ic1:
                    st.caption(f"In-Sample: {is_d['trades']}件 勝率{is_d['win_rate']:.0%} PF={is_d['pf']:.2f}")
                with ic2:
                    st.caption(f"Out-of-Sample: {oos_d['trades']}件 勝率{oos_d['win_rate']:.0%} PF={oos_d['pf']:.2f}")

        # --- Paper State: Watchlist, Regime, Strategy On/Off ---
        paper_state = _load_paper_state()
        if paper_state:
            st.markdown("---")

            # Regime + Anomaly status
            rc1, rc2, rc3 = st.columns(3)
            with rc1:
                regime = paper_state.get("regime", "unknown")
                regime_conf = paper_state.get("regime_confidence", 0)
                regime_colors = {
                    "trend_up": "green", "trend_down": "red", "range": "orange",
                    "volatile": "purple", "low_vol": "gray",
                }
                st.metric("レジーム", f"{regime} ({regime_conf:.0%})")
            with rc2:
                active_count = len(paper_state.get("active_strategies", []))
                disabled_count = len(paper_state.get("disabled_strategies", []))
                st.metric("戦略", f"{active_count} ON / {disabled_count} OFF")
            with rc3:
                if paper_state.get("halted"):
                    st.error(f"異常停止: {paper_state.get('halt_reason', '')}")
                else:
                    st.success("正常稼働")

            # Strategy status details (active/filter/supplement/watch/off)
            with st.expander("戦略 Status 詳細"):
                status_groups = paper_state.get("strategy_status", {})
                if status_groups:
                    for status_label in ["active", "filter", "supplement", "watch", "off"]:
                        names = status_groups.get(status_label, [])
                        if names:
                            color = {"active": "green", "filter": "blue", "supplement": "orange",
                                     "watch": "gray", "off": "red"}.get(status_label, "gray")
                            st.markdown(f"**:{color}[{status_label.upper()}]** ({len(names)})")
                            for s in names:
                                st.text(f"  {s}")
                else:
                    # Fallback to old on/off display
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.caption("ON (稼働中)")
                        for s in paper_state.get("active_strategies", []):
                            st.text(f"  {s}")
                    with sc2:
                        st.caption("OFF (停止中)")
                        for s in paper_state.get("disabled_strategies", []):
                            st.text(f"  {s}")

            # Proxy usage rate
            proxy_info = paper_state.get("proxy_summary", {})
            if proxy_info:
                with st.expander("Proxy 依存度 (データ品質)"):
                    proxy_data = []
                    for name, info in sorted(proxy_info.items(), key=lambda x: -x[1].get("proxy_usage_rate", 0)):
                        rate = info.get("proxy_usage_rate", 0)
                        quality = "HIGH" if rate < 0.3 else "MEDIUM" if rate < 0.7 else "LOW"
                        proxy_data.append({
                            "戦略": name,
                            "Status": info.get("status", "off"),
                            "Proxy率": f"{rate:.0%}",
                            "ペナルティ": f"{info.get('proxy_penalty', 0):.3f}",
                            "信頼度": quality,
                        })
                    st.dataframe(pd.DataFrame(proxy_data), hide_index=True, use_container_width=True)
                    st.caption(
                        "Proxy率 = 擬似特徴量への依存度。"
                        "HIGH信頼度 = real data 中心。LOW = 日足推定値が多く評価信頼度が低い。"
                    )

            # Data quality warnings
            warnings = paper_state.get("data_quality_warnings", [])
            if warnings:
                for w in warnings:
                    st.warning(f"Data Quality: {w}")

            # Sector Bias
            sb = paper_state.get("sector_bias")
            if sb and sb.get("fetch_success"):
                st.markdown("#### 米国→日本 セクターバイアス")
                sb_cols = st.columns(3)
                with sb_cols[0]:
                    spy_ret = sb.get("spy_return", 0)
                    st.metric("SPY前日", f"{spy_ret:+.2%}")
                with sb_cols[1]:
                    risk = "Risk-OFF" if sb.get("risk_off") else "Risk-ON"
                    st.metric("リスク", risk)
                with sb_cols[2]:
                    jp_bias = sb.get("jp_sector_bias", {})
                    bullish_n = sum(1 for v in jp_bias.values() if v > 0.3)
                    bearish_n = sum(1 for v in jp_bias.values() if v < -0.3)
                    st.metric("業種バイアス", f"強気{bullish_n} / 弱気{bearish_n}")

                # US ETF returns
                with st.expander("米国セクターETF詳細"):
                    us_ret = sb.get("us_returns", {})
                    if us_ret:
                        us_data = [{"ETF": k, "リターン": f"{v:+.2%}"} for k, v in sorted(us_ret.items(), key=lambda x: x[1], reverse=True)]
                        st.dataframe(pd.DataFrame(us_data), hide_index=True, use_container_width=True)

            # Watchlist
            watchlist = paper_state.get("watchlist", [])
            if watchlist:
                st.markdown("#### ウォッチリスト (採用理由付き)")
                wl_data = []
                for w in watchlist:
                    bias_label = w.get("sector_bias_label", "-")
                    bias_val = w.get("sector_bias", 0)
                    conv_score = w.get("ma_convergence_score")
                    conv_label = f"{conv_score:.2f}" if conv_score is not None else "-"
                    # 収束後候補ラベル
                    squeeze = w.get("squeeze_breakout_ready", False)
                    conv_tag = ""
                    if squeeze:
                        conv_tag = " [収束後候補]"
                    elif conv_score is not None and conv_score < 0.3:
                        conv_tag = " [拡散注意]"
                    wl_data.append({
                        "銘柄": format_ticker(w.get("code", "")),
                        "スコア": f"{w.get('combined', 0):.3f}",
                        "ギャップ": f"{w.get('gap_pct', 0):.1f}%",
                        "出来高比": f"{w.get('relative_volume', 0):.1f}x",
                        "収束": f"{conv_label}{conv_tag}",
                        "イベント": "有" if w.get("has_event") else "-",
                        "業種バイアス": f"{bias_label} ({bias_val:+.3f})" if bias_val else "-",
                        "採用理由": w.get("reason", "")[:50],
                    })
                st.dataframe(pd.DataFrame(wl_data), hide_index=True, use_container_width=True)

            # Open positions
            positions = paper_state.get("positions", {})
            if positions:
                st.markdown("#### 保有ポジション")
                pos_data = []
                for ticker, pos in positions.items():
                    pos_data.append({
                        "銘柄": format_ticker(ticker),
                        "戦略": pos.get("strategy", ""),
                        "方向": pos.get("direction", ""),
                        "エントリー": f"¥{pos.get('entry_price', 0):,.0f}",
                        "時刻": pos.get("entry_time", "")[:16],
                    })
                st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)

        # --- Feedback Summary ---
        feedback_md_path = Path.home() / "dev" / "KabuAI-Daytrade" / "knowledge" / "feedback_summary.md"
        feedback_plots_dir = Path.home() / "dev" / "KabuAI-Daytrade" / "knowledge" / "plots"
        if feedback_md_path.exists():
            st.markdown("---")
            st.markdown("#### 改善フィードバック")
            with st.expander("feedback_summary.md (Claude Code 向け)", expanded=False):
                st.markdown(feedback_md_path.read_text(encoding="utf-8"))
            # Show key plots if available
            plot_cols = st.columns(2)
            equity_plot = feedback_plots_dir / "equity_curve.png"
            dd_plot = feedback_plots_dir / "drawdown_curve.png"
            if equity_plot.exists():
                with plot_cols[0]:
                    st.image(str(equity_plot), caption="Equity Curve", use_container_width=True)
            if dd_plot.exists():
                with plot_cols[1]:
                    st.image(str(dd_plot), caption="Drawdown", use_container_width=True)
            plot_cols2 = st.columns(2)
            oos_plot = feedback_plots_dir / "strategy_oos.png"
            regime_plot = feedback_plots_dir / "regime_heatmap.png"
            if oos_plot.exists():
                with plot_cols2[0]:
                    st.image(str(oos_plot), caption="Strategy OOS PF", use_container_width=True)
            if regime_plot.exists():
                with plot_cols2[1]:
                    st.image(str(regime_plot), caption="Regime Heatmap", use_container_width=True)

        # --- Live Log ---
        st.markdown("#### ペーパートレードログ")
        log_lines = _tail_log(15)
        if log_lines:
            st.code("\n".join(log_lines), language="text")
        else:
            st.info("ログなし。`python run_paper.py` で開始してください。")

        # --- Knowledge ---
        if knowledge:
            st.markdown("#### 最新ナレッジ")
            for k in knowledge[:5]:
                cat_emoji = {"win_pattern": "✅", "loss_pattern": "❌", "strategy_insight": "💡", "market_insight": "📊"}.get(k["category"], "📝")
                st.caption(f"{cat_emoji} [{k['date']}] {k['content'][:100]}")

    finally:
        conn.close()

    # Auto-refresh every 30 seconds
    import time
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
    if time.time() - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = time.time()
        st.rerun()
