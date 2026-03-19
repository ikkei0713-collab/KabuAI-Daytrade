"""Analysis page -- deep performance analytics."""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime, timedelta

import streamlit as st

from ui.components.cards import kpi_card
from ui.components.charts import (
    create_equity_curve,
    create_feature_importance,
    create_holding_time_distribution,
    create_strategy_comparison,
    create_trade_time_distribution,
    create_win_loss_distribution,
    create_win_rate_trend,
)


# ── Demo data ─────────────────────────────────────────────────────────────────

def _generate_trades(days: int = 60):
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"]
    conditions = ["bull", "bear", "range", "volatile"]
    trades = []
    for i in range(days * 3):
        day_offset = random.randint(0, days - 1)
        hour = random.randint(9, 14)
        pnl = random.gauss(1500, 12000)
        entry = random.uniform(800, 6000)
        trades.append(dict(
            date=(datetime.now() - timedelta(days=day_offset)).date(),
            hour=hour,
            strategy_name=random.choice(strategies),
            direction=random.choice(["long", "short"]),
            pnl=round(pnl),
            pnl_pct=round(pnl / (entry * 100) * 100, 2),
            holding_minutes=random.randint(3, 350),
            market_condition=random.choice(conditions),
        ))
    return trades


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Analysis")

    # Date range selector
    col_from, col_to = st.columns(2)
    with col_from:
        date_from = st.date_input(
            "From",
            value=datetime.now().date() - timedelta(days=60),
            key="analysis_from",
        )
    with col_to:
        date_to = st.date_input("To", value=datetime.now().date(), key="analysis_to")

    trades = _generate_trades(60)
    trades = [t for t in trades if date_from <= t["date"] <= date_to]

    if not trades:
        st.warning("No trades in the selected date range.")
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    total_pnl = sum(t["pnl"] for t in trades)
    wins = [t for t in trades if t["pnl"] >= 0]
    losses = [t for t in trades if t["pnl"] < 0]
    win_rate = len(wins) / len(trades) if trades else 0
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    avg_hold = sum(t["holding_minutes"] for t in trades) / len(trades)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        sign = "+" if total_pnl >= 0 else ""
        kpi_card("Total P&L", f"{sign}{total_pnl:,.0f}")
    with k2:
        kpi_card("Win Rate", f"{win_rate:.1%}")
    with k3:
        kpi_card("Profit Factor", f"{pf:.2f}")
    with k4:
        kpi_card("Total Trades", str(len(trades)))
    with k5:
        kpi_card("Avg Hold", f"{avg_hold:.0f} min")

    st.markdown("")

    # ── Row 1: P&L Over Time & Strategy Comparison ────────────────────────────
    col_eq, col_strat = st.columns(2)

    with col_eq:
        # Cumulative P&L
        sorted_trades = sorted(trades, key=lambda t: t["date"])
        dates_seen = []
        cum_pnl = []
        running = 0
        daily_pnl: dict = {}
        for t in sorted_trades:
            daily_pnl.setdefault(t["date"], 0)
            daily_pnl[t["date"]] += t["pnl"]

        for d in sorted(daily_pnl.keys()):
            running += daily_pnl[d]
            dates_seen.append(d)
            cum_pnl.append(running)

        fig = create_equity_curve(dates_seen, cum_pnl, "Cumulative P&L")
        st.plotly_chart(fig, use_container_width=True)

    with col_strat:
        strat_pnl: dict[str, float] = {}
        for t in trades:
            strat_pnl.setdefault(t["strategy_name"], 0)
            strat_pnl[t["strategy_name"]] += t["pnl"]
        fig = create_strategy_comparison(
            list(strat_pnl.keys()),
            list(strat_pnl.values()),
            "P&L by Strategy",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Win Rate Trend & P&L Distribution ─────────────────────────────
    col_wr, col_dist = st.columns(2)

    with col_wr:
        # Rolling 10-trade win rate
        sorted_all = sorted(trades, key=lambda t: t["date"])
        window = 10
        wr_dates = []
        wr_values = []
        for i in range(window, len(sorted_all)):
            chunk = sorted_all[i - window: i]
            wr_val = sum(1 for c in chunk if c["pnl"] >= 0) / window
            wr_dates.append(chunk[-1]["date"])
            wr_values.append(wr_val)
        fig = create_win_rate_trend(wr_dates, wr_values, "Rolling 10-Trade Win Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col_dist:
        pnl_vals = [t["pnl"] for t in trades]
        fig = create_win_loss_distribution(pnl_vals, "Trade P&L Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Time Distribution & Holding Time ───────────────────────────────
    col_time, col_hold = st.columns(2)

    with col_time:
        hour_counts: Counter = Counter()
        for t in trades:
            hour_counts[t["hour"]] += 1
        hours = list(range(9, 15))
        counts = [hour_counts.get(h, 0) for h in hours]
        fig = create_trade_time_distribution(hours, counts, "Trades by Hour (JST)")
        st.plotly_chart(fig, use_container_width=True)

    with col_hold:
        hold_times = [t["holding_minutes"] for t in trades]
        fig = create_holding_time_distribution(hold_times, "Holding Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 4: By Market Condition & Feature Correlation ──────────────────────
    col_cond, col_feat = st.columns(2)

    with col_cond:
        st.markdown(
            '<div class="section-header"><h3>P&L by Market Condition</h3></div>',
            unsafe_allow_html=True,
        )
        cond_pnl: dict[str, list] = {}
        for t in trades:
            cond_pnl.setdefault(t["market_condition"], []).append(t["pnl"])

        for condition in ["bull", "bear", "range", "volatile"]:
            if condition in cond_pnl:
                vals = cond_pnl[condition]
                total = sum(vals)
                wr = sum(1 for v in vals if v >= 0) / len(vals) if vals else 0
                sign = "+" if total >= 0 else ""
                color = "#00d4aa" if total >= 0 else "#ff4757"
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                padding:10px 0; border-bottom:1px solid #2a2a4a; font-size:0.9rem;">
                        <span style="text-transform:capitalize; font-weight:500;">{condition}</span>
                        <div>
                            <span style="color:{color}; font-weight:600;">{sign}{total:,.0f}</span>
                            <span style="color:#64748b; margin-left:12px;">{len(vals)} trades</span>
                            <span style="color:#94a3b8; margin-left:12px;">WR: {wr:.0%}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with col_feat:
        st.markdown(
            '<div class="section-header"><h3>Feature Correlation with P&L</h3></div>',
            unsafe_allow_html=True,
        )
        features = [
            "volume_ratio", "momentum_5m", "spread_bps", "orderbook_imb",
            "vwap_dist", "atr_ratio", "sector_mom", "nikkei_corr",
        ]
        correlations = [round(random.uniform(-0.3, 0.5), 3) for _ in features]
        fig = create_feature_importance(features, correlations, "Feature-P&L Correlation", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ── By Strategy & Condition Matrix ────────────────────────────────────────
    st.markdown(
        '<div class="section-header"><h3>Strategy x Condition Breakdown</h3></div>',
        unsafe_allow_html=True,
    )
    import pandas as pd

    strat_cond: dict[str, dict[str, list]] = {}
    for t in trades:
        strat_cond.setdefault(t["strategy_name"], {}).setdefault(t["market_condition"], []).append(t["pnl"])

    rows = []
    for sname in sorted(strat_cond.keys()):
        row = {"Strategy": sname}
        for cond in ["bull", "bear", "range", "volatile"]:
            vals = strat_cond[sname].get(cond, [])
            if vals:
                wr = sum(1 for v in vals if v >= 0) / len(vals)
                total = sum(vals)
                row[f"{cond.capitalize()} WR"] = f"{wr:.0%}"
                row[f"{cond.capitalize()} P&L"] = f"{total:+,.0f}"
            else:
                row[f"{cond.capitalize()} WR"] = "-"
                row[f"{cond.capitalize()} P&L"] = "-"
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
