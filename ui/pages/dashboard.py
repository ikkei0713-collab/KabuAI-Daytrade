"""Dashboard page -- main overview of the trading system."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import streamlit as st

from ui.components.cards import kpi_card, system_status_item
from ui.components.charts import (
    create_equity_curve,
    create_strategy_donut,
)
from ui.components.tables import trades_table


# ── Demo data generators ─────────────────────────────────────────────────────
# These will be replaced with real data sources once the backend is connected.

def _demo_kpis() -> dict:
    return dict(
        total_pnl=random.randint(-30000, 80000),
        win_rate=random.uniform(0.45, 0.72),
        profit_factor=random.uniform(0.8, 2.5),
        active_positions=random.randint(0, 5),
        total_trades_today=random.randint(3, 25),
    )


def _demo_equity_curve():
    base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    dates, cum = [], []
    running = 0
    for i in range(30):
        d = base - timedelta(days=29 - i)
        dates.append(d)
        running += random.randint(-15000, 20000)
        cum.append(running)
    return dates, cum


def _demo_strategy_perf():
    names = ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"]
    vals = [random.randint(10000, 120000) for _ in names]
    return names, vals


def _demo_trades():
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"]
    conditions = ["bull", "bear", "range", "volatile"]
    trades = []
    for i in range(20):
        pnl = random.randint(-20000, 35000)
        entry = random.uniform(1000, 5000)
        trades.append(dict(
            ticker=f"{random.randint(1000,9999)}.T",
            strategy_name=random.choice(strategies),
            direction=random.choice(["long", "short"]),
            entry_price=round(entry, 1),
            exit_price=round(entry + pnl / 100, 1),
            pnl=pnl,
            pnl_pct=round(pnl / (entry * 100) * 100, 2),
            holding_minutes=random.randint(5, 300),
            market_condition=random.choice(conditions),
        ))
    return trades


def _demo_signals():
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb"]
    signals = []
    for _ in range(random.randint(2, 6)):
        signals.append(dict(
            ticker=f"{random.randint(1000,9999)}.T",
            strategy=random.choice(strategies),
            direction=random.choice(["LONG", "SHORT"]),
            confidence=round(random.uniform(0.6, 0.95), 2),
            price=round(random.uniform(800, 6000), 1),
        ))
    return signals


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Dashboard")
    st.caption(f"Market overview | {datetime.now().strftime('%Y-%m-%d %H:%M')} JST")

    kpis = _demo_kpis()

    # ── Row 1: KPI Cards ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pnl = kpis["total_pnl"]
        sign = "+" if pnl >= 0 else ""
        kpi_card(
            "Today's P&L",
            f"{sign}{pnl:,.0f}",
            delta=f"{sign}{pnl:,.0f}",
            delta_color="normal" if pnl >= 0 else "inverse",
        )
    with c2:
        wr = kpis["win_rate"]
        kpi_card("Win Rate", f"{wr:.1%}", delta=f"{wr - 0.5:+.1%} vs 50%")
    with c3:
        pf = kpis["profit_factor"]
        kpi_card(
            "Profit Factor",
            f"{pf:.2f}",
            delta="Good" if pf >= 1.5 else ("OK" if pf >= 1.0 else "Poor"),
        )
    with c4:
        kpi_card(
            "Active Positions",
            str(kpis["active_positions"]),
            delta=f"{kpis['total_trades_today']} trades today",
        )

    st.markdown("")

    # ── Row 2: Charts ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        dates, cum_pnl = _demo_equity_curve()
        fig = create_equity_curve(dates, cum_pnl, "Equity Curve (30 Days)")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        names, vals = _demo_strategy_perf()
        fig = create_strategy_donut(names, vals, "Strategy Performance")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Recent Trades & Active Signals ─────────────────────────────────
    col_trades, col_signals = st.columns([3, 2])

    with col_trades:
        st.markdown(
            '<div class="section-header"><h3>Recent Trades</h3></div>',
            unsafe_allow_html=True,
        )
        trades_table(_demo_trades())

    with col_signals:
        st.markdown(
            '<div class="section-header"><h3>Active Signals</h3></div>',
            unsafe_allow_html=True,
        )
        signals = _demo_signals()
        for sig in signals:
            dir_color = "#00d4aa" if sig["direction"] == "LONG" else "#ff4757"
            dir_icon = "\u25b2" if sig["direction"] == "LONG" else "\u25bc"
            st.markdown(
                f"""
                <div class="card" style="padding:12px 16px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>{sig['ticker']}</strong>
                            <span style="color:{dir_color}; margin-left:8px; font-weight:600;">
                                {dir_icon} {sig['direction']}
                            </span>
                        </div>
                        <span style="color:#94a3b8; font-size:0.82rem;">{sig['strategy']}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; margin-top:6px; font-size:0.82rem; color:#64748b;">
                        <span>Price: {sig['price']:,.1f}</span>
                        <span>Confidence: <strong style="color:#e2e8f0;">{sig['confidence']:.0%}</strong></span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Row 4: Market Condition & System Status ───────────────────────────────
    col_market, col_system = st.columns(2)

    with col_market:
        st.markdown(
            '<div class="section-header"><h3>Market Condition</h3></div>',
            unsafe_allow_html=True,
        )
        regime = random.choice(["bull", "bear", "range", "volatile"])
        regime_colors = {
            "bull": "#00d4aa",
            "bear": "#ff4757",
            "range": "#f59e0b",
            "volatile": "#6366f1",
        }
        rc = regime_colors[regime]
        nikkei = random.uniform(-2.5, 2.5)
        topix = random.uniform(-2.0, 2.0)
        usd_jpy = round(random.uniform(148, 156), 2)
        vix = round(random.uniform(12, 35), 1)

        st.markdown(
            f"""
            <div class="card">
                <div style="text-align:center; margin-bottom:16px;">
                    <span style="font-size:1.4rem; font-weight:700; color:{rc}; text-transform:uppercase;
                                 letter-spacing:0.15em;">{regime}</span>
                    <div style="color:#64748b; font-size:0.8rem; margin-top:2px;">Current Market Regime</div>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; font-size:0.88rem;">
                    <div>
                        <span style="color:#94a3b8;">Nikkei 225</span><br>
                        <strong style="color:{'#00d4aa' if nikkei >= 0 else '#ff4757'};">
                            {'+' if nikkei >= 0 else ''}{nikkei:.2f}%
                        </strong>
                    </div>
                    <div>
                        <span style="color:#94a3b8;">TOPIX</span><br>
                        <strong style="color:{'#00d4aa' if topix >= 0 else '#ff4757'};">
                            {'+' if topix >= 0 else ''}{topix:.2f}%
                        </strong>
                    </div>
                    <div>
                        <span style="color:#94a3b8;">USD/JPY</span><br>
                        <strong>{usd_jpy}</strong>
                    </div>
                    <div>
                        <span style="color:#94a3b8;">VIX</span><br>
                        <strong style="color:{'#ff4757' if vix > 25 else '#e2e8f0'};">{vix}</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_system:
        st.markdown(
            '<div class="section-header"><h3>System Status</h3></div>',
            unsafe_allow_html=True,
        )
        with st.container():
            system_status_item("Data Feed (J-Quants)", "online", "Connected")
            system_status_item("Broker Connection", "warning", "Paper Mode")
            system_status_item("Safety Guards", "online", "All Active")
            system_status_item("Strategy Engine", "online", "4 strategies loaded")
            system_status_item("Knowledge Base", "online", "Last update 2h ago")
            system_status_item("Daily Loss Limit", "online", f"Used {abs(min(0, random.randint(-30000, 0))):,.0f} / 50,000")
