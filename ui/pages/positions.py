"""Positions page -- active and historical positions."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import streamlit as st

from ui.components.cards import kpi_card
from ui.components.tables import positions_table, trades_table


# ── Demo data ─────────────────────────────────────────────────────────────────

def _demo_positions():
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"]
    positions = []
    for _ in range(random.randint(1, 5)):
        entry = round(random.uniform(800, 6000), 1)
        current = round(entry * random.uniform(0.97, 1.04), 1)
        direction = random.choice(["long", "short"])
        if direction == "long":
            pnl = (current - entry) * 100
        else:
            pnl = (entry - current) * 100
        positions.append(dict(
            id=f"pos_{random.randint(1000,9999)}",
            ticker=f"{random.randint(1000,9999)}.T",
            strategy_name=random.choice(strategies),
            direction=direction,
            entry_price=entry,
            current_price=current,
            unrealized_pnl=round(pnl),
            holding_minutes=random.randint(5, 300),
            stop_loss=round(entry * (0.97 if direction == "long" else 1.03), 1),
            take_profit=round(entry * (1.05 if direction == "long" else 0.95), 1),
        ))
    return positions


def _demo_historical_trades(n: int = 50):
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"]
    conditions = ["bull", "bear", "range", "volatile"]
    trades = []
    for i in range(n):
        entry = round(random.uniform(800, 6000), 1)
        pnl = random.randint(-25000, 40000)
        trades.append(dict(
            ticker=f"{random.randint(1000,9999)}.T",
            strategy_name=random.choice(strategies),
            direction=random.choice(["long", "short"]),
            entry_price=entry,
            exit_price=round(entry + pnl / 100, 1),
            pnl=pnl,
            pnl_pct=round(pnl / (entry * 100) * 100, 2),
            holding_minutes=random.randint(5, 350),
            market_condition=random.choice(conditions),
            entry_time=(datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            exit_time=(datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 5))).isoformat(),
        ))
    return trades


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Positions")

    tab_active, tab_history = st.tabs(["Active Positions", "Historical"])

    # ── Active Positions ──────────────────────────────────────────────────────
    with tab_active:
        positions = _demo_positions()

        # Summary KPIs
        total_pnl = sum(p["unrealized_pnl"] for p in positions)
        c1, c2, c3 = st.columns(3)
        with c1:
            sign = "+" if total_pnl >= 0 else ""
            kpi_card("Total Unrealized P&L", f"{sign}{total_pnl:,.0f}")
        with c2:
            kpi_card("Active Positions", str(len(positions)))
        with c3:
            if positions:
                avg_hold = sum(p["holding_minutes"] for p in positions) / len(positions)
            else:
                avg_hold = 0
            kpi_card("Avg Holding Time", f"{avg_hold:.0f} min")

        st.markdown("")
        positions_table(positions)

        # Position detail expanders
        st.markdown("")
        st.markdown(
            '<div class="section-header"><h3>Position Details</h3></div>',
            unsafe_allow_html=True,
        )
        for pos in positions:
            pnl_color = "#00d4aa" if pos["unrealized_pnl"] >= 0 else "#ff4757"
            pnl_sign = "+" if pos["unrealized_pnl"] >= 0 else ""
            with st.expander(
                f"{pos['ticker']}  |  {pos['direction'].upper()}  |  "
                f"P&L: {pnl_sign}{pos['unrealized_pnl']:,.0f}"
            ):
                dc1, dc2, dc3, dc4 = st.columns(4)
                dc1.metric("Strategy", pos["strategy_name"])
                dc2.metric("Entry Price", f"{pos['entry_price']:,.1f}")
                dc3.metric("Current Price", f"{pos['current_price']:,.1f}")
                dc4.metric("Holding", f"{pos['holding_minutes']} min")

                dc5, dc6, dc7, dc8 = st.columns(4)
                dc5.metric("Stop Loss", f"{pos['stop_loss']:,.1f}")
                dc6.metric("Take Profit", f"{pos['take_profit']:,.1f}")
                pnl_pct = pos["unrealized_pnl"] / (pos["entry_price"] * 100) * 100
                dc7.metric("P&L %", f"{pnl_pct:+.2f}%")
                dc8.markdown("")

                # Close button (paper trading)
                st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                if st.button(f"Close Position {pos['id']}", key=f"close_{pos['id']}"):
                    st.success(f"Position {pos['id']} ({pos['ticker']}) close order submitted (paper).")
                st.markdown("</div>", unsafe_allow_html=True)

    # ── Historical Positions ──────────────────────────────────────────────────
    with tab_history:
        col_from, col_to, col_strategy = st.columns(3)
        with col_from:
            date_from = st.date_input(
                "From",
                value=datetime.now().date() - timedelta(days=30),
                key="hist_from",
            )
        with col_to:
            date_to = st.date_input("To", value=datetime.now().date(), key="hist_to")
        with col_strategy:
            strat_filter = st.selectbox(
                "Strategy",
                ["All", "MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"],
                key="hist_strat",
            )

        historical = _demo_historical_trades(50)

        if strat_filter != "All":
            historical = [t for t in historical if t["strategy_name"] == strat_filter]

        st.caption(f"Showing {len(historical)} trades")
        trades_table(historical, max_rows=50)
