"""Strategies page -- strategy management and performance."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import streamlit as st

from ui.components.cards import strategy_card
from ui.components.charts import (
    create_equity_curve,
    create_pnl_heatmap,
    create_strategy_comparison,
)
from ui.components.tables import trades_table


# ── Demo data ─────────────────────────────────────────────────────────────────

_STRATEGIES = [
    dict(
        name="MomentumBreak",
        description="Breakout on strong momentum with volume confirmation",
        is_active=True,
        win_rate=0.58,
        profit_factor=1.72,
        total_trades=142,
        avg_pnl=3200,
        best_condition="bull",
        params={"lookback": 20, "volume_mult": 1.5, "atr_mult": 2.0, "min_momentum": 0.02},
    ),
    dict(
        name="GapReversion",
        description="Mean reversion on opening gaps with support/resistance levels",
        is_active=True,
        win_rate=0.63,
        profit_factor=1.95,
        total_trades=98,
        avg_pnl=4100,
        best_condition="range",
        params={"min_gap_pct": 1.5, "reversion_target": 0.5, "max_gap_pct": 5.0, "volume_confirm": True},
    ),
    dict(
        name="OrderBookImb",
        description="Trade order book imbalances with price level analysis",
        is_active=True,
        win_rate=0.52,
        profit_factor=1.35,
        total_trades=210,
        avg_pnl=1800,
        best_condition="volatile",
        params={"imbalance_ratio": 3.0, "depth_levels": 5, "min_size": 10000, "decay_seconds": 30},
    ),
    dict(
        name="EventDriven",
        description="Trade on corporate events: earnings, dividends, splits",
        is_active=False,
        win_rate=0.48,
        profit_factor=0.92,
        total_trades=45,
        avg_pnl=-1500,
        best_condition="bull",
        params={"event_types": ["earnings", "dividend"], "pre_event_days": 3, "hold_through": False},
    ),
]


def _demo_strategy_trades(strategy_name: str, n: int = 15):
    conditions = ["bull", "bear", "range", "volatile"]
    trades = []
    for _ in range(n):
        entry = round(random.uniform(800, 6000), 1)
        pnl = random.randint(-20000, 35000)
        trades.append(dict(
            ticker=f"{random.randint(1000,9999)}.T",
            strategy_name=strategy_name,
            direction=random.choice(["long", "short"]),
            entry_price=entry,
            exit_price=round(entry + pnl / 100, 1),
            pnl=pnl,
            pnl_pct=round(pnl / (entry * 100) * 100, 2),
            holding_minutes=random.randint(5, 300),
            market_condition=random.choice(conditions),
        ))
    return trades


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Strategies")

    tab_overview, tab_detail = st.tabs(["Overview", "Strategy Detail"])

    # ── Overview ──────────────────────────────────────────────────────────────
    with tab_overview:
        # Strategy comparison chart
        names = [s["name"] for s in _STRATEGIES]
        pnls = [s["avg_pnl"] * s["total_trades"] for s in _STRATEGIES]
        fig = create_strategy_comparison(names, pnls, "Total P&L by Strategy")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("")

        # Strategy cards grid
        st.markdown(
            '<div class="section-header"><h3>Strategy Cards</h3></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for i, strat in enumerate(_STRATEGIES):
            with cols[i % 2]:
                strategy_card(
                    name=strat["name"],
                    is_active=strat["is_active"],
                    win_rate=strat["win_rate"],
                    profit_factor=strat["profit_factor"],
                    total_trades=strat["total_trades"],
                    best_condition=strat["best_condition"],
                    avg_pnl=strat["avg_pnl"],
                )

    # ── Strategy Detail ───────────────────────────────────────────────────────
    with tab_detail:
        selected_name = st.selectbox(
            "Select Strategy",
            [s["name"] for s in _STRATEGIES],
            key="strategy_select",
        )
        strat = next(s for s in _STRATEGIES if s["name"] == selected_name)

        # Toggle status
        col_status, col_spacer = st.columns([1, 3])
        with col_status:
            new_active = st.toggle("Active", value=strat["is_active"], key=f"toggle_{strat['name']}")
            if new_active != strat["is_active"]:
                st.info(f"Strategy {strat['name']} {'activated' if new_active else 'deactivated'} (paper mode).")

        # Full metrics
        st.markdown(
            '<div class="section-header"><h3>Performance Metrics</h3></div>',
            unsafe_allow_html=True,
        )
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Win Rate", f"{strat['win_rate']:.1%}")
        mc2.metric("Profit Factor", f"{strat['profit_factor']:.2f}")
        mc3.metric("Total Trades", str(strat["total_trades"]))
        mc4.metric("Avg P&L", f"{strat['avg_pnl']:+,.0f}")
        mc5.metric("Best Condition", strat["best_condition"].capitalize())

        st.markdown("")

        # Mini equity curve
        col_chart, col_heatmap = st.columns(2)
        with col_chart:
            base = datetime.now().replace(hour=9, minute=0)
            dates = [base - timedelta(days=29 - i) for i in range(30)]
            cum_pnl = []
            running = 0
            for _ in range(30):
                running += random.randint(-8000, 12000)
                cum_pnl.append(running)
            fig = create_equity_curve(dates, cum_pnl, f"{strat['name']} Equity Curve")
            st.plotly_chart(fig, use_container_width=True)

        with col_heatmap:
            conditions = ["bull", "bear", "range", "volatile"]
            weeks = [f"W{i}" for i in range(1, 9)]
            z = [[random.randint(-15000, 25000) for _ in weeks] for _ in conditions]
            fig = create_pnl_heatmap(weeks, conditions, z, "P&L by Market Condition")
            st.plotly_chart(fig, use_container_width=True)

        # Parameters
        st.markdown(
            '<div class="section-header"><h3>Parameters</h3></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size:0.85rem; color:#64748b; margin-bottom:12px;">{strat["description"]}</p>',
            unsafe_allow_html=True,
        )
        param_cols = st.columns(len(strat["params"]))
        for j, (k, v) in enumerate(strat["params"].items()):
            with param_cols[j]:
                st.text_input(k, value=str(v), key=f"param_{strat['name']}_{k}", disabled=True)

        # Recent trades for this strategy
        st.markdown(
            '<div class="section-header"><h3>Recent Trades</h3></div>',
            unsafe_allow_html=True,
        )
        trades_table(_demo_strategy_trades(strat["name"]))
