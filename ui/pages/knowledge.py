"""Knowledge & Learning page -- the brain of KabuAI."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import streamlit as st

from ui.components.cards import knowledge_card
from ui.components.charts import create_feature_importance


# ── Demo data ─────────────────────────────────────────────────────────────────

_WIN_PATTERNS = [
    dict(
        content="High win rate when entering long positions during the first 30 minutes of trading "
                "on days where Nikkei 225 futures are up >0.5% pre-market, combined with volume surge >2x average.",
        confidence=0.82,
        date="2026-03-18",
        supporting_count=23,
    ),
    dict(
        content="Gap reversion strategy performs best when the gap is between 1.5% and 3.0%, and the stock has "
                "strong support within 1% of the gap-fill target. Win rate: 72%.",
        confidence=0.78,
        date="2026-03-15",
        supporting_count=18,
    ),
    dict(
        content="Momentum breakouts with order book imbalance >4:1 on the bid side have a 68% win rate "
                "and average profit factor of 2.1.",
        confidence=0.71,
        date="2026-03-12",
        supporting_count=14,
    ),
]

_LOSS_PATTERNS = [
    dict(
        content="Significant losses occur when entering long positions during high VIX (>25) days "
                "in range-bound markets. Average loss: -8,500 per trade.",
        confidence=0.85,
        date="2026-03-17",
        supporting_count=19,
    ),
    dict(
        content="Short positions on ex-dividend dates consistently underperform. "
                "The dividend adjustment creates adverse price movements. Win rate drops to 31%.",
        confidence=0.73,
        date="2026-03-14",
        supporting_count=11,
    ),
    dict(
        content="Holding positions through lunch break (11:30-12:30) increases drawdown by 40% on average. "
                "Better to close before lunch in volatile markets.",
        confidence=0.68,
        date="2026-03-10",
        supporting_count=16,
    ),
]

_STRATEGY_INSIGHTS = [
    dict(
        content="MomentumBreak: Optimal performance in bull regime with sector rotation into tech. "
                "Reduce position size by 30% in range markets.",
        confidence=0.76,
        date="2026-03-16",
        supporting_count=28,
    ),
    dict(
        content="GapReversion: Adding VWAP confirmation improves win rate from 58% to 67%. "
                "The VWAP acts as dynamic support/resistance for gap fills.",
        confidence=0.81,
        date="2026-03-13",
        supporting_count=22,
    ),
]

_CANDIDATES = [
    dict(
        id="cand_001",
        strategy_name="MomentumBreak",
        proposed_changes={"volume_mult": "1.5 -> 2.0", "atr_mult": "2.0 -> 1.8"},
        reason="Backtested over 60 days: higher volume threshold filters false breakouts. "
               "Tighter ATR multiplier captures moves earlier.",
        expected_improvement="Win rate +4%, PF +0.15",
        status="pending",
        created_at="2026-03-18",
    ),
    dict(
        id="cand_002",
        strategy_name="GapReversion",
        proposed_changes={"add_vwap_confirm": "True", "reversion_target": "0.5 -> 0.6"},
        reason="VWAP confirmation reduces false entries. Higher reversion target captures more profit per trade.",
        expected_improvement="Win rate +5%, Avg P&L +800",
        status="pending",
        created_at="2026-03-17",
    ),
    dict(
        id="cand_003",
        strategy_name="OrderBookImb",
        proposed_changes={"imbalance_ratio": "3.0 -> 3.5", "decay_seconds": "30 -> 20"},
        reason="Higher imbalance ratio filters noise. Shorter decay window focuses on immediate impact.",
        expected_improvement="Win rate +3%, fewer false signals",
        status="approved",
        created_at="2026-03-14",
    ),
    dict(
        id="cand_004",
        strategy_name="EventDriven",
        proposed_changes={"hold_through": "False -> True for earnings beats only"},
        reason="Historical data shows holding through positive earnings surprises yields +12% avg return.",
        expected_improvement="Avg P&L turnaround to positive",
        status="rejected",
        created_at="2026-03-10",
    ),
]

_FEATURE_NAMES = [
    "volume_ratio", "momentum_5m", "spread_bps", "orderbook_imbalance",
    "vwap_distance", "atr_ratio", "sector_momentum", "nikkei_corr",
    "bid_depth", "trade_velocity",
]
_FEATURE_IMPORTANCES = [0.18, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.04]


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Knowledge & Learning")
    st.caption("AI-driven insights from trading activity | Auto-updated daily")

    tab_win, tab_loss, tab_insights, tab_improvements, tab_timeline = st.tabs([
        "Win Patterns",
        "Loss Patterns",
        "Strategy Insights",
        "Improvements",
        "Timeline",
    ])

    # ── Win Patterns ──────────────────────────────────────────────────────────
    with tab_win:
        st.markdown(
            '<div class="section-header"><h3>Discovered Win Patterns</h3></div>',
            unsafe_allow_html=True,
        )

        col_cards, col_chart = st.columns([3, 2])
        with col_cards:
            for wp in _WIN_PATTERNS:
                knowledge_card(
                    category="win_pattern",
                    content=wp["content"],
                    confidence=wp["confidence"],
                    date_str=wp["date"],
                    supporting_count=wp["supporting_count"],
                )

        with col_chart:
            st.markdown("**Feature Importance for Wins**")
            fig = create_feature_importance(
                _FEATURE_NAMES,
                _FEATURE_IMPORTANCES,
                "Features Correlated with Wins",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Common win conditions
        st.markdown(
            '<div class="section-header"><h3>Common Win Conditions</h3></div>',
            unsafe_allow_html=True,
        )
        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Market Regime", "Bull", delta="68% of wins")
        wc2.metric("Best Entry Time", "9:05-9:30", delta="Morning edge")
        wc3.metric("Volume Condition", ">1.5x avg", delta="Confirmation")
        wc4.metric("Momentum", "Positive 5m", delta="Trend aligned")

    # ── Loss Patterns ─────────────────────────────────────────────────────────
    with tab_loss:
        st.markdown(
            '<div class="section-header"><h3>Discovered Loss Patterns</h3></div>',
            unsafe_allow_html=True,
        )

        col_cards2, col_chart2 = st.columns([3, 2])
        with col_cards2:
            for lp in _LOSS_PATTERNS:
                knowledge_card(
                    category="loss_pattern",
                    content=lp["content"],
                    confidence=lp["confidence"],
                    date_str=lp["date"],
                    supporting_count=lp["supporting_count"],
                )

        with col_chart2:
            loss_features = [
                "vix_level", "holding_time", "spread_bps", "lunch_crossover",
                "ex_dividend", "low_volume", "counter_trend", "late_entry",
            ]
            loss_importances = [0.22, 0.18, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07]
            st.markdown("**Feature Importance for Losses**")
            fig = create_feature_importance(
                loss_features,
                loss_importances,
                "Features Correlated with Losses",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Common loss conditions
        st.markdown(
            '<div class="section-header"><h3>Common Loss Conditions</h3></div>',
            unsafe_allow_html=True,
        )
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Market Regime", "Range", delta="45% of losses")
        lc2.metric("Worst Entry Time", "14:00-14:30", delta="Late day traps")
        lc3.metric("VIX Threshold", ">25", delta="High volatility")
        lc4.metric("Holding Risk", ">120 min", delta="Overexposure")

    # ── Strategy Insights ─────────────────────────────────────────────────────
    with tab_insights:
        st.markdown(
            '<div class="section-header"><h3>Per-Strategy Optimal Conditions</h3></div>',
            unsafe_allow_html=True,
        )
        for si in _STRATEGY_INSIGHTS:
            knowledge_card(
                category="strategy_insight",
                content=si["content"],
                confidence=si["confidence"],
                date_str=si["date"],
                supporting_count=si["supporting_count"],
            )

        # Strategy-condition matrix
        st.markdown("")
        st.markdown(
            '<div class="section-header"><h3>Strategy x Condition Win Rate Matrix</h3></div>',
            unsafe_allow_html=True,
        )
        import pandas as pd

        matrix_data = {
            "Strategy": ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"],
            "Bull": ["68%", "52%", "55%", "61%"],
            "Bear": ["42%", "48%", "58%", "35%"],
            "Range": ["45%", "71%", "51%", "44%"],
            "Volatile": ["53%", "44%", "62%", "38%"],
        }
        df_matrix = pd.DataFrame(matrix_data)
        st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    # ── Improvements ──────────────────────────────────────────────────────────
    with tab_improvements:
        st.markdown(
            '<div class="section-header"><h3>Improvement Candidates</h3></div>',
            unsafe_allow_html=True,
        )

        for cand in _CANDIDATES:
            status_class = {
                "pending": "badge-pending",
                "approved": "badge-active",
                "rejected": "badge-rejected",
                "applied": "badge-filled",
            }.get(cand["status"], "badge-inactive")

            changes_html = " | ".join(
                f"<code>{k}</code>: {v}" for k, v in cand["proposed_changes"].items()
            )

            st.markdown(
                f"""
                <div class="card" style="border-left:3px solid {'#f59e0b' if cand['status'] == 'pending' else '#00d4aa' if cand['status'] in ('approved','applied') else '#ff4757'};">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <strong>{cand['strategy_name']}</strong>
                        <span class="badge {status_class}">{cand['status']}</span>
                    </div>
                    <div style="font-size:0.85rem; margin-bottom:8px;">
                        <span style="color:#94a3b8;">Changes:</span> {changes_html}
                    </div>
                    <p style="font-size:0.88rem; color:#e2e8f0; margin:0 0 6px 0;">{cand['reason']}</p>
                    <div style="font-size:0.82rem; color:#64748b;">
                        Expected: <strong style="color:#00d4aa;">{cand['expected_improvement']}</strong>
                        &nbsp;|&nbsp; Created: {cand['created_at']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if cand["status"] == "pending":
                bc1, bc2, bc3 = st.columns([1, 1, 4])
                with bc1:
                    st.markdown('<div class="success-btn">', unsafe_allow_html=True)
                    if st.button("Approve", key=f"approve_{cand['id']}"):
                        st.success(f"Candidate {cand['id']} approved!")
                    st.markdown("</div>", unsafe_allow_html=True)
                with bc2:
                    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                    if st.button("Reject", key=f"reject_{cand['id']}"):
                        st.warning(f"Candidate {cand['id']} rejected.")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("")

    # ── Timeline ──────────────────────────────────────────────────────────────
    with tab_timeline:
        st.markdown(
            '<div class="section-header"><h3>Knowledge Timeline</h3></div>',
            unsafe_allow_html=True,
        )

        all_entries = []
        for wp in _WIN_PATTERNS:
            all_entries.append(("win_pattern", wp["date"], wp["content"], wp["confidence"], wp["supporting_count"]))
        for lp in _LOSS_PATTERNS:
            all_entries.append(("loss_pattern", lp["date"], lp["content"], lp["confidence"], lp["supporting_count"]))
        for si in _STRATEGY_INSIGHTS:
            all_entries.append(("strategy_insight", si["date"], si["content"], si["confidence"], si["supporting_count"]))

        # Add some market insights
        all_entries.append((
            "market_insight", "2026-03-19",
            "Correlation between USD/JPY movements and Nikkei 225 has increased to 0.78 this week. "
            "Consider USD/JPY as a leading indicator for morning entries.",
            0.72, 31,
        ))
        all_entries.append((
            "market_insight", "2026-03-11",
            "Sector rotation from financials to tech observed. Momentum strategies on tech stocks "
            "outperforming by 2.3x compared to broad market application.",
            0.69, 15,
        ))

        # Sort by date descending
        all_entries.sort(key=lambda x: x[1], reverse=True)

        for cat, date_str, content, conf, count in all_entries:
            knowledge_card(
                category=cat,
                content=content,
                confidence=conf,
                date_str=date_str,
                supporting_count=count,
            )
