"""Settings page -- system configuration and status."""

from __future__ import annotations

import streamlit as st

from ui.components.cards import system_status_item


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Settings")

    tab_trading, tab_safety, tab_strategies, tab_data, tab_system = st.tabs([
        "Trading Parameters",
        "Safety Settings",
        "Strategy Control",
        "Data Sources",
        "System Info",
    ])

    # ── Trading Parameters ────────────────────────────────────────────────────
    with tab_trading:
        st.markdown(
            '<div class="section-header"><h3>Trading Parameters</h3></div>',
            unsafe_allow_html=True,
        )
        st.caption("Core trading configuration. Changes apply to paper trading only.")

        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "Total Capital (JPY)",
                value=3_000_000,
                step=100_000,
                format="%d",
                key="total_capital",
                help="Total available capital for trading",
            )
            st.number_input(
                "Max Positions",
                value=5,
                min_value=1,
                max_value=20,
                key="max_positions",
                help="Maximum number of simultaneous positions",
            )
            st.number_input(
                "Max Position Size (JPY)",
                value=500_000,
                step=50_000,
                format="%d",
                key="max_pos_size",
                help="Maximum capital per single position",
            )

        with c2:
            st.number_input(
                "Min Confidence Threshold",
                value=0.60,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                key="min_confidence",
                help="Minimum signal confidence to enter a trade",
            )
            st.number_input(
                "Strategy Score Threshold",
                value=0.50,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                key="strat_threshold",
                help="Minimum strategy performance score to remain active",
            )
            st.number_input(
                "Max Holding Time (min)",
                value=360,
                min_value=10,
                max_value=600,
                step=30,
                key="max_holding",
                help="Maximum holding time before forced close",
            )

        st.markdown("")
        st.markdown(
            '<div class="section-header"><h3>Market Hours (JST)</h3></div>',
            unsafe_allow_html=True,
        )
        h1, h2, h3, h4 = st.columns(4)
        h1.text_input("Pre-Market Scan", value="08:30", disabled=True)
        h2.text_input("Market Open", value="09:00", disabled=True)
        h3.text_input("Force Close", value="14:50", disabled=True)
        h4.text_input("Market Close", value="15:00", disabled=True)

    # ── Safety Settings ───────────────────────────────────────────────────────
    with tab_safety:
        st.markdown(
            '<div class="section-header"><h3>Safety Guards</h3></div>',
            unsafe_allow_html=True,
        )
        st.caption("All safety settings are read-only. These guards cannot be disabled.")

        st.markdown(
            """
            <div class="card" style="border-left:3px solid #00d4aa;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                    <span class="status-dot online"></span>
                    <strong>Paper Trading Mode</strong>
                    <span class="badge badge-active">ENFORCED</span>
                </div>
                <p style="color:#94a3b8; font-size:0.88rem; margin:0;">
                    Live trading is disabled. All orders are simulated. ALLOW_LIVE_TRADING = False.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        safety_items = [
            ("Daily Loss Limit", "-50,000 JPY", "System halts all trading when daily P&L reaches this threshold."),
            ("Max Position Size", "500,000 JPY", "No single position can exceed this value."),
            ("Max Simultaneous Positions", "5", "Hard cap on concurrent open positions."),
            ("Force Close Time", "14:50 JST", "All positions are force-closed 10 minutes before market close."),
            ("Max Holding Time", "360 min (6h)", "Positions held longer than this are automatically closed."),
        ]

        for label, value, desc in safety_items:
            st.markdown(
                f"""
                <div class="card" style="padding:14px 18px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <strong>{label}</strong>
                        <span style="color:#00d4aa; font-weight:600;">{value}</span>
                    </div>
                    <p style="color:#64748b; font-size:0.82rem; margin:4px 0 0 0;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Strategy Control ──────────────────────────────────────────────────────
    with tab_strategies:
        st.markdown(
            '<div class="section-header"><h3>Strategy Enable / Disable</h3></div>',
            unsafe_allow_html=True,
        )
        st.caption("Toggle strategies on or off. Changes take effect on next trading session.")

        strategies = [
            ("MomentumBreak", True, "Breakout on strong momentum with volume confirmation"),
            ("GapReversion", True, "Mean reversion on opening gaps"),
            ("OrderBookImb", True, "Order book imbalance trading"),
            ("EventDriven", False, "Corporate event-driven trading"),
        ]

        for name, default_active, desc in strategies:
            col_toggle, col_desc = st.columns([1, 3])
            with col_toggle:
                active = st.toggle(name, value=default_active, key=f"settings_toggle_{name}")
            with col_desc:
                status = "Active" if active else "Inactive"
                badge_class = "badge-active" if active else "badge-inactive"
                st.markdown(
                    f'<span class="badge {badge_class}">{status}</span> '
                    f'<span style="color:#94a3b8; font-size:0.85rem; margin-left:8px;">{desc}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown("")

    # ── Data Sources ──────────────────────────────────────────────────────────
    with tab_data:
        st.markdown(
            '<div class="section-header"><h3>Data Source Status</h3></div>',
            unsafe_allow_html=True,
        )

        system_status_item("J-Quants API", "online", "Connected | Token valid")
        system_status_item("TDnet Scraper", "online", "Last scrape: 5 min ago")
        system_status_item("Tachibana Securities", "offline", "Not configured")
        system_status_item("Market Data Feed", "online", "Real-time (delayed 20min)")
        system_status_item("Historical DB", "online", "SQLite | 1.2 GB")

        st.markdown("")
        st.markdown(
            '<div class="section-header"><h3>API Configuration</h3></div>',
            unsafe_allow_html=True,
        )
        st.text_input("J-Quants Email", value="configured", disabled=True, type="password")
        st.text_input("J-Quants Password", value="configured", disabled=True, type="password")
        st.text_input("Tachibana API URL", value="Not set", disabled=True)

    # ── System Info ───────────────────────────────────────────────────────────
    with tab_system:
        st.markdown(
            '<div class="section-header"><h3>System Information</h3></div>',
            unsafe_allow_html=True,
        )

        info_items = [
            ("Version", "0.1.0"),
            ("Python", "3.11"),
            ("Streamlit", "1.38+"),
            ("Database", "SQLite (kabuai.db)"),
            ("Log Directory", "~/dev/KabuAI-Daytrade/logs/"),
            ("Config", "pydantic-settings (.env)"),
            ("Trading Mode", "Paper Trading"),
            ("Strategies Loaded", "4"),
            ("Knowledge Entries", "47"),
        ]

        for label, value in info_items:
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:10px 0; border-bottom:1px solid #2a2a4a; font-size:0.9rem;">
                    <span style="color:#94a3b8;">{label}</span>
                    <strong style="color:#e2e8f0;">{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown(
            '<div class="section-header"><h3>Disk Usage</h3></div>',
            unsafe_allow_html=True,
        )
        col_db, col_logs = st.columns(2)
        with col_db:
            st.metric("Database Size", "1.2 GB")
        with col_logs:
            st.metric("Log Files", "340 MB")
