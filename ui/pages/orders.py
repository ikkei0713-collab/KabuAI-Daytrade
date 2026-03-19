"""Orders page -- pending and historical orders."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import streamlit as st

from ui.components.tables import orders_table


# ── Demo data ─────────────────────────────────────────────────────────────────

def _demo_pending_orders():
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb"]
    orders = []
    for _ in range(random.randint(0, 4)):
        orders.append(dict(
            id=f"ord_{random.randint(10000,99999)}",
            ticker=f"{random.randint(1000,9999)}.T",
            direction=random.choice(["long", "short"]),
            order_type=random.choice(["market", "limit"]),
            price=round(random.uniform(800, 6000), 1),
            quantity=100,
            status="pending",
            strategy_name=random.choice(strategies),
            timestamp=datetime.now().isoformat(),
        ))
    return orders


def _demo_order_history(n: int = 40):
    strategies = ["MomentumBreak", "GapReversion", "OrderBookImb", "EventDriven"]
    statuses = ["filled", "filled", "filled", "cancelled", "rejected"]
    orders = []
    for i in range(n):
        status = random.choice(statuses)
        orders.append(dict(
            id=f"ord_{random.randint(10000,99999)}",
            ticker=f"{random.randint(1000,9999)}.T",
            direction=random.choice(["long", "short"]),
            order_type=random.choice(["market", "limit"]),
            price=round(random.uniform(800, 6000), 1),
            quantity=100,
            status=status,
            strategy_name=random.choice(strategies),
            timestamp=(datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 8))).isoformat(),
            entry_reason=random.choice([
                "Momentum breakout above VWAP",
                "Gap reversion signal at support",
                "Order book imbalance > 3:1",
                "Event catalyst: earnings beat",
                "Volume surge with price confirmation",
            ]) if status == "filled" else "",
        ))
    return orders


# ── Render ────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown("## Orders")

    tab_pending, tab_history = st.tabs(["Pending Orders", "Order History"])

    # ── Pending Orders ────────────────────────────────────────────────────────
    with tab_pending:
        pending = _demo_pending_orders()

        if not pending:
            st.info("No pending orders at this time.")
        else:
            st.caption(f"{len(pending)} pending order(s)")
            orders_table(pending)

            # Order detail expanders
            for order in pending:
                dir_icon = "\u25b2" if order["direction"] == "long" else "\u25bc"
                with st.expander(
                    f"{order['ticker']}  |  {dir_icon} {order['direction'].upper()}  |  "
                    f"{order['order_type'].upper()} @ {order['price']:,.1f}"
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Order ID", order["id"])
                    c2.metric("Strategy", order["strategy_name"])
                    c3.metric("Quantity", f"{order['quantity']} shares")
                    c4.metric("Status", order["status"].upper())

                    col_cancel, col_spacer = st.columns([1, 3])
                    with col_cancel:
                        st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                        if st.button("Cancel Order", key=f"cancel_{order['id']}"):
                            st.warning(f"Order {order['id']} cancelled (paper).")
                        st.markdown("</div>", unsafe_allow_html=True)

    # ── Order History ─────────────────────────────────────────────────────────
    with tab_history:
        col_from, col_to, col_status = st.columns(3)
        with col_from:
            st.date_input(
                "From",
                value=datetime.now().date() - timedelta(days=30),
                key="order_from",
            )
        with col_to:
            st.date_input("To", value=datetime.now().date(), key="order_to")
        with col_status:
            status_filter = st.selectbox(
                "Status",
                ["All", "filled", "cancelled", "rejected"],
                key="order_status",
            )

        history = _demo_order_history(40)
        if status_filter != "All":
            history = [o for o in history if o["status"] == status_filter]

        st.caption(f"Showing {len(history)} orders")
        orders_table(history)

        # Order details
        st.markdown(
            '<div class="section-header"><h3>Order Details</h3></div>',
            unsafe_allow_html=True,
        )
        filled = [o for o in history if o["status"] == "filled" and o.get("entry_reason")]
        for order in filled[:5]:
            dir_color = "#00d4aa" if order["direction"] == "long" else "#ff4757"
            status_class = {
                "filled": "badge-filled",
                "cancelled": "badge-inactive",
                "rejected": "badge-rejected",
                "pending": "badge-pending",
            }.get(order["status"], "badge-inactive")
            st.markdown(
                f"""
                <div class="card" style="padding:14px 18px; margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <strong>{order['ticker']}</strong>
                            <span style="color:{dir_color}; margin-left:6px; font-size:0.85rem; font-weight:600;">
                                {order['direction'].upper()}
                            </span>
                            <span class="badge {status_class}" style="margin-left:8px;">
                                {order['status']}
                            </span>
                        </div>
                        <span style="color:#94a3b8; font-size:0.82rem;">{order['strategy_name']}</span>
                    </div>
                    <div style="margin-top:8px; font-size:0.85rem; color:#94a3b8;">
                        <strong style="color:#64748b;">Entry Reason:</strong> {order['entry_reason']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
