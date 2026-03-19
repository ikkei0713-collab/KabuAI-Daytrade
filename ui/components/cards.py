"""Reusable card components for the trading dashboard."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = "normal",
) -> None:
    """Render a styled KPI metric using st.metric.

    Parameters
    ----------
    title : str
        Label shown above the number.
    value : str
        The main large number / text.
    delta : str | None
        Optional delta text shown below.
    delta_color : str
        One of ``"normal"``, ``"inverse"``, ``"off"``.
    """
    st.metric(label=title, value=value, delta=delta, delta_color=delta_color)


def strategy_card(
    name: str,
    is_active: bool = True,
    win_rate: float = 0.0,
    profit_factor: float = 0.0,
    total_trades: int = 0,
    best_condition: str = "N/A",
    avg_pnl: float = 0.0,
) -> None:
    """Render a strategy summary card using markdown + container."""
    status_badge = (
        '<span class="badge badge-active">Active</span>'
        if is_active
        else '<span class="badge badge-inactive">Inactive</span>'
    )
    pnl_class = "profit" if avg_pnl >= 0 else "loss"
    pnl_sign = "+" if avg_pnl >= 0 else ""

    html = f"""
    <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <strong style="font-size:1.05rem;">{name}</strong>
            {status_badge}
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.88rem;">
            <div>
                <span style="color:#94a3b8;">Win Rate</span><br>
                <strong>{win_rate:.1%}</strong>
            </div>
            <div>
                <span style="color:#94a3b8;">Profit Factor</span><br>
                <strong>{profit_factor:.2f}</strong>
            </div>
            <div>
                <span style="color:#94a3b8;">Total Trades</span><br>
                <strong>{total_trades}</strong>
            </div>
            <div>
                <span style="color:#94a3b8;">Avg P&L</span><br>
                <strong class="{pnl_class}">{pnl_sign}{avg_pnl:,.0f}</strong>
            </div>
        </div>
        <div style="margin-top:10px; font-size:0.8rem; color:#64748b;">
            Best condition: <strong style="color:#e2e8f0;">{best_condition}</strong>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def trade_card(
    ticker: str,
    strategy: str,
    direction: str,
    pnl: float,
    pnl_pct: float,
    entry_reason: str = "",
    holding_minutes: int = 0,
) -> None:
    """Render a compact trade summary card."""
    pnl_class = "profit" if pnl >= 0 else "loss"
    pnl_sign = "+" if pnl >= 0 else ""
    dir_icon = "\u25b2" if direction == "long" else "\u25bc"

    html = f"""
    <div class="card" style="padding:14px 18px;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <strong>{ticker}</strong>
                <span style="color:#94a3b8; margin-left:8px; font-size:0.82rem;">
                    {dir_icon} {direction.upper()} | {strategy}
                </span>
            </div>
            <div style="text-align:right;">
                <span class="card-value {pnl_class}" style="font-size:1.1rem;">
                    {pnl_sign}{pnl:,.0f}
                </span>
                <span style="font-size:0.82rem; color:#94a3b8; margin-left:6px;">
                    ({pnl_sign}{pnl_pct:.2f}%)
                </span>
            </div>
        </div>
        <div style="font-size:0.78rem; color:#64748b; margin-top:4px;">
            {f"Reason: {entry_reason} | " if entry_reason else ""}Held {holding_minutes} min
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def knowledge_card(
    category: str,
    content: str,
    confidence: float = 0.5,
    date_str: str = "",
    supporting_count: int = 0,
) -> None:
    """Render a knowledge / insight card."""
    cat_colors = {
        "win_pattern": ("#00d4aa", "Win Pattern"),
        "loss_pattern": ("#ff4757", "Loss Pattern"),
        "strategy_insight": ("#6366f1", "Strategy Insight"),
        "market_insight": ("#3b82f6", "Market Insight"),
    }
    color, label = cat_colors.get(category, ("#94a3b8", category))

    conf_bar_width = int(confidence * 100)

    html = f"""
    <div class="card" style="border-left:3px solid {color};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="color:{color}; font-weight:600; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.06em;">
                {label}
            </span>
            <span style="color:#64748b; font-size:0.75rem;">{date_str}</span>
        </div>
        <p style="margin:0 0 10px 0; font-size:0.92rem; line-height:1.5;">{content}</p>
        <div style="display:flex; justify-content:space-between; align-items:center; font-size:0.78rem; color:#64748b;">
            <span>Confidence</span>
            <span>{supporting_count} supporting trade{"s" if supporting_count != 1 else ""}</span>
        </div>
        <div style="background:#2a2a4a; border-radius:4px; height:4px; margin-top:4px;">
            <div style="background:{color}; width:{conf_bar_width}%; height:100%; border-radius:4px;"></div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def system_status_item(label: str, status: str = "online", detail: str = "") -> None:
    """Render a single system status row."""
    html = f"""
    <div style="display:flex; align-items:center; padding:6px 0; font-size:0.88rem;">
        <span class="status-dot {status}"></span>
        <span style="color:#e2e8f0; flex:1;">{label}</span>
        <span style="color:#64748b; font-size:0.8rem;">{detail}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
