"""Styled table components for the trading dashboard."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st
from streamlit.column_config import (
    Column,
    NumberColumn,
    TextColumn,
    DatetimeColumn,
)


def _pnl_color(val: float) -> str:
    """Return CSS color string based on P&L sign."""
    if val > 0:
        return "color: #00d4aa; font-weight: 600;"
    elif val < 0:
        return "color: #ff4757; font-weight: 600;"
    return "color: #94a3b8;"


def trades_table(trades: list[dict], max_rows: int = 20) -> None:
    """Display a formatted recent-trades table.

    Each trade dict should contain:
        ticker, strategy_name, direction, entry_price, exit_price,
        pnl, pnl_pct, holding_minutes, entry_time, exit_time, market_condition
    """
    if not trades:
        st.info("No trades to display.")
        return

    df = pd.DataFrame(trades[:max_rows])

    display_cols = {
        "ticker": "Ticker",
        "strategy_name": "Strategy",
        "direction": "Dir",
        "entry_price": "Entry",
        "exit_price": "Exit",
        "pnl": "P&L",
        "pnl_pct": "P&L %",
        "holding_minutes": "Hold (min)",
        "market_condition": "Condition",
    }
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].rename(columns=display_cols)

    column_config = {}
    if "P&L" in df_display.columns:
        column_config["P&L"] = NumberColumn("P&L", format="¥%,.0f")
    if "P&L %" in df_display.columns:
        column_config["P&L %"] = NumberColumn("P&L %", format="%.2f%%")
    if "Entry" in df_display.columns:
        column_config["Entry"] = NumberColumn("Entry", format="¥%,.1f")
    if "Exit" in df_display.columns:
        column_config["Exit"] = NumberColumn("Exit", format="¥%,.1f")

    st.dataframe(
        df_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        height=min(len(df_display) * 38 + 40, 600),
    )


def positions_table(positions: list[dict]) -> None:
    """Display active positions table.

    Each position dict should contain:
        ticker, strategy_name, direction, entry_price, current_price,
        unrealized_pnl, holding_minutes, stop_loss, take_profit
    """
    if not positions:
        st.info("No active positions.")
        return

    df = pd.DataFrame(positions)

    display_cols = {
        "ticker": "Ticker",
        "strategy_name": "Strategy",
        "direction": "Dir",
        "entry_price": "Entry",
        "current_price": "Current",
        "unrealized_pnl": "Unrealized P&L",
        "holding_minutes": "Hold (min)",
        "stop_loss": "SL",
        "take_profit": "TP",
    }
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].rename(columns=display_cols)

    # Compute P&L% if possible
    if "Unrealized P&L" in df_display.columns and "Entry" in df_display.columns:
        df_display["P&L %"] = (df_display["Unrealized P&L"] / (df_display["Entry"] * 100)) * 100

    column_config = {
        "Unrealized P&L": NumberColumn("Unrealized P&L", format="¥%,.0f"),
        "Entry": NumberColumn("Entry", format="¥%,.1f"),
        "Current": NumberColumn("Current", format="¥%,.1f"),
        "SL": NumberColumn("SL", format="¥%,.1f"),
        "TP": NumberColumn("TP", format="¥%,.1f"),
    }
    if "P&L %" in df_display.columns:
        column_config["P&L %"] = NumberColumn("P&L %", format="%.2f%%")

    st.dataframe(
        df_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )


def orders_table(orders: list[dict]) -> None:
    """Display orders table with status badges.

    Each order dict should contain:
        id, ticker, direction, order_type, price, quantity,
        status, strategy_name, timestamp
    """
    if not orders:
        st.info("No orders to display.")
        return

    df = pd.DataFrame(orders)

    display_cols = {
        "id": "Order ID",
        "ticker": "Ticker",
        "direction": "Dir",
        "order_type": "Type",
        "price": "Price",
        "quantity": "Qty",
        "status": "Status",
        "strategy_name": "Strategy",
        "timestamp": "Time",
    }
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].rename(columns=display_cols)

    column_config = {}
    if "Price" in df_display.columns:
        column_config["Price"] = NumberColumn("Price", format="¥%,.1f")
    if "Time" in df_display.columns:
        column_config["Time"] = DatetimeColumn("Time", format="YYYY-MM-DD HH:mm")

    st.dataframe(
        df_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )
