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
        st.info("表示するトレードがありません。")
        return

    df = pd.DataFrame(trades[:max_rows])

    display_cols = {
        "ticker": "銘柄",
        "strategy_name": "戦略",
        "direction": "方向",
        "entry_price": "エントリー",
        "exit_price": "エグジット",
        "pnl": "損益",
        "pnl_pct": "損益 %",
        "holding_minutes": "保有 (分)",
        "market_condition": "市場状況",
    }
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].rename(columns=display_cols)

    column_config = {}
    if "損益" in df_display.columns:
        column_config["損益"] = NumberColumn("損益", format="¥%,.0f")
    if "損益 %" in df_display.columns:
        column_config["損益 %"] = NumberColumn("損益 %", format="%.2f%%")
    if "エントリー" in df_display.columns:
        column_config["エントリー"] = NumberColumn("エントリー", format="¥%,.1f")
    if "エグジット" in df_display.columns:
        column_config["エグジット"] = NumberColumn("エグジット", format="¥%,.1f")

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
        st.info("アクティブなポジションがありません。")
        return

    df = pd.DataFrame(positions)

    display_cols = {
        "ticker": "銘柄",
        "strategy_name": "戦略",
        "direction": "方向",
        "entry_price": "エントリー",
        "current_price": "現在値",
        "unrealized_pnl": "未実現損益",
        "holding_minutes": "保有 (分)",
        "stop_loss": "SL",
        "take_profit": "TP",
    }
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].rename(columns=display_cols)

    # Compute P&L% if possible
    if "未実現損益" in df_display.columns and "エントリー" in df_display.columns:
        df_display["損益 %"] = (df_display["未実現損益"] / (df_display["エントリー"] * 100)) * 100

    column_config = {
        "未実現損益": NumberColumn("未実現損益", format="¥%,.0f"),
        "エントリー": NumberColumn("エントリー", format="¥%,.1f"),
        "現在値": NumberColumn("現在値", format="¥%,.1f"),
        "SL": NumberColumn("SL", format="¥%,.1f"),
        "TP": NumberColumn("TP", format="¥%,.1f"),
    }
    if "損益 %" in df_display.columns:
        column_config["損益 %"] = NumberColumn("損益 %", format="%.2f%%")

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
        st.info("表示する注文がありません。")
        return

    df = pd.DataFrame(orders)

    display_cols = {
        "id": "注文ID",
        "ticker": "銘柄",
        "direction": "方向",
        "order_type": "種別",
        "price": "価格",
        "quantity": "数量",
        "status": "ステータス",
        "strategy_name": "戦略",
        "timestamp": "時刻",
    }
    available = [c for c in display_cols if c in df.columns]
    df_display = df[available].rename(columns=display_cols)

    column_config = {}
    if "価格" in df_display.columns:
        column_config["価格"] = NumberColumn("価格", format="¥%,.1f")
    if "時刻" in df_display.columns:
        column_config["時刻"] = DatetimeColumn("時刻", format="YYYY-MM-DD HH:mm")

    st.dataframe(
        df_display,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
    )
