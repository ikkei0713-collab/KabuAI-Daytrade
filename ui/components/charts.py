"""Reusable Plotly chart components with dark trading terminal theme."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

# ── Shared layout defaults ────────────────────────────────────────────────────

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="#0e1117",
    plot_bgcolor="#1a1a2e",
    font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto", color="#e2e8f0"),
    margin=dict(l=40, r=24, t=48, b=32),
    xaxis=dict(gridcolor="#2a2a4a", zerolinecolor="#2a2a4a"),
    yaxis=dict(gridcolor="#2a2a4a", zerolinecolor="#2a2a4a"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#16213e", font_size=12, font_color="#e2e8f0"),
)

PROFIT_COLOR = "#00d4aa"
LOSS_COLOR = "#ff4757"
ACCENT_COLOR = "#6366f1"
INFO_COLOR = "#3b82f6"
WARNING_COLOR = "#f59e0b"

STRATEGY_COLORS = [
    "#6366f1", "#00d4aa", "#3b82f6", "#f59e0b",
    "#ec4899", "#8b5cf6", "#14b8a6", "#f97316",
]


def _apply_defaults(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(**_LAYOUT_DEFAULTS, title=dict(text=title, x=0.02, font_size=15), height=height)
    return fig


# ── Candlestick ───────────────────────────────────────────────────────────────

def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Price Chart",
    height: int = 480,
) -> go.Figure:
    """Create a candlestick chart. Expects columns: Open, High, Low, Close and a DatetimeIndex."""
    fig = go.Figure(
        data=go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=PROFIT_COLOR,
            decreasing_line_color=LOSS_COLOR,
            increasing_fillcolor=PROFIT_COLOR,
            decreasing_fillcolor=LOSS_COLOR,
        )
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return _apply_defaults(fig, title, height)


# ── Equity Curve ──────────────────────────────────────────────────────────────

def create_equity_curve(
    dates: Sequence,
    cumulative_pnl: Sequence[float],
    title: str = "Equity Curve",
    height: int = 380,
) -> go.Figure:
    """Line chart of cumulative P&L over time."""
    colors = [PROFIT_COLOR if v >= 0 else LOSS_COLOR for v in cumulative_pnl]
    last_color = colors[-1] if colors else PROFIT_COLOR

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(cumulative_pnl),
            mode="lines",
            line=dict(color=last_color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({','.join(str(int(last_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))}, 0.08)",
            hovertemplate="Date: %{x}<br>Cumulative P&L: %{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b", line_width=1)
    return _apply_defaults(fig, title, height)


# ── Strategy Comparison ──────────────────────────────────────────────────────

def create_strategy_comparison(
    strategy_names: Sequence[str],
    pnl_values: Sequence[float],
    title: str = "Strategy P&L Comparison",
    height: int = 380,
) -> go.Figure:
    """Horizontal bar chart comparing strategy P&L."""
    colors = [PROFIT_COLOR if v >= 0 else LOSS_COLOR for v in pnl_values]
    fig = go.Figure(
        go.Bar(
            x=list(pnl_values),
            y=list(strategy_names),
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return _apply_defaults(fig, title, height)


# ── Strategy Performance Donut ────────────────────────────────────────────────

def create_strategy_donut(
    labels: Sequence[str],
    values: Sequence[float],
    title: str = "Strategy Allocation",
    height: int = 380,
) -> go.Figure:
    """Donut chart for strategy performance share."""
    fig = go.Figure(
        go.Pie(
            labels=list(labels),
            values=list(values),
            hole=0.55,
            marker=dict(colors=STRATEGY_COLORS[: len(labels)]),
            textinfo="label+percent",
            textfont=dict(size=11),
            hovertemplate="%{label}: %{value:,.0f} (%{percent})<extra></extra>",
        )
    )
    return _apply_defaults(fig, title, height)


# ── P&L Heatmap ──────────────────────────────────────────────────────────────

def create_pnl_heatmap(
    dates: Sequence,
    categories: Sequence[str],
    z_values: list[list[float]],
    title: str = "P&L Heatmap",
    height: int = 380,
) -> go.Figure:
    """Heatmap of P&L by date and category (strategy or condition)."""
    fig = go.Figure(
        go.Heatmap(
            x=list(dates),
            y=list(categories),
            z=z_values,
            colorscale=[
                [0.0, LOSS_COLOR],
                [0.5, "#1a1a2e"],
                [1.0, PROFIT_COLOR],
            ],
            hovertemplate="Date: %{x}<br>%{y}: %{z:,.0f}<extra></extra>",
        )
    )
    return _apply_defaults(fig, title, height)


# ── Feature Importance ────────────────────────────────────────────────────────

def create_feature_importance(
    features: Sequence[str],
    importances: Sequence[float],
    title: str = "Feature Importance",
    height: int = 380,
) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    sorted_pairs = sorted(zip(importances, features))
    s_importances, s_features = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig = go.Figure(
        go.Bar(
            x=list(s_importances),
            y=list(s_features),
            orientation="h",
            marker=dict(
                color=list(s_importances),
                colorscale=[[0, INFO_COLOR], [1, ACCENT_COLOR]],
            ),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    return _apply_defaults(fig, title, height)


# ── Win/Loss Distribution ────────────────────────────────────────────────────

def create_win_loss_distribution(
    pnl_values: Sequence[float],
    title: str = "Trade P&L Distribution",
    height: int = 350,
) -> go.Figure:
    """Histogram of individual trade P&L values."""
    wins = [v for v in pnl_values if v >= 0]
    losses = [v for v in pnl_values if v < 0]

    fig = go.Figure()
    if losses:
        fig.add_trace(
            go.Histogram(
                x=losses, name="Losses", marker_color=LOSS_COLOR,
                opacity=0.8, nbinsx=20,
            )
        )
    if wins:
        fig.add_trace(
            go.Histogram(
                x=wins, name="Wins", marker_color=PROFIT_COLOR,
                opacity=0.8, nbinsx=20,
            )
        )
    fig.update_layout(barmode="overlay", xaxis_title="P&L (JPY)", yaxis_title="Count")
    return _apply_defaults(fig, title, height)


# ── Holding Time Distribution ─────────────────────────────────────────────────

def create_holding_time_distribution(
    holding_minutes: Sequence[int],
    title: str = "Holding Time Distribution",
    height: int = 350,
) -> go.Figure:
    """Histogram of holding durations."""
    fig = go.Figure(
        go.Histogram(
            x=list(holding_minutes),
            marker_color=ACCENT_COLOR,
            opacity=0.85,
            nbinsx=25,
        )
    )
    fig.update_layout(xaxis_title="Minutes", yaxis_title="Count")
    return _apply_defaults(fig, title, height)


# ── Win Rate Trend ────────────────────────────────────────────────────────────

def create_win_rate_trend(
    dates: Sequence,
    win_rates: Sequence[float],
    title: str = "Win Rate Trend (Rolling)",
    height: int = 350,
) -> go.Figure:
    """Line chart of rolling win rate."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(dates),
            y=list(win_rates),
            mode="lines+markers",
            line=dict(color=ACCENT_COLOR, width=2),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Win Rate: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="#64748b", line_width=1)
    fig.update_layout(yaxis=dict(tickformat=".0%", range=[0, 1]))
    return _apply_defaults(fig, title, height)


# ── Trade Time of Day Distribution ────────────────────────────────────────────

def create_trade_time_distribution(
    hours: Sequence[int],
    counts: Sequence[int],
    title: str = "Trades by Time of Day",
    height: int = 350,
) -> go.Figure:
    """Bar chart of trade count by hour."""
    fig = go.Figure(
        go.Bar(
            x=list(hours),
            y=list(counts),
            marker_color=INFO_COLOR,
            opacity=0.85,
            hovertemplate="Hour: %{x}:00<br>Trades: %{y}<extra></extra>",
        )
    )
    fig.update_layout(xaxis_title="Hour (JST)", yaxis_title="Trade Count")
    return _apply_defaults(fig, title, height)
