"""ダッシュボード -- メイン画面"""

import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from ui.components.charts import create_equity_curve


def _demo_kpis():
    return dict(
        total_pnl=random.randint(-30000, 80000),
        win_rate=random.uniform(0.45, 0.72),
        profit_factor=random.uniform(0.8, 2.5),
        active_positions=random.randint(0, 5),
    )


def _demo_equity():
    base = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)
    dates, cum = [], []
    running = 0
    for i in range(30):
        dates.append(base - timedelta(days=29 - i))
        running += random.randint(-15000, 20000)
        cum.append(running)
    return dates, cum


def _demo_positions():
    tickers = ["7203", "6758", "9984", "8306", "6861"]
    names = ["トヨタ", "ソニー", "ソフトバンクG", "三菱UFJ", "キーエンス"]
    rows = []
    for i in range(random.randint(0, 4)):
        entry = random.uniform(1500, 8000)
        current = entry * random.uniform(0.97, 1.04)
        pnl = (current - entry) * 100
        rows.append({
            "銘柄": f"{tickers[i]} {names[i]}",
            "方向": random.choice(["ロング", "ショート"]),
            "取得価格": round(entry, 0),
            "現在値": round(current, 0),
            "損益": round(pnl, 0),
            "保有(分)": random.randint(10, 240),
        })
    return rows


def _demo_trades():
    strategies = ["gap_go", "vwap_reclaim", "orb", "rsi_reversal", "trend_follow"]
    rows = []
    for _ in range(10):
        pnl = random.randint(-20000, 35000)
        rows.append({
            "銘柄": f"{random.randint(1000,9999)}",
            "戦略": random.choice(strategies),
            "方向": random.choice(["ロング", "ショート"]),
            "損益": pnl,
            "保有(分)": random.randint(5, 300),
        })
    return rows


def render():
    st.markdown("## 📊 ダッシュボード")

    kpis = _demo_kpis()

    # KPI
    c1, c2, c3, c4 = st.columns(4)
    pnl = kpis["total_pnl"]
    sign = "+" if pnl >= 0 else ""
    c1.metric("本日損益", f"¥{sign}{pnl:,.0f}")
    c2.metric("勝率", f"{kpis['win_rate']:.1%}")
    c3.metric("PF", f"{kpis['profit_factor']:.2f}")
    c4.metric("ポジション", f"{kpis['active_positions']}")

    # 資産推移
    dates, cum_pnl = _demo_equity()
    fig = create_equity_curve(dates, cum_pnl, "資産推移（30日）")
    st.plotly_chart(fig, use_container_width=True)

    # ポジション & トレード
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 保有ポジション")
        positions = _demo_positions()
        if positions:
            df = pd.DataFrame(positions)
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("ポジションなし")

    with col2:
        st.markdown("#### 直近トレード")
        trades = _demo_trades()
        df = pd.DataFrame(trades)
        st.dataframe(df, hide_index=True, use_container_width=True)
