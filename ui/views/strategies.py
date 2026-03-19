"""戦略ページ -- 全16戦略の成績一覧"""

import random

import pandas as pd
import streamlit as st


def _demo_strategies():
    names = [
        ("ギャップGo", "ギャップ"),
        ("ギャップフェード", "ギャップ"),
        ("寄付ドライブ", "寄付"),
        ("ORB", "寄付"),
        ("VWAP奪還", "モメンタム"),
        ("VWAPバウンス", "モメンタム"),
        ("トレンドフォロー", "モメンタム"),
        ("過伸び反転", "逆張り"),
        ("RSI逆張り", "逆張り"),
        ("急落リバウンド", "逆張り"),
        ("板偏り", "板需給"),
        ("大口吸収", "板需給"),
        ("スプレッド縮小", "板需給"),
        ("TDnetイベント", "イベント"),
        ("決算モメンタム", "イベント"),
        ("材料初動", "イベント"),
    ]
    rows = []
    for name, cat in names:
        wr = random.uniform(0.35, 0.75)
        rows.append({
            "カテゴリ": cat,
            "戦略": name,
            "トレード数": random.randint(5, 80),
            "勝率": round(wr, 3),
            "PF": round(random.uniform(0.6, 3.0), 2),
            "平均損益": random.randint(-5000, 15000),
            "保有(分)": random.randint(10, 180),
            "状態": "有効" if random.random() > 0.15 else "無効",
        })
    return rows


def render():
    st.markdown("## 🎯 戦略一覧")

    data = _demo_strategies()
    df = pd.DataFrame(data)

    cats = ["全て"] + sorted(df["カテゴリ"].unique().tolist())
    selected = st.selectbox("カテゴリ", cats, label_visibility="collapsed")
    if selected != "全て":
        df = df[df["カテゴリ"] == selected]

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "勝率": st.column_config.ProgressColumn("勝率", min_value=0, max_value=1, format="%.1%%"),
            "平均損益": st.column_config.NumberColumn("平均損益", format="¥%d"),
        },
    )

    st.markdown("#### 勝率ランキング")
    top = pd.DataFrame(data).sort_values("勝率", ascending=False).head(5)
    st.bar_chart(top.set_index("戦略")["勝率"])
