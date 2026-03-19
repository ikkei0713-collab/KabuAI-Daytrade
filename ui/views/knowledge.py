"""ナレッジページ -- 勝ち/負けパターンと改善候補"""

import random

import streamlit as st

from ui.components.cards import knowledge_card


def render():
    st.markdown("## 🧠 ナレッジ")

    tab1, tab2, tab3 = st.tabs(["勝ちパターン", "負けパターン", "改善候補"])

    with tab1:
        knowledge_card(
            "win_pattern",
            "VWAP奪還は強気相場 + 出来高2倍以上で勝率78%",
            confidence=0.78,
            date_str="2026-03-18",
            supporting_count=12,
        )
        knowledge_card(
            "win_pattern",
            "ORBは寄付30分以内 + ATR > 2%で最も効果的",
            confidence=0.72,
            date_str="2026-03-17",
            supporting_count=8,
        )
        knowledge_card(
            "win_pattern",
            "決算モメンタムは上方修正 + 小型株で期待値が最大",
            confidence=0.68,
            date_str="2026-03-15",
            supporting_count=5,
        )

    with tab2:
        knowledge_card(
            "loss_pattern",
            "ギャップフェードはレンジ相場で損切り連発。出来高低下時に注意",
            confidence=0.65,
            date_str="2026-03-18",
            supporting_count=7,
        )
        knowledge_card(
            "loss_pattern",
            "RSI逆張りはトレンド相場で逆行。地合い判定との連携必須",
            confidence=0.71,
            date_str="2026-03-16",
            supporting_count=9,
        )

    with tab3:
        st.markdown("##### 保留中の改善案")

        candidates = [
            ("VWAP奪還", "出来高閾値を1.5倍→2.0倍に引き上げ", "勝率+5%見込み"),
            ("ギャップフェード", "レンジ相場では無効化を提案", "損失-30%見込み"),
            ("RSI逆張り", "RSI閾値を15→10に変更", "偽シグナル削減"),
        ]

        for strategy, change, expected in candidates:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{strategy}**: {change}")
                    st.caption(f"期待効果: {expected}")
                with col2:
                    st.button("承認", key=f"approve_{strategy}", type="primary")
                with col3:
                    st.button("却下", key=f"reject_{strategy}")
                st.divider()
