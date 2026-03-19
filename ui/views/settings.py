"""設定ページ"""

import streamlit as st


def render():
    st.markdown("## ⚙️ 設定")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### トレード設定")
        st.number_input("総資金 (円)", value=3_000_000, step=100_000, disabled=True)
        st.number_input("最大ポジション数", value=5, step=1, disabled=True)
        st.number_input("1ポジション上限 (円)", value=500_000, step=50_000, disabled=True)
        st.number_input("日次損失上限 (円)", value=-50_000, step=10_000, disabled=True)
        st.number_input("最大保有時間 (分)", value=360, step=30, disabled=True)

    with col2:
        st.markdown("#### 安全設定")
        st.toggle("本番取引禁止", value=True, disabled=True)
        st.toggle("ペーパートレード", value=True, disabled=True)
        st.text_input("マーケット開場", value="09:00", disabled=True)
        st.text_input("強制決済時刻", value="14:50", disabled=True)
        st.text_input("マーケット閉場", value="15:00", disabled=True)

    st.divider()
    st.markdown("#### データソース")
    st.success("J-Quants API: 接続済み (V2 APIキー認証)")
    st.warning("立花証券API: 未接続 (将来実装)")
