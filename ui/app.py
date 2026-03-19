"""KabuAI デイトレード -- メインアプリ"""

import streamlit as st

st.set_page_config(
    page_title="KabuAI デイトレード",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.theme import load_theme  # noqa: E402
load_theme()

with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        "<h2>KabuAI</h2>"
        "<p>デイトレード AI</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    pages = {
        "📊 ダッシュボード": "dashboard",
        "🎯 戦略": "strategies",
        "🧠 ナレッジ": "knowledge",
        "⚙️ 設定": "settings",
    }

    selected = st.radio("ナビ", list(pages.keys()), label_visibility="collapsed")
    page_key = pages[selected]

    st.markdown("---")
    st.markdown(
        '<span class="status-dot online"></span> ペーパートレード稼働中',
        unsafe_allow_html=True,
    )

if page_key == "dashboard":
    from ui.views.dashboard import render  # noqa: E402
elif page_key == "strategies":
    from ui.views.strategies import render  # noqa: E402
elif page_key == "knowledge":
    from ui.views.knowledge import render  # noqa: E402
elif page_key == "settings":
    from ui.views.settings import render  # noqa: E402
else:
    from ui.views.dashboard import render  # noqa: E402

render()
