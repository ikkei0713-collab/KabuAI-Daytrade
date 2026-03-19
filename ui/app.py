"""KabuAI-Daytrade -- Main Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="KabuAI Daytrade",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply dark trading terminal theme
from ui.theme import load_theme  # noqa: E402

load_theme()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        "<h2>KabuAI Daytrade</h2>"
        "<p>Intelligent Trading Terminal</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    pages = {
        "\U0001f3e0 Dashboard": "dashboard",
        "\U0001f4ca Positions": "positions",
        "\U0001f4cb Orders": "orders",
        "\U0001f3af Strategies": "strategies",
        "\U0001f9e0 Knowledge": "knowledge",
        "\U0001f4c8 Analysis": "analysis",
        "\u2699\ufe0f Settings": "settings",
    }

    selected = st.radio(
        "Navigation",
        list(pages.keys()),
        label_visibility="collapsed",
    )
    page_key = pages[selected]

    st.markdown("---")
    st.markdown(
        '<span class="status-dot online"></span> Paper Trading Mode',
        unsafe_allow_html=True,
    )
    st.caption("v0.1.0 | Last refresh: live")

# ── Page Router ────────────────────────────────────────────────────────────────
if page_key == "dashboard":
    from ui.pages.dashboard import render  # noqa: E402
elif page_key == "positions":
    from ui.pages.positions import render  # noqa: E402
elif page_key == "orders":
    from ui.pages.orders import render  # noqa: E402
elif page_key == "strategies":
    from ui.pages.strategies import render  # noqa: E402
elif page_key == "knowledge":
    from ui.pages.knowledge import render  # noqa: E402
elif page_key == "analysis":
    from ui.pages.analysis import render  # noqa: E402
elif page_key == "settings":
    from ui.pages.settings import render  # noqa: E402
else:
    from ui.pages.dashboard import render  # noqa: E402

render()
