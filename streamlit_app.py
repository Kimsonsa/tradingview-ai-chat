"""
TradeAI — TradingView 차트 + AI 채팅 Streamlit 앱
전체 UI를 HTML 컴포넌트로 렌더링
"""

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(
    page_title="TradeAI — 실시간 차트 AI 분석",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 모든 Streamlit 기본 UI 숨기기
st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
    header[data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; padding: 0 !important; background: transparent !important; }
    .stMainBlockContainer { padding: 0 !important; max-width: 100% !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    section[data-testid="stMain"] { padding: 0 !important; }
    footer { display: none !important; }
    .stDeployButton { display: none !important; }
    iframe { border: none !important; }
</style>
""", unsafe_allow_html=True)

# HTML 템플릿 로드
template_dir = Path(__file__).parent / "templates"

def load_file(name):
    return (template_dir / name).read_text(encoding="utf-8")

html = load_file("app.html")
css = load_file("styles.css")
app_js = load_file("app.js")
chat_js = load_file("chat.js")

# 템플릿에 CSS/JS 삽입
full_html = html.replace("/* __STYLES__ */", css)
full_html = full_html.replace("/* __CHAT_JS__ */", chat_js)
full_html = full_html.replace("/* __APP_JS__ */", app_js)

components.html(full_html, height=900, scrolling=False)
