"""
TradeAI — TradingView 차트 + AI 채팅 Streamlit 앱
전체화면 차트 + 팝업 설정/채팅
"""

import streamlit as st
from openai import OpenAI
import streamlit.components.v1 as components

# ─── 페이지 설정 ───
st.set_page_config(
    page_title="TradeAI — 실시간 차트 AI 분석",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── 세션 상태 초기화 ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interval" not in st.session_state:
    st.session_state.interval = "60"
if "symbol" not in st.session_state:
    st.session_state.symbol = "BINANCE:BTCUSDT.P"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model" not in st.session_state:
    st.session_state.model = "gpt-5.5"

INTERVAL_LABELS = {
    "1": "1분", "5": "5분", "15": "15분",
    "60": "1시간", "240": "4시간", "D": "1일"
}

# ─── 전체화면 CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: #080b12;
        font-family: 'Inter', sans-serif;
    }

    /* 사이드바 완전 숨기기 */
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="stSidebarCollapsedControl"] { display: none !important; }

    /* 헤더 최소화 */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0px !important;
        min-height: 0px !important;
        padding: 0 !important;
    }

    /* 메인 영역 패딩 제거 */
    .stMainBlockContainer {
        padding: 0 !important;
        max-width: 100% !important;
    }

    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    section[data-testid="stMain"] {
        padding: 0 !important;
    }

    /* iframe (차트) 전체화면 */
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
    }

    /* 하단 Streamlit 브랜딩 숨기기 */
    footer { display: none !important; }
    .stDeployButton { display: none !important; }

    /* 플로팅 버튼 스타일 */
    .floating-buttons {
        position: fixed;
        top: 12px;
        right: 16px;
        z-index: 99999;
        display: flex;
        gap: 8px;
    }

    .float-btn {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(13, 17, 23, 0.85);
        backdrop-filter: blur(12px);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 18px;
        color: #e8ecf1;
        transition: all 0.2s;
        text-decoration: none;
    }

    .float-btn:hover {
        background: rgba(0, 245, 160, 0.15);
        border-color: rgba(0, 245, 160, 0.4);
        box-shadow: 0 0 20px rgba(0, 245, 160, 0.1);
        transform: translateY(-1px);
    }

    /* 로고 플로팅 */
    .floating-logo {
        position: fixed;
        top: 12px;
        left: 16px;
        z-index: 99999;
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(13, 17, 23, 0.85);
        backdrop-filter: blur(12px);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .floating-logo .icon {
        font-size: 18px;
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .floating-logo .text {
        font-size: 15px;
        font-weight: 700;
        color: #e8ecf1;
    }

    .floating-logo .accent {
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .floating-logo .badge {
        font-size: 10px;
        font-weight: 600;
        color: #00f5a0;
        padding: 2px 8px;
        background: rgba(0, 245, 160, 0.1);
        border-radius: 10px;
        margin-left: 4px;
    }

    /* 다이얼로그 스타일 */
    div[data-testid="stDialog"] > div {
        background: #0d1117 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
    }

    div[data-testid="stDialog"] .stMarkdown p { color: #8892a4; }
    div[data-testid="stDialog"] .stMarkdown h4 { color: #e8ecf1; }
    div[data-testid="stDialog"] .stMarkdown strong { color: #00f5a0; }

    div[data-testid="stDialog"] .stTextInput > div > div > input {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e8ecf1 !important;
    }

    div[data-testid="stDialog"] .stSelectbox > div > div {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e8ecf1 !important;
    }

    /* 채팅 메시지 스타일 */
    .stChatMessage {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
    }

    /* 버튼 */
    div[data-testid="stDialog"] .stButton > button {
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%) !important;
        color: #080b12 !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    /* Streamlit 기본 버튼 (플로팅용) - 투명하게 */
    .stMainBlockContainer .stButton > button {
        position: fixed;
        z-index: 100000;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.1) !important;
        background: rgba(13, 17, 23, 0.85) !important;
        backdrop-filter: blur(12px);
        font-size: 18px;
        color: #e8ecf1 !important;
        padding: 0 !important;
        min-height: 0 !important;
        line-height: 1 !important;
    }

    .stMainBlockContainer .stButton > button:hover {
        background: rgba(0, 245, 160, 0.15) !important;
        border-color: rgba(0, 245, 160, 0.4) !important;
        box-shadow: 0 0 20px rgba(0, 245, 160, 0.1);
        color: #00f5a0 !important;
    }

    /* 설정 버튼 위치 */
    div[data-testid="stColumn"]:nth-child(1) .stButton > button {
        top: 12px;
        right: 68px;
    }

    /* 채팅 버튼 위치 */
    div[data-testid="stColumn"]:nth-child(2) .stButton > button {
        top: 12px;
        right: 16px;
    }

    /* 컬럼 컨테이너 숨기기 (버튼만 플로팅) */
    .stMainBlockContainer > div > div > div[data-testid="stHorizontalBlock"] {
        position: fixed;
        z-index: 100000;
        height: 0;
        overflow: visible;
    }
</style>
""", unsafe_allow_html=True)

# ─── 플로팅 로고 ───
st.markdown(f"""
<div class="floating-logo">
    <span class="icon">◈</span>
    <span class="text">Trade<span class="accent">AI</span></span>
    <span class="badge">{st.session_state.symbol.replace('BINANCE:', '')}</span>
</div>
""", unsafe_allow_html=True)

# ─── TradingView 차트 HTML ───
def get_chart_html(symbol, interval):
    return f"""
    <div style="height:100vh;width:100vw;background:#080b12;overflow:hidden;">
        <div class="tradingview-widget-container" style="height:100%;width:100%">
            <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
            <script type="text/javascript"
                src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js"
                async>
            {{
                "autosize": true,
                "symbol": "{symbol}",
                "interval": "{interval}",
                "timezone": "Asia/Seoul",
                "theme": "dark",
                "style": "1",
                "locale": "ko",
                "backgroundColor": "rgba(8, 11, 18, 1)",
                "gridColor": "rgba(30, 40, 60, 0.25)",
                "allow_symbol_change": false,
                "calendar": false,
                "hide_top_toolbar": false,
                "hide_legend": false,
                "save_image": false,
                "studies": [
                    "MAExp@tv-basicstudies",
                    "MAExp@tv-basicstudies",
                    "MAExp@tv-basicstudies",
                    "RSI@tv-basicstudies",
                    "Volume@tv-basicstudies"
                ],
                "support_host": "https://www.tradingview.com"
            }}
            </script>
        </div>
    </div>
    """

# ─── 시스템 프롬프트 ───
def get_system_prompt(symbol, interval_label):
    return f"""당신은 전문 암호화폐 트레이딩 분석가입니다. 사용자가 보고 있는 차트를 기반으로 기술적 분석을 제공합니다.

현재 차트 정보:
- 종목: {symbol}
- 타임프레임: {interval_label}
- 거래소: Binance (선물)

적용된 보조지표:
- EMA 20, 50, 200
- RSI (14)
- 거래량 (Volume)

분석 시 다음을 포함해주세요:
1. 현재 추세 분석 (EMA 배열 기반)
2. RSI 과매수/과매도 상태
3. 거래량 분석
4. 주요 지지/저항 레벨
5. 매매 전략 제안

답변은 한국어로 해주세요. 구체적인 수치와 함께 분석해주세요.
마크다운 포맷을 사용하여 가독성 좋게 답변해주세요.

⚠️ 중요: 이것은 투자 조언이 아닌 기술적 분석 의견임을 항상 명시해주세요."""


# ─── 설정 다이얼로그 ───
@st.dialog("⚙️ 설정")
def settings_dialog():
    st.markdown("#### 🔑 API 설정")
    new_api_key = st.text_input(
        "OpenAI API 키",
        type="password",
        value=st.session_state.api_key,
        placeholder="sk-...",
    )
    new_model = st.selectbox(
        "AI 모델",
        ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        index=["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"].index(st.session_state.model),
    )

    st.markdown("---")
    st.markdown("#### ⏱️ 타임프레임")
    new_interval = st.selectbox(
        "차트 타임프레임",
        options=list(INTERVAL_LABELS.keys()),
        format_func=lambda x: INTERVAL_LABELS[x],
        index=list(INTERVAL_LABELS.keys()).index(st.session_state.interval),
    )

    st.markdown("---")
    if st.button("💾 저장", use_container_width=True, type="primary"):
        st.session_state.api_key = new_api_key
        st.session_state.model = new_model
        if new_interval != st.session_state.interval:
            st.session_state.interval = new_interval
        st.rerun()


# ─── 채팅 다이얼로그 ───
@st.dialog("💬 AI 트레이딩 어시스턴트", width="large")
def chat_dialog():
    interval_label = INTERVAL_LABELS.get(st.session_state.interval, st.session_state.interval)

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:0 0 8px 0;">
        <div style="width:28px;height:28px;border-radius:6px;background:linear-gradient(135deg,#00f5a0,#00d4ff);
            display:flex;align-items:center;justify-content:center;font-size:10px;font-weight:700;color:#080b12;">AI</div>
        <div style="font-size:11px;color:#8892a4;">
            {interval_label} · {st.session_state.symbol.replace('BINANCE:', '')} · 
            모델: <strong style="color:#00f5a0">{st.session_state.model}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.api_key:
        st.warning("🔑 설정(⚙️)에서 OpenAI API 키를 먼저 입력하세요.")

    # 빠른 분석 버튼
    qa_cols = st.columns(4)
    quick_items = [
        ("📊 차트 분석", "현재 차트 기술적 분석을 해주세요"),
        ("💡 매매 전략", "현재 매매 전략을 제안해주세요"),
        ("📐 지지/저항", "주요 지지선과 저항선을 알려주세요"),
        ("📈 지표 분석", "현재 RSI와 EMA 상태를 분석해주세요"),
    ]
    for i, (label, msg) in enumerate(quick_items):
        with qa_cols[i]:
            if st.button(label, key=f"qa_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": msg})
                st.session_state._need_response = True
                st.rerun()

    # 메시지 컨테이너
    chat_container = st.container(height=400)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;">
                <div style="font-size:32px;margin-bottom:8px;">◈</div>
                <p style="color:#8892a4;font-size:13px;">차트에 대해 궁금한 점을 물어보세요</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"], avatar="◈" if msg["role"] == "assistant" else "👤"):
                    st.markdown(msg["content"])

    # AI 응답 생성 (pending response)
    if st.session_state.get("_need_response") and st.session_state.api_key:
        st.session_state._need_response = False
        try:
            client = OpenAI(api_key=st.session_state.api_key)
            sys_prompt = get_system_prompt(st.session_state.symbol, interval_label)
            with chat_container:
                with st.chat_message("assistant", avatar="◈"):
                    stream = client.chat.completions.create(
                        model=st.session_state.model,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            *st.session_state.messages
                        ],
                        stream=True,
                        temperature=0.7,
                        max_tokens=2000,
                    )
                    response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"오류: {e}")

    # 입력 폼
    with st.form("chat_form", clear_on_submit=True):
        cols = st.columns([7, 1])
        with cols[0]:
            user_input = st.text_input(
                "메시지",
                label_visibility="collapsed",
                placeholder="차트에 대해 질문하세요...",
            )
        with cols[1]:
            submitted = st.form_submit_button("전송")
    
    if submitted and user_input:
        if not st.session_state.api_key:
            st.warning("🔑 API 키를 먼저 설정하세요.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state._need_response = True
            st.rerun()

    # 하단 버튼
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ 대화 초기화", key="clear_chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with c2:
        if st.button("✕ 닫기", key="close_chat", use_container_width=True):
            st.rerun()


# ─── 플로팅 버튼 (설정 & 채팅) ───
btn_cols = st.columns([1, 1])
with btn_cols[0]:
    if st.button("⚙️", key="settings_btn"):
        settings_dialog()
with btn_cols[1]:
    if st.button("💬", key="chat_btn"):
        chat_dialog()

# 퀵 액션에서 돌아온 경우 자동으로 채팅 다이얼로그 열기
if st.session_state.get("_need_response"):
    chat_dialog()

# ─── 전체화면 차트 ───
chart_html = get_chart_html(st.session_state.symbol, st.session_state.interval)
components.html(chart_html, height=900, scrolling=False)
