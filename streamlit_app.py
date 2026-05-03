"""
TradeAI — TradingView 차트 + AI 채팅 Streamlit 앱
바이낸스 BTCUSDT.P 실시간 차트와 GPT AI 분석
"""

import streamlit as st
from openai import OpenAI
import streamlit.components.v1 as components

# ─── 페이지 설정 ───
st.set_page_config(
    page_title="TradeAI — 실시간 차트 AI 분석",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── 커스텀 CSS ───
st.markdown("""
<style>
    /* 전체 배경 & 폰트 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: #080b12;
        font-family: 'Inter', sans-serif;
    }

    /* 헤더 숨기기 */
    header[data-testid="stHeader"] {
        background: #0d1117;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown label {
        color: #8892a4;
    }

    /* 채팅 메시지 스타일 */
    .stChatMessage {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
    }

    /* 채팅 입력 */
    .stChatInput > div {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }

    .stChatInput textarea {
        color: #e8ecf1 !important;
    }

    /* 버튼 스타일 */
    .stButton > button {
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%) !important;
        color: #080b12 !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    .stButton > button:hover {
        box-shadow: 0 0 20px rgba(0, 245, 160, 0.15) !important;
    }

    /* 로고 영역 */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0 16px 0;
    }

    .logo-icon {
        font-size: 24px;
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .logo-text {
        font-size: 20px;
        font-weight: 700;
        color: #e8ecf1;
    }

    .logo-accent {
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* 심볼 뱃지 */
    .symbol-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: rgba(0, 245, 160, 0.08);
        border: 1px solid rgba(0, 245, 160, 0.2);
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #00f5a0;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }

    .badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #00f5a0;
        display: inline-block;
        animation: pulse 2s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* 타임프레임 버튼 그룹 */
    .tf-group {
        display: flex;
        gap: 4px;
        padding: 3px;
        background: #131a26;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 16px;
    }

    /* 퀵 액션 버튼 */
    .quick-btn {
        display: inline-block;
        padding: 8px 14px;
        margin: 4px;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        background: #131a26;
        color: #e8ecf1;
        font-size: 13px;
        cursor: pointer;
        text-decoration: none;
    }

    .quick-btn:hover {
        border-color: rgba(0, 245, 160, 0.3);
        background: rgba(0, 245, 160, 0.05);
    }

    /* 구분선 */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.06);
        margin: 12px 0;
    }

    /* selectbox 스타일 */
    .stSelectbox > div > div {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e8ecf1 !important;
    }

    /* text_input 스타일 */
    .stTextInput > div > div > input {
        background: #131a26 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e8ecf1 !important;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #131a26;
        border-radius: 10px;
        padding: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 7px;
        color: #8892a4;
        font-weight: 500;
        padding: 6px 16px;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00f5a0 0%, #00d4ff 100%) !important;
        color: #080b12 !important;
        font-weight: 600;
    }

    /* 마크다운 내 강조 */
    .stMarkdown strong { color: #00f5a0; }
    .stMarkdown em { color: #00d4ff; }

    /* 경고 배너 */
    .api-warning {
        padding: 10px 16px;
        background: rgba(255, 193, 7, 0.08);
        border: 1px solid rgba(255, 193, 7, 0.2);
        border-radius: 8px;
        color: #ffc107;
        font-size: 13px;
        text-align: center;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── 세션 상태 초기화 ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "interval" not in st.session_state:
    st.session_state.interval = "60"
if "symbol" not in st.session_state:
    st.session_state.symbol = "BINANCE:BTCUSDT.P"

INTERVAL_LABELS = {
    "1": "1분", "5": "5분", "15": "15분",
    "60": "1시간", "240": "4시간", "D": "1일"
}

# ─── TradingView 차트 HTML 생성 ───
def get_chart_html(symbol, interval):
    return f"""
    <div style="height:100%;width:100%;background:#080b12;">
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


# ─── 사이드바 ───
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <span class="logo-icon">◈</span>
        <span class="logo-text">Trade<span class="logo-accent">AI</span></span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="symbol-badge">
        <span class="badge-dot"></span>
        {st.session_state.symbol}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API 키 설정
    st.markdown("##### 🔑 API 설정")
    api_key = st.text_input(
        "OpenAI API 키",
        type="password",
        placeholder="sk-...",
        help="API 키는 세션 동안만 유지됩니다."
    )

    model = st.selectbox(
        "AI 모델",
        ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        index=0,
    )

    st.markdown("---")

    # 타임프레임 선택
    st.markdown("##### ⏱️ 타임프레임")
    interval = st.selectbox(
        "차트 타임프레임",
        options=list(INTERVAL_LABELS.keys()),
        format_func=lambda x: INTERVAL_LABELS[x],
        index=list(INTERVAL_LABELS.keys()).index(st.session_state.interval),
        label_visibility="collapsed"
    )
    if interval != st.session_state.interval:
        st.session_state.interval = interval
        st.rerun()

    st.markdown("---")

    # 퀵 액션
    st.markdown("##### ⚡ 빠른 분석")
    quick_actions = {
        "📊 차트 분석": "현재 차트 기술적 분석을 해주세요",
        "💡 매매 전략": "현재 매매 전략을 제안해주세요",
        "📐 지지/저항": "주요 지지선과 저항선을 알려주세요",
        "📈 지표 분석": "현재 RSI와 EMA 상태를 분석해주세요",
    }

    for label, msg in quick_actions.items():
        if st.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state.messages.append({"role": "user", "content": msg})
            st.session_state._trigger_response = True
            st.rerun()

    st.markdown("---")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── 메인 레이아웃 ───
chart_col, chat_col = st.columns([3, 2], gap="small")

# ─── 차트 영역 ───
with chart_col:
    chart_html = get_chart_html(st.session_state.symbol, st.session_state.interval)
    components.html(chart_html, height=620, scrolling=False)

# ─── 채팅 영역 ───
with chat_col:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;padding:8px 0 12px 0;">
        <div style="width:32px;height:32px;border-radius:6px;background:linear-gradient(135deg,#00f5a0,#00d4ff);
            display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#080b12;">AI</div>
        <div>
            <div style="font-size:14px;font-weight:600;color:#e8ecf1;">AI 트레이딩 어시스턴트</div>
            <div style="font-size:11px;color:#8892a4;">
                <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#00f5a0;margin-right:4px;"></span>
                {INTERVAL_LABELS[st.session_state.interval]} · {st.session_state.symbol.replace('BINANCE:', '')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div class="api-warning">
            🔑 AI 채팅을 사용하려면 좌측 사이드바에서 OpenAI API 키를 설정하세요.
        </div>
        """, unsafe_allow_html=True)

    # 채팅 메시지 표시
    chat_container = st.container(height=460)
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;">
                <div style="width:48px;height:48px;margin:0 auto 12px;border-radius:12px;
                    background:linear-gradient(135deg,#00f5a0,#00d4ff);display:flex;align-items:center;
                    justify-content:center;font-size:20px;color:#080b12;">◈</div>
                <h4 style="background:linear-gradient(135deg,#00f5a0,#00d4ff);-webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;margin-bottom:8px;">안녕하세요! AI 트레이딩 어시스턴트입니다</h4>
                <p style="color:#8892a4;font-size:13px;line-height:1.6;">
                    차트를 보면서 궁금한 점이나 분석이 필요한 부분을 물어보세요.<br>
                    좌측 사이드바의 <strong style="color:#00f5a0;">빠른 분석</strong> 버튼도 활용해보세요!
                </p>
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"], avatar="◈" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    # AI 응답 생성 (퀵 액션 트리거)
    if getattr(st.session_state, "_trigger_response", False) and api_key:
        st.session_state._trigger_response = False
        client = OpenAI(api_key=api_key)
        interval_label = INTERVAL_LABELS.get(st.session_state.interval, st.session_state.interval)
        sys_prompt = get_system_prompt(st.session_state.symbol, interval_label)

        with chat_container:
            with st.chat_message("assistant", avatar="◈"):
                stream = client.chat.completions.create(
                    model=model,
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
        st.rerun()

    # 채팅 입력
    if prompt := st.chat_input("차트에 대해 질문하세요...", key="chat_input"):
        if not api_key:
            st.warning("🔑 먼저 사이드바에서 OpenAI API 키를 설정해주세요.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})

            client = OpenAI(api_key=api_key)
            interval_label = INTERVAL_LABELS.get(st.session_state.interval, st.session_state.interval)
            sys_prompt = get_system_prompt(st.session_state.symbol, interval_label)

            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar="◈"):
                    stream = client.chat.completions.create(
                        model=model,
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
            st.rerun()
