"""
TradeAI Assistant — TradingView 자동 캡쳐 + AI 분석 데스크탑 앱
"""
import streamlit as st
from core.capture import capture_tradingview, image_to_base64, parse_window_title, BINANCE_INTERVAL_MAP
from core.market_data import get_market_context, INTERVAL_OPTIONS
from core.ai_client import analyze_chart

# ─── 페이지 설정 ───
st.set_page_config(
    page_title="TradeAI Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* 사이드바 스타일 */
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    /* 채팅 메시지 */
    .stChatMessage { border-radius: 10px !important; }

    /* 캡쳐 미리보기 */
    .capture-preview {
        border: 1px solid rgba(0,245,160,0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 8px 0;
    }

    /* 상태 배지 */
    .status-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
    }
    .status-badge.ready {
        background: rgba(0,245,160,0.1);
        color: #00f5a0;
        border: 1px solid rgba(0,245,160,0.2);
    }
    .status-badge.error {
        background: rgba(255,71,87,0.1);
        color: #ff4757;
        border: 1px solid rgba(255,71,87,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ─── 세션 상태 초기화 ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model" not in st.session_state:
    st.session_state.model = "gpt-5.5"
if "interval" not in st.session_state:
    st.session_state.interval = "1시간"
if "symbol" not in st.session_state:
    st.session_state.symbol = "BTCUSDT"
if "last_capture" not in st.session_state:
    st.session_state.last_capture = None
if "last_capture_b64" not in st.session_state:
    st.session_state.last_capture_b64 = None
if "auto_capture" not in st.session_state:
    st.session_state.auto_capture = True

# ─── 사이드바: 설정 ───
with st.sidebar:
    st.markdown("### ◈ TradeAI Assistant")
    st.markdown("---")

    # API 설정
    st.markdown("#### 🔑 API 설정")
    api_key = st.text_input("OpenAI API 키", type="password",
                            value=st.session_state.api_key, placeholder="sk-...")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key

    model = st.selectbox("AI 모델", ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                         index=["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"].index(st.session_state.model))
    st.session_state.model = model

    st.markdown("---")

    # 감지된 차트 정보 (자동)
    st.markdown("#### 📊 감지된 차트")
    st.markdown(f"**종목**: `{st.session_state.symbol}`")
    st.markdown(f"**타임프레임**: `{st.session_state.interval}`")
    st.caption("⬆️ TradingView 창에서 자동 감지")

    st.markdown("---")

    # 캡쳐 설정
    st.markdown("#### 📸 TradingView 캡쳐")
    st.session_state.auto_capture = st.checkbox("질문 시 자동 캡쳐", value=st.session_state.auto_capture)

    if st.button("📸 지금 캡쳐", use_container_width=True):
        with st.spinner("TradingView 캡쳐 중..."):
            img, title = capture_tradingview()
            if img:
                st.session_state.last_capture = img
                st.session_state.last_capture_b64 = image_to_base64(img)
                # 자동 감지
                sym, tf = parse_window_title(title)
                if sym:
                    st.session_state.symbol = sym
                if tf:
                    st.session_state.interval = tf
                st.success(f"✅ 캡쳐 완료: {title}")
                st.rerun()
            else:
                st.error(title)

    if st.session_state.last_capture:
        st.image(st.session_state.last_capture, caption="마지막 캡쳐", use_container_width=True)

    st.markdown("---")
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_capture = None
        st.session_state.last_capture_b64 = None
        st.rerun()

# ─── 메인: 채팅 ───
st.markdown("## ◈ AI 트레이딩 어시스턴트")
st.caption(f"📊 {st.session_state.symbol} · {st.session_state.interval} · 모델: {st.session_state.model}")

# 이전 메시지 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], caption="분석한 차트", use_container_width=True)

# 빠른 분석 버튼
if not st.session_state.messages:
    st.markdown("---")
    cols = st.columns(4)
    quick_items = [
        ("📊 차트 분석", "현재 차트를 종합 분석해줘"),
        ("💡 매매 전략", "지금 진입해도 될까? 매매 전략 제안해줘"),
        ("📐 지지/저항", "차트에 보이는 주요 지지선과 저항선 분석해줘"),
        ("📈 추세 판단", "현재 추세 방향과 강도를 판단해줘"),
    ]
    for i, (label, msg) in enumerate(quick_items):
        with cols[i]:
            if st.button(label, key=f"quick_{i}", use_container_width=True):
                st.session_state._pending_msg = msg
                st.rerun()

# Pending 메시지 처리
pending = st.session_state.pop("_pending_msg", None)

# 채팅 입력
user_input = st.chat_input("차트에 대해 질문하세요...")
prompt = pending or user_input

if prompt:
    if not st.session_state.api_key:
        st.warning("🔑 사이드바에서 OpenAI API 키를 먼저 입력하세요.")
    else:
        # 자동 캡쳐
        capture_b64 = None
        capture_img = None
        if st.session_state.auto_capture:
            with st.spinner("📸 TradingView 캡쳐 중..."):
                img, title = capture_tradingview()
                if img:
                    st.session_state.last_capture = img
                    st.session_state.last_capture_b64 = image_to_base64(img)
                    capture_b64 = st.session_state.last_capture_b64
                    capture_img = img
                    # 자동 감지
                    sym, tf = parse_window_title(title)
                    if sym:
                        st.session_state.symbol = sym
                    if tf:
                        st.session_state.interval = tf
        elif st.session_state.last_capture_b64:
            capture_b64 = st.session_state.last_capture_b64
            capture_img = st.session_state.last_capture

        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 실시간 데이터 수집
        with st.spinner("📊 실시간 데이터 수집 중..."):
            market_data = get_market_context(st.session_state.symbol, st.session_state.interval)

        # AI 분석
        with st.chat_message("assistant", avatar="🤖"):
            if capture_img:
                st.image(capture_img, caption="📸 분석 중인 차트", use_container_width=True)

            response = st.write_stream(
                analyze_chart(
                    api_key=st.session_state.api_key,
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    image_base64=capture_b64,
                    market_data=market_data,
                )
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "image": capture_img,
        })
