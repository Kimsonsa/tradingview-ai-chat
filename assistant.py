"""
TradeAI Assistant — TradingView 자동 캡쳐 + AI 분석 데스크탑 앱
"""
import streamlit as st
from core.capture import capture_tradingview, image_to_base64, parse_window_title, detect_chart_info, BINANCE_INTERVAL_MAP
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

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* 사이드바 */
    section[data-testid="stSidebar"] {
        background: #FFF8E7;
        border-right: 1px solid #E8DFC8;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #5D4E37;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #4A4A4A;
    }

    /* 채팅 메시지 */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid #E8DFC8 !important;
        background: #FFFEF8 !important;
    }

    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #D4C9A8;
        color: #5D4E37;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #2962FF;
        color: #2962FF;
        background: #FFF8E7;
    }

    /* 입력 필드 */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #D4C9A8 !important;
        background: #FFFEF8 !important;
        color: #2D3436 !important;
    }

    /* 캡쳐 이미지 */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E8DFC8;
    }

    /* 헤더 */
    h2 { color: #3D3425; }

    /* 구분선 */
    hr { border-color: #E8DFC8 !important; }

    /* 코드 블록 (분석 결과) */
    .stMarkdown code {
        background: #FFF3D6 !important;
        color: #5D4E37 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── 설정 파일 (로컬 영구 저장) ───
import json, os
CONFIG_PATH = os.path.join(os.path.dirname(__file__), ".tradeai_config.json")

def load_config():
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_config(data):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass

_config = load_config()

# ─── 세션 상태 초기화 (저장된 설정 우선) ───
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = _config.get("api_key", "")
if "model" not in st.session_state:
    st.session_state.model = _config.get("model", "gpt-5.5")
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

# ─── 페이지 로드 시 TradingView 창에서 자동 감지 ───
try:
    _sym, _tf, _title = detect_chart_info()
    if _sym:
        st.session_state.symbol = _sym
    if _tf:
        st.session_state.interval = _tf
except Exception:
    pass

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
        save_config({"api_key": api_key, "model": st.session_state.model})

    MODELS = ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    model = st.selectbox("AI 모델", MODELS,
                         index=MODELS.index(st.session_state.model) if st.session_state.model in MODELS else 0)
    if model != st.session_state.model:
        st.session_state.model = model
        save_config({"api_key": st.session_state.api_key, "model": model})

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
                b64, size = image_to_base64(img)
                st.session_state.last_capture = img
                st.session_state.last_capture_b64 = b64
                # 자동 감지
                sym, tf = parse_window_title(title)
                if sym:
                    st.session_state.symbol = sym
                if tf:
                    st.session_state.interval = tf
                st.success(f"✅ 캡쳐 완료 ({size // 1024}KB)")
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
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.markdown("## ◈ AI 트레이딩 어시스턴트")
    st.caption(f"📊 {st.session_state.symbol} · {st.session_state.interval} · 모델: {st.session_state.model}")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ 채팅 초기화", key="clear_main"):
        st.session_state.messages = []
        st.session_state.last_capture = None
        st.session_state.last_capture_b64 = None
        st.rerun()

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
        capture_size = 0
        if st.session_state.auto_capture:
            with st.spinner("📸 TradingView 캡쳐 중..."):
                img, title = capture_tradingview()
                if img:
                    b64, size = image_to_base64(img)
                    st.session_state.last_capture = img
                    st.session_state.last_capture_b64 = b64
                    capture_b64 = b64
                    capture_img = img
                    capture_size = size
                    # 자동 감지
                    sym, tf = parse_window_title(title)
                    if sym:
                        st.session_state.symbol = sym
                    if tf:
                        st.session_state.interval = tf
                else:
                    st.warning(f"⚠️ 캡쳐 실패: {title}")
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
                st.caption(f"✅ 이미지 첨부됨 ({capture_size // 1024}KB) — {st.session_state.model}")
            else:
                st.caption("⚠️ 차트 이미지 없이 데이터만으로 분석합니다")

            try:
                response = st.write_stream(
                    analyze_chart(
                        api_key=st.session_state.api_key,
                        model=st.session_state.model,
                        messages=st.session_state.messages,
                        image_base64=capture_b64,
                        market_data=market_data,
                    )
                )
            except Exception as e:
                response = f"⚠️ API 오류: {str(e)}"
                st.error(response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "image": capture_img,
        })
