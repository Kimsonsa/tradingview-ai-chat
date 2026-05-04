"""
TradeAI Assistant — 트레이딩 세션 관리 + AI 분석 데스크탑 앱
ChatGPT 스타일 사이드바: 새 대화 / 대화 목록 / 설정
"""
import streamlit as st
import json
import copy
from datetime import datetime
from core.capture import capture_tradingview, image_to_base64, parse_window_title, detect_chart_info
from core.market_data import get_market_context
from core.ai_client import analyze_chart, analyze_trade_summary
from core.session_manager import (
    create_session, save_session, load_session, delete_session, list_sessions,
)

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
        background: #1A1D23;
        border-right: 1px solid #2D3139;
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #C5C8D0 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #E8EAED !important;
    }

    /* 사이드바 버튼 */
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 1px solid #3D4149;
        color: #E8EAED;
        background: #2D3139;
        transition: all 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #2962FF;
        color: #ffffff;
        background: #353B45;
    }

    /* 새 거래 버튼 강조 */
    section[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button {
        background: linear-gradient(135deg, #2962FF 0%, #1E88E5 100%);
        border: none;
        color: white;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button:hover {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
    }

    /* 세션 카드 */
    .session-card {
        padding: 10px 12px;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        transition: background 0.15s;
        border: 1px solid transparent;
    }
    .session-card:hover {
        background: #2D3139;
    }
    .session-card.active {
        background: #2D3139;
        border-color: #2962FF;
    }
    .session-title {
        font-size: 13px;
        font-weight: 500;
        color: #E8EAED;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .session-meta {
        font-size: 11px;
        color: #8B8F97;
    }
    .session-pnl-pos { color: #26A69A; font-weight: 600; }
    .session-pnl-neg { color: #EF5350; font-weight: 600; }

    /* 히스토리 구분 */
    .history-label {
        font-size: 11px;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 12px 0 6px 4px;
        border-top: 1px solid #2D3139;
        margin-top: 8px;
    }

    /* 채팅 메시지 */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid #E8DFC8 !important;
        background: #FFFEF8 !important;
    }

    /* 메인 버튼 스타일 */
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

    /* 포지션 종료 버튼 (빨간색) */
    .close-position-btn > button {
        background: linear-gradient(135deg, #EF5350 0%, #E53935 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
    }
    .close-position-btn > button:hover {
        background: linear-gradient(135deg, #E53935 0%, #C62828 100%) !important;
        color: white !important;
    }

    /* 입력 필드 */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 8px !important;
        border: 1px solid #D4C9A8 !important;
        background: #FFFEF8 !important;
        color: #2D3436 !important;
    }

    /* 사이드바 입력 필드 */
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: #2D3139 !important;
        border-color: #3D4149 !important;
        color: #E8EAED !important;
    }

    /* 캡쳐 이미지 */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #E8DFC8;
    }

    /* 헤더 */
    h2 { color: #3D3425; }
    hr { border-color: #E8DFC8 !important; }

    .stMarkdown code {
        background: #FFF3D6 !important;
        color: #5D4E37 !important;
    }

    /* 읽기 전용 배너 */
    .readonly-banner {
        background: #FFF3CD;
        border: 1px solid #F0D78C;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 16px;
        color: #856404;
        font-size: 14px;
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

# ─── 세션 상태 초기화 ───
if "api_key" not in st.session_state:
    st.session_state.api_key = _config.get("api_key", "")
if "model" not in st.session_state:
    st.session_state.model = _config.get("model", "gpt-5.5")
if "auto_capture" not in st.session_state:
    st.session_state.auto_capture = True

# 다중 탭: tabs = { tab_id: { session_data } }
if "tabs" not in st.session_state:
    st.session_state.tabs = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = None  # 히스토리 열람 중인 session_id

# 시작 시 탭이 없으면 자동으로 하나 생성
if not st.session_state.tabs and st.session_state.viewing_history is None:
    new_sess = create_session()
    # TradingView에서 종목 감지 시도
    try:
        _sym, _tf, _ = detect_chart_info()
        if _sym:
            new_sess["symbol"] = _sym
        if _tf:
            new_sess["interval"] = _tf
    except Exception:
        pass
    st.session_state.tabs[new_sess["id"]] = new_sess
    st.session_state.active_tab = new_sess["id"]


def _deep_clean(obj):
    """재귀적으로 JSON 직렬화 불가능한 객체 제거 (PIL Image 등)"""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _deep_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_clean(v) for v in obj]
    # PIL Image, bytes, 기타 → None
    return None


def _safe_save_session(session):
    """세션을 안전하게 저장 (비직렬화 객체 완전 제거 후 저장)"""
    clean = dict(session)
    clean.pop("last_capture", None)
    clean.pop("last_capture_b64", None)
    clean_msgs = []
    for msg in clean.get("messages", []):
        m = {"role": msg.get("role", ""), "content": str(msg.get("content", ""))}
        clean_msgs.append(m)
    clean["messages"] = clean_msgs
    clean = _deep_clean(clean)
    save_session.__wrapped_clean__ = clean  # bypass
    # 직접 저장
    import os
    sessions_dir = os.path.join(os.path.dirname(__file__), "sessions")
    os.makedirs(sessions_dir, exist_ok=True)
    path = os.path.join(sessions_dir, f"{clean['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def _get_active_session():
    """현재 활성 탭의 세션 반환"""
    if st.session_state.active_tab and st.session_state.active_tab in st.session_state.tabs:
        return st.session_state.tabs[st.session_state.active_tab]
    return None


def _get_tab_label(sess):
    """탭 표시 라벨 생성"""
    sym = sess.get("symbol", "")
    intv = sess.get("interval", "")
    msg_count = len(sess.get("messages", []))
    if sym:
        label = f"{sym}"
        if intv:
            label += f" · {intv}"
    else:
        label = "새 대화"
    if msg_count > 0:
        label += f" ({msg_count})"
    return label


def _format_time(iso_str):
    """ISO 시간 문자열을 간단한 형식으로"""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now()
        if dt.date() == now.date():
            return dt.strftime("%H:%M")
        elif dt.year == now.year:
            return dt.strftime("%m/%d %H:%M")
        else:
            return dt.strftime("%Y/%m/%d")
    except Exception:
        return ""


# ═══════════════════════════════════════════════
# 사이드바
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ◈ TradeAI")

    # ── 새 거래 버튼 ──
    if st.button("➕ 새 거래", use_container_width=True, key="new_trade"):
        new_sess = create_session()
        try:
            _sym, _tf, _ = detect_chart_info()
            if _sym:
                new_sess["symbol"] = _sym
            if _tf:
                new_sess["interval"] = _tf
        except Exception:
            pass
        st.session_state.tabs[new_sess["id"]] = new_sess
        st.session_state.active_tab = new_sess["id"]
        st.session_state.viewing_history = None
        st.rerun()

    # ── 활성 탭 목록 ──
    if st.session_state.tabs:
        st.markdown('<div class="history-label">📊 진행 중</div>', unsafe_allow_html=True)
        for tab_id, sess in list(st.session_state.tabs.items()):
            label = _get_tab_label(sess)
            is_active = (tab_id == st.session_state.active_tab and
                         st.session_state.viewing_history is None)
            prefix = "▸ " if is_active else "  "
            if st.button(f"{prefix}{label}", key=f"tab_{tab_id}", use_container_width=True):
                st.session_state.active_tab = tab_id
                st.session_state.viewing_history = None
                st.rerun()

    # ── 거래 히스토리 ──
    history = list_sessions()
    closed_history = [s for s in history if s.get("status") == "closed"]
    if closed_history:
        st.markdown('<div class="history-label">📋 거래 히스토리</div>', unsafe_allow_html=True)
        for s in closed_history[:20]:  # 최근 20개만
            summary = s.get("summary") or {}
            sym = summary.get("symbol") or s.get("symbol") or "?"
            direction = summary.get("direction", "")
            result = summary.get("result", "")
            pnl = summary.get("pnl_percent")
            title = summary.get("title") or f"{sym} 거래"
            time_str = _format_time(s.get("closed_at") or s.get("created_at", ""))

            # PnL 표시
            if pnl is not None:
                pnl_class = "session-pnl-pos" if pnl >= 0 else "session-pnl-neg"
                pnl_str = f"+{pnl}%" if pnl >= 0 else f"{pnl}%"
            else:
                pnl_class = ""
                pnl_str = ""

            # 결과 아이콘
            if result == "익절":
                icon = "✅"
            elif result == "손절":
                icon = "❌"
            else:
                icon = "📊"

            # 버튼 라벨
            btn_label = f"{icon} {title}"
            if pnl_str:
                btn_label += f"  {pnl_str}"
            btn_label += f"  {time_str}"

            is_viewing = st.session_state.viewing_history == s["id"]
            if is_viewing:
                btn_label = "▸ " + btn_label

            if st.button(btn_label, key=f"hist_{s['id']}", use_container_width=True):
                st.session_state.viewing_history = s["id"]
                st.rerun()

    # ── 설정 (하단 접이식) ──
    st.markdown("---")
    with st.expander("⚙️ 설정", expanded=False):
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

        st.session_state.auto_capture = st.checkbox(
            "질문 시 자동 캡쳐", value=st.session_state.auto_capture)


# ═══════════════════════════════════════════════
# 메인 영역
# ═══════════════════════════════════════════════

# ── 히스토리 열람 모드 ──
if st.session_state.viewing_history:
    hist_data = load_session(st.session_state.viewing_history)
    if hist_data:
        summary = hist_data.get("summary") or {}
        sym = summary.get("symbol") or hist_data.get("symbol", "?")
        direction = summary.get("direction", "")
        result = summary.get("result", "미확정")
        pnl = summary.get("pnl_percent")
        entry = summary.get("entry_price")
        exit_p = summary.get("exit_price")
        actions = summary.get("actions", [])
        note = summary.get("note", "")

        # 결과 아이콘
        if result == "익절":
            result_icon = "✅"
        elif result == "손절":
            result_icon = "❌"
        else:
            result_icon = "📊"

        # 헤더
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"## {result_icon} {sym} {direction}")
            # 매매 요약
            info_parts = []
            if entry:
                info_parts.append(f"진입 `{entry:,.1f}`" if isinstance(entry, (int, float)) else f"진입 `{entry}`")
            if exit_p:
                info_parts.append(f"종료 `{exit_p:,.1f}`" if isinstance(exit_p, (int, float)) else f"종료 `{exit_p}`")
            if pnl is not None:
                pnl_str = f"+{pnl}%" if pnl >= 0 else f"{pnl}%"
                info_parts.append(f"PnL **{pnl_str}**")
            if actions:
                info_parts.append(" · ".join(actions))
            if info_parts:
                st.markdown(" → ".join(info_parts))
            if note:
                st.caption(note)
        with col2:
            if st.button("🗑️ 삭제", key="delete_hist"):
                delete_session(st.session_state.viewing_history)
                st.session_state.viewing_history = None
                st.rerun()
            if st.button("← 돌아가기", key="back_hist"):
                st.session_state.viewing_history = None
                st.rerun()

        st.markdown('<div class="readonly-banner">📖 읽기 전용 — 과거 거래 기록을 열람 중입니다</div>',
                    unsafe_allow_html=True)

        # 대화 내용 표시
        for msg in hist_data.get("messages", []):
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    else:
        st.error("세션을 찾을 수 없습니다.")
        st.session_state.viewing_history = None

    st.stop()  # 히스토리 열람 모드에서는 여기서 중단


# ── 활성 세션 모드 ──
sess = _get_active_session()
if sess is None:
    st.markdown("## ◈ TradeAI Assistant")
    st.markdown("왼쪽 사이드바에서 **➕ 새 거래**를 눌러 시작하세요.")
    st.stop()

# TradingView 감지 시도 (활성 세션에 종목 없을 때)
if not sess.get("symbol"):
    try:
        _sym, _tf, _ = detect_chart_info()
        if _sym:
            sess["symbol"] = _sym
        if _tf:
            sess["interval"] = _tf
    except Exception:
        pass

# ── 헤더 ──
col_title, col_capture, col_close = st.columns([5, 1, 1])
with col_title:
    sym_display = sess.get("symbol") or "종목 감지 중..."
    intv_display = sess.get("interval") or ""
    st.markdown(f"## ◈ {sym_display} {intv_display}")
    st.caption(f"모델: {st.session_state.model} · 세션: {_format_time(sess.get('created_at', ''))}")

with col_capture:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📸 캡쳐", key="manual_capture", use_container_width=True):
        with st.spinner("캡쳐 중..."):
            img, title = capture_tradingview()
            if img:
                b64, size = image_to_base64(img)
                sess["last_capture"] = img
                sess["last_capture_b64"] = b64
                sym, tf = parse_window_title(title)
                if sym:
                    sess["symbol"] = sym
                if tf:
                    sess["interval"] = tf
                st.success(f"✅ {size // 1024}KB")
                st.rerun()
            else:
                st.error(title)

with col_close:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📤 포지션 종료", key="close_position", use_container_width=True):
        if sess.get("messages"):
            with st.spinner("🤖 거래 분석 중..."):
                try:
                    summary = analyze_trade_summary(
                        api_key=st.session_state.api_key,
                        model=st.session_state.model,
                        messages=sess["messages"],
                    )
                except Exception as e:
                    summary = {
                        "symbol": sess.get("symbol", "UNKNOWN"),
                        "direction": "UNKNOWN", "result": "미확정",
                        "entry_price": None, "exit_price": None, "pnl_percent": None,
                        "actions": [], "title": "분석 실패", "note": str(e),
                    }

            # 세션 종료 처리
            sess["status"] = "closed"
            sess["closed_at"] = datetime.now().isoformat()
            sess["summary"] = summary
            if summary.get("symbol") and summary["symbol"] != "UNKNOWN":
                sess["symbol"] = summary["symbol"]
            _safe_save_session(sess)

            # 탭에서 제거
            tab_id = sess["id"]
            del st.session_state.tabs[tab_id]

            # 다른 탭이 있으면 그쪽으로, 없으면 새 세션
            if st.session_state.tabs:
                st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
            else:
                st.session_state.active_tab = None

            st.rerun()
        else:
            # 대화 없이 종료 → 그냥 탭 닫기
            tab_id = sess["id"]
            del st.session_state.tabs[tab_id]
            if st.session_state.tabs:
                st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
            else:
                st.session_state.active_tab = None
            st.rerun()


# ── 이전 메시지 표시 ──
for msg in sess.get("messages", []):
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"], caption="분석한 차트", use_container_width=True)

# ── 빠른 분석 버튼 (새 대화일 때만) ──
if not sess.get("messages"):
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

# ── 채팅 입력 ──
user_input = st.chat_input("차트에 대해 질문하세요...")
prompt = pending or user_input

if prompt:
    if not st.session_state.api_key:
        st.warning("🔑 사이드바 설정에서 OpenAI API 키를 먼저 입력하세요.")
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
                    sess["last_capture"] = img
                    sess["last_capture_b64"] = b64
                    capture_b64 = b64
                    capture_img = img
                    capture_size = size
                    sym, tf = parse_window_title(title)
                    if sym:
                        sess["symbol"] = sym
                    if tf:
                        sess["interval"] = tf
                else:
                    st.warning(f"⚠️ 캡쳐 실패: {title}")
        elif sess.get("last_capture_b64"):
            capture_b64 = sess["last_capture_b64"]
            capture_img = sess.get("last_capture")

        # 사용자 메시지 추가
        sess["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 실시간 데이터 수집
        symbol = sess.get("symbol", "BTCUSDT")
        interval = sess.get("interval", "1시간")
        with st.spinner("📊 실시간 데이터 수집 중..."):
            market_data = get_market_context(symbol, interval)

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
                        messages=sess["messages"],
                        image_base64=capture_b64,
                        market_data=market_data,
                    )
                )
            except Exception as e:
                response = f"⚠️ API 오류: {str(e)}"
                st.error(response)

        sess["messages"].append({
            "role": "assistant",
            "content": str(response) if response else "",
            "image": capture_img,
        })
