"""
TradeAI Assistant — 트레이딩 세션 관리 + AI 분석 데스크탑 앱
ChatGPT 스타일 사이드바: 새 대화 / 대화 목록 / 설정
"""
import streamlit as st
import streamlit.components.v1 as components
import json
from datetime import datetime
from core.capture import capture_tradingview, image_to_base64, parse_window_title, detect_chart_info
from core.market_data import get_market_context, get_multi_timeframe_context, parse_requested_timeframes
from core.ai_client import analyze_chart, analyze_trade_summary, is_claude_model, CLAUDE_MODELS
from core.rsi_wave import (
    analyze_rsi_wave, generate_summary_text, format_rsi_wave_for_ai,
    format_machine_context, RSI_WAVE_SYSTEM_PROMPT, WAVE_TIMEFRAMES,
)
from core.rsi_render import (
    generate_wave_svg, generate_price_ladder_svg, generate_tf_cards,
)
from core.session_manager import (
    create_session, save_session, load_session, delete_session, list_sessions,
)

# 모바일 분석 요청 처리 워커 (앱 실행 중 백그라운드로 작업 큐 폴링)
try:
    from core.job_worker import start_worker_thread
    start_worker_thread()
except Exception:
    pass

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

    /* ═══ 글로벌 ═══ */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ═══ 사이드바 — 웜톤 라이트 테마 ═══ */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFF9F0 0%, #FFF5E6 40%, #FFF0DB 100%) !important;
        border-right: 1px solid #E8D5B8 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        background: transparent !important;
    }

    /* 사이드바 텍스트 */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {
        color: #6B5D4F !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown h4 {
        color: #3D3425 !important;
        letter-spacing: -0.3px;
    }

    /* 사이드바 버튼 */
    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        font-size: 13px;
        border: 1px solid #D9C9AD;
        color: #5D4E37;
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 3px rgba(139, 115, 85, 0.08);
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        border-color: #2962FF;
        color: #2962FF;
        background: rgba(255, 255, 255, 0.9);
        box-shadow: 0 4px 12px rgba(41, 98, 255, 0.12);
        transform: translateY(-1px);
    }

    /* 새 거래 버튼 강조 */
    section[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button {
        background: linear-gradient(135deg, #2962FF 0%, #448AFF 100%);
        border: none;
        color: white !important;
        font-weight: 600;
        font-size: 14px;
        box-shadow: 0 4px 14px rgba(41, 98, 255, 0.3);
        letter-spacing: 0.2px;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type > button:hover {
        background: linear-gradient(135deg, #1E88E5 0%, #2962FF 100%);
        box-shadow: 0 6px 20px rgba(41, 98, 255, 0.4);
        transform: translateY(-2px);
        color: white !important;
    }

    /* 세션 카드 */
    .session-card {
        padding: 10px 12px;
        border-radius: 10px;
        margin: 4px 0;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .session-card:hover {
        background: rgba(255, 255, 255, 0.6);
        border-color: #E0D0B8;
    }
    .session-card.active {
        background: rgba(255, 255, 255, 0.7);
        border-color: #2962FF;
        box-shadow: 0 2px 8px rgba(41, 98, 255, 0.1);
    }
    .session-title {
        font-size: 13px;
        font-weight: 500;
        color: #3D3425;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .session-meta {
        font-size: 11px;
        color: #9A8B78;
    }
    .session-pnl-pos { color: #16A34A; font-weight: 600; }
    .session-pnl-neg { color: #EF4444; font-weight: 600; }

    /* 히스토리 구분 */
    .history-label {
        font-size: 11px;
        color: #9A8B78;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 14px 0 6px 4px;
        border-top: 1px solid #E0D0B8;
        margin-top: 8px;
    }

    /* 사이드바 구분선 */
    section[data-testid="stSidebar"] hr {
        border-color: #E0D0B8 !important;
    }

    /* 사이드바 expander (설정) */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #6B5D4F !important;
        font-size: 13px;
        background: rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        border: 1px solid #E0D0B8;
    }
    section[data-testid="stSidebar"] .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.4);
        border-radius: 0 0 10px 10px;
        border: 1px solid #E0D0B8;
        border-top: none;
    }

    /* 사이드바 입력 필드 */
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.7) !important;
        border-color: #D9C9AD !important;
        color: #3D3425 !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] .stTextInput > div > div > input:focus {
        border-color: #2962FF !important;
        box-shadow: 0 0 0 2px rgba(41, 98, 255, 0.15) !important;
    }

    /* 사이드바 체크박스 */
    section[data-testid="stSidebar"] .stCheckbox label {
        color: #6B5D4F !important;
    }

    /* ═══ 메인 영역 ═══ */

    /* 채팅 메시지 */
    .stChatMessage {
        border-radius: 14px !important;
        border: 1px solid #E8DFC8 !important;
        background: #FFFEF8 !important;
        box-shadow: 0 1px 4px rgba(139, 115, 85, 0.06);
        transition: box-shadow 0.2s ease;
    }
    .stChatMessage:hover {
        box-shadow: 0 2px 8px rgba(139, 115, 85, 0.1);
    }

    /* 메인 버튼 스타일 */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        border: 1px solid #D4C9A8;
        color: #5D4E37;
        background: rgba(255, 255, 255, 0.7);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton > button:hover {
        border-color: #2962FF;
        color: #2962FF;
        background: #FFF8E7;
        box-shadow: 0 3px 10px rgba(41, 98, 255, 0.1);
        transform: translateY(-1px);
    }

    /* 포지션 종료 버튼 (빨간색) */
    .close-position-btn > button {
        background: linear-gradient(135deg, #EF5350 0%, #E53935 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        box-shadow: 0 3px 12px rgba(239, 83, 80, 0.25) !important;
    }
    .close-position-btn > button:hover {
        background: linear-gradient(135deg, #E53935 0%, #C62828 100%) !important;
        color: white !important;
        box-shadow: 0 5px 16px rgba(239, 83, 80, 0.35) !important;
        transform: translateY(-1px);
    }

    /* 입력 필드 */
    .stTextInput > div > div > input,
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 1px solid #D4C9A8 !important;
        background: #FFFEF8 !important;
        color: #2D3436 !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #2962FF !important;
        box-shadow: 0 0 0 3px rgba(41, 98, 255, 0.1) !important;
    }

    /* 채팅 입력 */
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1.5px solid #D4C9A8 !important;
        background: #FFFEF8 !important;
        box-shadow: 0 2px 8px rgba(139, 115, 85, 0.08);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stChatInput > div:focus-within {
        border-color: #2962FF !important;
        box-shadow: 0 4px 16px rgba(41, 98, 255, 0.12) !important;
    }

    /* 캡쳐 이미지 */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #E8DFC8;
        box-shadow: 0 2px 8px rgba(139, 115, 85, 0.08);
    }

    /* 헤더 */
    h2 {
        color: #3D3425;
        letter-spacing: -0.3px;
    }
    hr { border-color: #E8DFC8 !important; }

    .stMarkdown code {
        background: #FFF3D6 !important;
        color: #5D4E37 !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.9em;
    }

    /* 읽기 전용 배너 */
    .readonly-banner {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFF3CD 100%);
        border: 1px solid #F0D78C;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 16px;
        color: #856404;
        font-size: 14px;
        box-shadow: 0 1px 4px rgba(240, 215, 140, 0.2);
    }

    /* 스피너 */
    .stSpinner > div {
        border-color: #2962FF transparent transparent transparent !important;
    }

    /* 스크롤바 (사이드바) */
    section[data-testid="stSidebar"] ::-webkit-scrollbar {
        width: 4px;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-track {
        background: transparent;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
        background: #D4C9A8;
        border-radius: 4px;
    }
    section[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
        background: #BBA98A;
    }
    /* 자동캡쳐 체크박스 (채팅 입력 위) */
    [data-testid="stMain"] > div > div > div > div > .stCheckbox {
        margin-bottom: -8px;
    }
    [data-testid="stMain"] > div > div > div > div > .stCheckbox label {
        font-size: 12px !important;
        color: #9A8B78 !important;
        font-weight: 400;
    }
    [data-testid="stMain"] > div > div > div > div > .stCheckbox label span {
        font-size: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── 설정 파일 (로컬 영구 저장 — 멀티페이지 공용 모듈) ───
from core.app_config import load_config, update_config

_config = load_config()

# ─── 세션 상태 초기화 ───
if "api_key" not in st.session_state:
    st.session_state.api_key = _config.get("api_key", "")
if "claude_api_key" not in st.session_state:
    st.session_state.claude_api_key = _config.get("claude_api_key", "")
if "model" not in st.session_state:
    st.session_state.model = _config.get("model", "gpt-5.5")
if "account_size" not in st.session_state:
    st.session_state.account_size = float(_config.get("account_size", 0) or 0)
if "risk_pct" not in st.session_state:
    st.session_state.risk_pct = float(_config.get("risk_pct", 1.0) or 1.0)


def _persist_config():
    """현재 설정을 로컬 설정파일에 병합 저장 (watchlist 등 다른 키 보존)"""
    update_config(
        api_key=st.session_state.api_key,
        claude_api_key=st.session_state.claude_api_key,
        model=st.session_state.model,
        account_size=st.session_state.account_size,
        risk_pct=st.session_state.risk_pct,
    )


def _active_api_key():
    """선택된 모델의 프로바이더(OpenAI/Claude)에 맞는 API 키 반환"""
    if is_claude_model(st.session_state.model):
        return st.session_state.claude_api_key
    return st.session_state.api_key


def _key_warning():
    """현재 모델 프로바이더에 맞는 키 입력 안내 메시지"""
    provider = "Claude(Anthropic)" if is_claude_model(st.session_state.model) else "OpenAI"
    return f"🔑 사이드바 설정에서 {provider} API 키를 먼저 입력하세요."
if "auto_capture" not in st.session_state:
    st.session_state.auto_capture = False
if "pending_captures" not in st.session_state:
    st.session_state.pending_captures = []  # [(img, b64, label), ...]

# 직전 run에서 답변이 완료됐으면 알림 (rerun 직전 토스트는 사라지므로 플래그로 전달)
if st.session_state.pop("_toast_done", False):
    st.toast("✅ 답변 완료", icon="✅")

# 다중 탭: tabs = { tab_id: { session_data } }
if "tabs" not in st.session_state:
    st.session_state.tabs = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = None  # 히스토리 열람 중인 session_id

# Streamlit은 클릭마다 전체 rerun → 매번 DB 풀스캔하지 않도록 목록을 짧게 캐시.
# 저장/삭제 시 _safe_save_session/_delete_session 이 즉시 무효화한다.
@st.cache_data(ttl=30, show_spinner=False)
def _cached_list_sessions():
    return list_sessions()


# 시작 시 탭이 없으면 → 기존 active 세션 복원 시도, 없으면 새로 생성
if not st.session_state.tabs and st.session_state.viewing_history is None:
    # DB/로컬에서 active 상태 세션 복원
    _restored = False
    try:
        all_sessions = _cached_list_sessions()
        for s_info in all_sessions:
            if s_info.get("status") == "active" and s_info.get("msg_count", 0) > 0:
                full_sess = load_session(s_info["id"])
                if full_sess and full_sess.get("messages"):
                    st.session_state.tabs[full_sess["id"]] = full_sess
                    if not st.session_state.active_tab:
                        st.session_state.active_tab = full_sess["id"]
                    _restored = True
    except Exception:
        pass

    if not _restored:
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

# 워치리스트 페이지에서 "분석 열기"로 넘어온 경우 → 해당 심볼 탭 활성화/생성
_open_sym = st.session_state.pop("_open_symbol", None)
if _open_sym:
    _exist = next((tid for tid, s in st.session_state.tabs.items()
                   if s.get("symbol") == _open_sym), None)
    if _exist:
        st.session_state.active_tab = _exist
    else:
        _ns = create_session(symbol=_open_sym, interval="15분")
        st.session_state.tabs[_ns["id"]] = _ns
        st.session_state.active_tab = _ns["id"]
    st.session_state.viewing_history = None


def _safe_save_session(session):
    """세션을 안전하게 저장 (Supabase + 로컬) + 목록 캐시 무효화"""
    save_session(session)
    _cached_list_sessions.clear()


def _delete_session(session_id):
    """세션 삭제 (Supabase + 로컬) + 목록 캐시 무효화"""
    delete_session(session_id)
    _cached_list_sessions.clear()


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
    """ISO 시간 문자열을 날짜+시간 형식으로"""
    try:
        dt = datetime.fromisoformat(iso_str)
        now = datetime.now()
        if dt.year == now.year:
            return dt.strftime("%m/%d %H:%M")
        else:
            return dt.strftime("%Y/%m/%d %H:%M")
    except Exception:
        return ""


def _close_position(sess):
    """포지션 종료 — 대화가 있으면 AI 요약 후 closed 저장, 없으면 탭만 닫기.
    (상단/하단 종료 버튼 공용 핸들러)"""
    if sess.get("messages"):
        with st.spinner("🤖 거래 분석 중..."):
            try:
                summary = analyze_trade_summary(
                    api_key=_active_api_key(),
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

    # 탭에서 제거 (대화 없이 종료 → 그냥 탭 닫기)
    tab_id = sess["id"]
    del st.session_state.tabs[tab_id]
    st.session_state.active_tab = (
        list(st.session_state.tabs.keys())[0] if st.session_state.tabs else None
    )
    st.rerun()


def _scroll_chat_bottom():
    """채팅 하단으로 자동 스크롤 — 긴 대화에서 새 메시지/답변이 화면 밖에
    그려져 '아무 반응 없음'처럼 보이는 문제 해결"""
    components.html(
        """<script>
        const doc = window.parent.document;
        const el = doc.querySelector('section[data-testid="stMain"]')
                || doc.querySelector('[data-testid="stAppViewContainer"]')
                || doc.querySelector('section.main');
        if (el) { el.scrollTo({top: el.scrollHeight, behavior: 'smooth'}); }
        </script>""",
        height=0,
    )


@st.cache_data(ttl=10, show_spinner=False)
def _current_price(symbol):
    """현재가 — 1분봉 마지막 종가 (10초 캐시로 rerun 비용 절감)"""
    from core.market_data import fetch_klines
    return fetch_klines(symbol, "1m", 2)[-1]["close"]


@st.cache_data(ttl=60, show_spinner=False)
def _machine_context(symbol):
    """기계 판정 스냅샷 (60초 캐시) — 캡쳐 채팅/지지저항/자유 질문 모두
    RSI 파동 엔진과 같은 기준(레짐·게이트·바텀라인·레벨 맵) 위에서 답하게 한다."""
    try:
        return format_machine_context(symbol, analyze_rsi_wave(symbol))
    except Exception:
        return ""


def _liq_price(direction, entry, qty, margin, mmr=0.005):
    """예상 청산가 (USDT-M 무기한 단순화 모델, 유지증거금률 0.5% 가정).

    격리: margin = 입력 증거금 / 교차: margin = 계좌 전체(설정값) 근사.
    저레버리지로 청산가가 0 이하(롱)면 None — 사실상 청산 없음.
    """
    try:
        notional = entry * qty
        if not entry or not qty or not margin or notional <= 0:
            return None
        im = margin / notional  # = 1/레버리지
        liq = entry * (1 - im + mmr) if direction == "롱" else entry * (1 + im - mmr)
        return liq if liq > 0 else None
    except Exception:
        return None


def _position_pnl(p, cur):
    """포지션 dict + 현재가 → (PnL%, 손절까지%, 목표까지%)"""
    entry = p.get("entry")
    if not entry or not cur:
        return None, None, None
    if p.get("direction") == "롱":
        pnl = (cur - entry) / entry * 100
    else:
        pnl = (entry - cur) / entry * 100
    to_stop = (p["stop"] - cur) / cur * 100 if p.get("stop") else None
    to_target = (p["target"] - cur) / cur * 100 if p.get("target") else None
    return pnl, to_stop, to_target


def _position_context(sess):
    """AI 컨텍스트에 주입할 사용자 보유 포지션 블록 (없으면 '')"""
    p = sess.get("position")
    if not p:
        return ""
    try:
        cur = _current_price(sess.get("symbol") or "BTCUSDT")
    except Exception:
        cur = None
    pnl, _, _ = _position_pnl(p, cur)
    parts = [f"방향 {p.get('direction')}", f"진입가 {p.get('entry')}"]
    if p.get("qty"):
        parts.append(f"수량 {p['qty']:,.6g}개")
    if p.get("stop"):
        parts.append(f"손절 {p['stop']}")
    if p.get("target"):
        parts.append(f"목표 {p['target']}")
    _eff_m = (st.session_state.account_size
              if p.get("margin_mode") == "교차" and st.session_state.account_size > 0
              else p.get("margin"))
    _liq = _liq_price(p.get("direction"), p.get("entry"), p.get("qty"), _eff_m)
    if _liq:
        parts.append(f"예상 청산가 {_liq:,.6g} ({p.get('margin_mode', '격리')})")
    if cur is not None and pnl is not None:
        parts.append(f"현재가 {cur} (PnL {pnl:+.2f}%)")
    return (
        "\n\n📌 사용자 실제 보유 포지션: " + " | ".join(parts) +
        "\n(이 포지션 기준으로 손절/익절/홀딩 관점을 포함해 답하라. "
        "단, 데이터가 포지션과 반대 방향이면 동조하지 말고 분명히 경고할 것)"
    )


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
        for tab_id, sess_item in list(st.session_state.tabs.items()):
            label = _get_tab_label(sess_item)
            is_active = (tab_id == st.session_state.active_tab and
                         st.session_state.viewing_history is None)
            prefix = "▸ " if is_active else "  "
            sb_col1, sb_col2 = st.columns([5, 1])
            with sb_col1:
                if st.button(f"{prefix}{label}", key=f"tab_{tab_id}", use_container_width=True):
                    st.session_state.active_tab = tab_id
                    st.session_state.viewing_history = None
                    st.rerun()
            with sb_col2:
                if st.button("✕", key=f"del_{tab_id}"):
                    _delete_session(tab_id)
                    del st.session_state.tabs[tab_id]
                    if st.session_state.active_tab == tab_id:
                        if st.session_state.tabs:
                            st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
                        else:
                            st.session_state.active_tab = None
                    st.rerun()

    # ── 모바일 분석 리포트 (읽기 전용 — 탭 복원 안 함) ──
    history = _cached_list_sessions()
    report_history = [s for s in history if s.get("status") == "report"]
    if report_history:
        st.markdown('<div class="history-label">🌊 분석 리포트 (모바일)</div>', unsafe_allow_html=True)
        for s in report_history[:15]:
            sym = s.get("symbol") or "?"
            intv = s.get("interval") or ""
            time_str = _format_time(s.get("created_at", ""))
            btn_label = f"🌊 {sym} {intv}  {time_str}".replace("  ", " ").strip()
            if st.session_state.viewing_history == s["id"]:
                btn_label = "▸ " + btn_label
            rp_col1, rp_col2 = st.columns([5, 1])
            with rp_col1:
                if st.button(btn_label, key=f"report_{s['id']}", use_container_width=True):
                    st.session_state.viewing_history = s["id"]
                    st.rerun()
            with rp_col2:
                if st.button("✕", key=f"delreport_{s['id']}"):
                    _delete_session(s["id"])
                    # 보던 리포트/열어둔 탭이면 함께 정리
                    if st.session_state.viewing_history == s["id"]:
                        st.session_state.viewing_history = None
                    if s["id"] in st.session_state.tabs:
                        del st.session_state.tabs[s["id"]]
                        if st.session_state.active_tab == s["id"]:
                            st.session_state.active_tab = (
                                list(st.session_state.tabs.keys())[0]
                                if st.session_state.tabs else None
                            )
                    st.rerun()

    # ── 거래 히스토리 ──
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

    # ── 페이지 바로가기 ──
    st.markdown('<div class="history-label">도구</div>', unsafe_allow_html=True)
    try:
        st.page_link("pages/1_워치리스트.py", label="📊 워치리스트")
        st.page_link("pages/2_신호성적표.py", label="📈 신호 성적표")
        st.page_link("pages/3_백테스트.py", label="🧪 백테스트")
    except Exception:
        # pages/ 폴더가 생기기 전에 시작된 서버는 페이지 레지스트리가 비어
        # KeyError가 난다(업데이트 직후 구 프로세스) → 재시작 안내만 표시
        st.caption("⚠️ 새 페이지 로드 실패 — 앱을 완전히 종료 후 다시 실행하세요.")

    # ── 설정 (하단 접이식) ──
    st.markdown("---")
    with st.expander("⚙️ 설정", expanded=False):
        api_key = st.text_input("OpenAI API 키", type="password",
                                value=st.session_state.api_key, placeholder="sk-...")
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            _persist_config()

        claude_key = st.text_input("Claude(Anthropic) API 키", type="password",
                                   value=st.session_state.claude_api_key,
                                   placeholder="sk-ant-...")
        if claude_key != st.session_state.claude_api_key:
            st.session_state.claude_api_key = claude_key
            _persist_config()

        MODELS = ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"] + CLAUDE_MODELS
        model = st.selectbox("AI 모델", MODELS,
                             index=MODELS.index(st.session_state.model) if st.session_state.model in MODELS else 0)
        if model != st.session_state.model:
            st.session_state.model = model
            _persist_config()

        st.markdown("**리스크 설정**")
        acct = st.number_input("계좌 크기 (USDT)", min_value=0.0,
                               value=float(st.session_state.account_size), step=100.0)
        risk = st.number_input("거래당 리스크 (%)", min_value=0.1, max_value=10.0,
                               value=float(st.session_state.risk_pct), step=0.1)
        if acct != st.session_state.account_size or risk != st.session_state.risk_pct:
            st.session_state.account_size = acct
            st.session_state.risk_pct = risk
            _persist_config()

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

        # 모바일 분석 리포트는 거래 기록과 다르게 표시
        is_report = hist_data.get("status") == "report"
        if is_report:
            result_icon = "🌊"

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
                _delete_session(st.session_state.viewing_history)
                st.session_state.viewing_history = None
                st.rerun()
            if st.button("← 돌아가기", key="back_hist"):
                st.session_state.viewing_history = None
                st.rerun()

        # 모바일 분석 리포트 → PC에서 그 맥락 그대로 대화 이어가기
        # 읽기 전용 세션을 활성 탭으로 승격하면, 아래쪽 채팅 입력/핸들러가
        # 그대로 동작하여 실시간 데이터 기반으로 대화를 이어가고 Supabase에
        # 저장한다(모바일과 양방향 동기화).
        if is_report:
            if st.button("💬 이 분석으로 대화 이어가기", key="continue_report",
                         use_container_width=True, type="primary"):
                # 리포트 세션은 interval 이 비어있을 수 있다. 비어 있으면 채팅 시
                # 타임프레임 목록이 비어 실시간 데이터 수집이 안 되므로 기본값 보정.
                if not hist_data.get("interval"):
                    hist_data["interval"] = "15분"
                if not hist_data.get("symbol"):
                    hist_data["symbol"] = "BTCUSDT"
                st.session_state.tabs[hist_data["id"]] = hist_data
                st.session_state.active_tab = hist_data["id"]
                st.session_state.viewing_history = None
                st.rerun()

        _banner_txt = ("📖 읽기 전용 — '대화 이어가기'를 누르면 이 분석으로 채팅할 수 있어요" if is_report
                       else "📖 읽기 전용 — 과거 거래 기록을 열람 중입니다")
        st.markdown(f'<div class="readonly-banner">{_banner_txt}</div>',
                    unsafe_allow_html=True)

        # 대화 내용 표시
        for msg in hist_data.get("messages", []):
            _av = "🧾" if msg.get("trade_event") else ("🤖" if msg["role"] == "assistant" else "👤")
            with st.chat_message(msg["role"], avatar=_av):
                if msg.get("rsi_wave_html"):
                    _wh = 960 if "진입 지도" in msg["rsi_wave_html"] else 500
                    components.html(msg["rsi_wave_html"], height=_wh, scrolling=False)
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
col_title, col_close, col_delete = st.columns([6, 1, 1])
with col_title:
    sym_display = sess.get("symbol") or "종목 감지 중..."
    intv_display = sess.get("interval") or ""
    st.markdown(f"## ◈ {sym_display} {intv_display}")
    st.caption(f"모델: {st.session_state.model} · 세션: {_format_time(sess.get('created_at', ''))}")

with col_close:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📁 방 종료·요약", key="close_position", use_container_width=True,
                 help="대화를 AI로 요약해 거래 히스토리에 보관하고 이 방을 닫습니다. "
                      "포지션 청산은 아래 포지션 패널에서 하세요."):
        _close_position(sess)

with col_delete:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 삭제", key="delete_current", use_container_width=True):
        tab_id = sess["id"]
        _delete_session(tab_id)
        del st.session_state.tabs[tab_id]
        if st.session_state.tabs:
            st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
        else:
            st.session_state.active_tab = None
        st.rerun()

# ── 종목 직접 입력 / 수정 ──
# 자동인식(TradingView 창)이 안 잡히거나(숫자심볼·비USDT·창 미실행), 다른
# 종목을 분석하고 싶을 때 수동으로 지정. 입력하면 자동인식이 이를 덮어쓰지
# 않는다(상단 detect_chart_info는 symbol 이 비어 있을 때만 동작).
with st.expander("🔧 종목 직접 입력 / 수정 (자동인식이 안 잡힐 때)", expanded=not sess.get("symbol")):
    msc1, msc2 = st.columns([3, 1])
    with msc1:
        _manual_sym = st.text_input(
            "종목", value=sess.get("symbol", ""),
            key=f"sym_in_{sess['id']}", label_visibility="collapsed",
            placeholder="예: ZECUSDT, 1000PEPEUSDT",
        )
    with msc2:
        if st.button("적용", key=f"sym_apply_{sess['id']}", use_container_width=True):
            _v = (_manual_sym or "").strip().upper()
            if _v:
                sess["symbol"] = _v
                if not sess.get("interval"):
                    sess["interval"] = "15분"   # 수동 지정 시 데이터 수집 기본 TF 보정
                _safe_save_session(sess)
                st.rerun()


# (포지션 패널은 채팅 입력 바로 위로 이동 — 진입/청산이 이 방의 거래 기록으로 남음)


# 퀵 분석 프롬프트 → 짧은 표시 매핑
_PROMPT_DISPLAY_MAP = {
    "[종합 차트 분석]": "📊 종합 차트 분석",
    "[매매 전략 수립]": "💡 매매 전략 분석",
    "[지지/저항 & 매물대 분석]": "📐 지지/저항 & 매물대 분석",
    "[추세 종합 판단]": "📈 추세 종합 판단",
}

def _shorten_prompt(content):
    """퀵 분석 프롬프트를 짧은 라벨로 변환"""
    for tag, label in _PROMPT_DISPLAY_MAP.items():
        if content.strip().startswith(tag):
            return label
    return content

# ── 이전 메시지 표시 ──
for msg in sess.get("messages", []):
    _av = "🧾" if msg.get("trade_event") else ("🤖" if msg["role"] == "assistant" else "👤")
    with st.chat_message(msg["role"], avatar=_av):
        if msg.get("rsi_wave_html"):
            _wh = 960 if "진입 지도" in msg["rsi_wave_html"] else 500
            components.html(msg["rsi_wave_html"], height=_wh, scrolling=False)
        display = _shorten_prompt(msg["content"]) if msg["role"] == "user" else msg["content"]
        st.markdown(display)
        if msg.get("image"):
            st.image(msg["image"], caption="분석한 차트", use_container_width=True)

# ── 첨부된 캡쳐 미리보기 ──
if st.session_state.pending_captures:
    st.markdown(f"**📎 첨부된 캡쳐 ({len(st.session_state.pending_captures)}장)**")
    cap_cols = st.columns(min(len(st.session_state.pending_captures), 4))
    remove_idx = None
    for ci, item in enumerate(st.session_state.pending_captures):
        cap_img = item[0]
        cap_label = item[2]
        with cap_cols[ci % len(cap_cols)]:
            st.image(cap_img, caption=cap_label, use_container_width=True)
            if st.button("✕ 삭제", key=f"rm_cap_{ci}"):
                remove_idx = ci
    if remove_idx is not None:
        st.session_state.pending_captures.pop(remove_idx)
        st.rerun()

# ── 빠른 분석 버튼 (항상 표시) ──
st.markdown("---")
cols = st.columns(3)

# (버튼 라벨, 채팅 표시 텍스트, AI 전체 프롬프트)
_QUICK_ITEMS = [
    ("📐 지지/저항", "📐 지지/저항 & 매물대 분석", """[지지/저항 & 매물대 분석] 아래 모든 타임프레임의 실시간 데이터를 기반으로 주요 지지/저항 레벨을 분석해줘.

각 타임프레임에서 파악되는:
1. 주요 지지선 (가격대 + 근거)
2. 주요 저항선 (가격대 + 근거)
3. 매물대 (거래량이 집중된 가격 구간)
4. EMA/볼린저밴드가 형성하는 동적 지지/저항

최종 정리:
- 강한 지지 구간 TOP 3
- 강한 저항 구간 TOP 3
- 현재가 기준 가장 가까운 지지/저항

주의: 함께 제공되는 '기계 판정 스냅샷'의 핵심 레벨 맵과 정합성을 유지하고,
각 레벨이 어떤 컨플루언스(EMA/VWAP/BB/최근 고저)에서 나왔는지 표기할 것."""),
]

# 🌊 RSI 파동 분석 버튼
with cols[0]:
    if st.button("🌊 RSI 파동 분석", key="rsi_wave_btn", use_container_width=True):
        st.session_state._pending_rsi_wave = True
        st.rerun()

# 기존 퀵 분석 버튼 (지지/저항)
for i, (btn_label, display_label, full_prompt) in enumerate(_QUICK_ITEMS):
    with cols[i + 1]:
        if st.button(btn_label, key=f"quick_{i}", use_container_width=True):
            st.session_state._pending_msg = full_prompt
            st.session_state._pending_display = display_label
            st.session_state._pending_force_multi = True
            st.rerun()

# 📸 캡쳐 버튼
with cols[2]:
    cap_count = len(st.session_state.pending_captures)
    cap_label = f"📸 캡쳐 ({cap_count})" if cap_count else "📸 캡쳐"
    if st.button(cap_label, key="quick_capture", use_container_width=True):
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
                label = f"{sym or '차트'} {tf or ''} ({size // 1024}KB)"
                st.session_state.pending_captures.append((img, b64, label, tf or ''))
            else:
                st.toast(f"⚠️ {title}")
        st.rerun()


# Pending 메시지 처리
pending = st.session_state.pop("_pending_msg", None)
pending_display = st.session_state.pop("_pending_display", None)
force_multi = st.session_state.pop("_pending_force_multi", False)
pending_rsi_wave = st.session_state.pop("_pending_rsi_wave", False)

# ── 🌊 RSI 파동 분석 처리 ──
if pending_rsi_wave and _active_api_key():
    symbol = sess.get("symbol", "BTCUSDT")

    # 사용자 메시지 추가
    user_prompt = "🌊 RSI 파동 분석"
    sess["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)
    st.toast("🌊 RSI 파동 분석 시작 — 데이터 수집 중", icon="🤖")
    _scroll_chat_bottom()

    # 데이터 수집 + 분석
    tf_label = " / ".join(WAVE_TIMEFRAMES)
    with st.spinner(f"🌊 RSI 파동 데이터 수집 중 ({tf_label})..."):
        rsi_results = analyze_rsi_wave(symbol)

    # 신호 로깅 (가중치 검증용 — best-effort, 실패해도 분석 진행)
    try:
        from core.signal_logger import log_rsi_wave_signals
        log_rsi_wave_signals(symbol, rsi_results)
    except Exception:
        pass

    # SVG + 카드 + 종합 판정 생성
    svg_html = generate_wave_svg(rsi_results)
    ladder_frag = generate_price_ladder_svg(rsi_results)  # 가격 레벨 & 진입 지도
    # 파동 맵 HTML 문서에 가격 지도 프래그먼트를 주입 (한 iframe에 함께 렌더)
    if ladder_frag:
        combined_html = svg_html.replace("</body>", ladder_frag + "\n</body>")
    else:
        combined_html = svg_html
    tf_cards = generate_tf_cards(rsi_results)
    summary_text = generate_summary_text(rsi_results)

    # AI용 데이터 포맷팅 (+보유 포지션 컨텍스트)
    ai_prompt_text = format_rsi_wave_for_ai(symbol, rsi_results) + _position_context(sess)

    # AI 분석 스트리밍
    with st.chat_message("assistant", avatar="🤖"):
        # 1) SVG 파동 맵 + 가격 레벨/진입 지도
        components.html(combined_html, height=960 if ladder_frag else 500, scrolling=False)
        # 2) 종합 판정
        st.markdown(summary_text)
        st.markdown("---")
        # 3) 타임프레임별 상세 카드
        with st.expander("📋 타임프레임별 상세 데이터", expanded=False):
            st.markdown(tf_cards)
        st.markdown("---")
        # 4) AI 코멘터리 스트리밍
        st.caption(f"🤖 AI 분석 ({st.session_state.model})")
        try:
            # RSI 파동용 메시지 구성 — AI에게 분석 데이터를 user 메시지로 전달
            ai_messages = list(sess["messages"])
            ai_messages[-1] = {"role": "user", "content": ai_prompt_text}

            ai_response = st.write_stream(
                analyze_chart(
                    api_key=_active_api_key(),
                    model=st.session_state.model,
                    messages=ai_messages,
                    system_prompt_override=RSI_WAVE_SYSTEM_PROMPT,
                )
            )
        except Exception as e:
            ai_response = f"⚠️ AI 분석 오류: {str(e)}"
            st.error(ai_response)

    # 세션에 저장
    full_content = summary_text + "\n\n" + str(ai_response) if ai_response else summary_text
    sess["messages"].append({
        "role": "assistant",
        "content": full_content,
        "rsi_wave_html": combined_html,
    })
    _safe_save_session(sess)
    st.session_state["_toast_done"] = True
    st.rerun()

elif pending_rsi_wave and not _active_api_key():
    st.warning(_key_warning())

# ═══════════════════════════════════════════════
# 🎯 포지션 패널 — 채팅 입력 바로 위 (진입/수정/청산이 이 방의 기록으로 남음)
# ═══════════════════════════════════════════════

def _fmt_duration(start_iso):
    """보유 시간 표기"""
    try:
        sec = (datetime.now() - datetime.fromisoformat(start_iso)).total_seconds()
        h, m = int(sec // 3600), int(sec % 3600 // 60)
        return f"{h}시간 {m}분" if h else f"{m}분"
    except Exception:
        return "?"


def _append_trade_event(sess, text):
    """거래 이벤트를 채팅 기록에 남긴다 — AI 컨텍스트 + 복기 자료가 된다"""
    sess["messages"].append({"role": "user", "content": text, "trade_event": True})


_pos = sess.get("position")
_cur = None
if _pos:
    try:
        _cur = _current_price(sess.get("symbol") or "BTCUSDT")
    except Exception:
        _cur = None
    if _cur:
        _pnl, _to_stop, _to_target = _position_pnl(_pos, _cur)
        # 수량이 있으면 USDT 손익 병기
        _pnl_usdt = None
        if _pos.get("qty") and _pnl is not None:
            _diff = (_cur - _pos["entry"]) if _pos["direction"] == "롱" else (_pos["entry"] - _cur)
            _pnl_usdt = _diff * _pos["qty"]
        _pnl_txt = None
        if _pnl is not None:
            _pnl_txt = f"{_pnl:+.2f}%" + (f" ({_pnl_usdt:+,.1f}$)" if _pnl_usdt is not None else "")
        # 예상 청산가 (교차면 계좌 전체를 담보로 근사)
        _eff_margin = (st.session_state.account_size
                       if _pos.get("margin_mode") == "교차" and st.session_state.account_size > 0
                       else _pos.get("margin"))
        _liq = _liq_price(_pos["direction"], _pos["entry"], _pos.get("qty"), _eff_margin)

        pm1, pm2, pm3, pm4, pm5, pm6 = st.columns([1.1, 1.1, 0.9, 0.9, 1.1, 0.5])
        pm1.metric(f"🎯 {_pos['direction']} 진입가", f"{_pos['entry']:,.6g}",
                   f"보유 {_fmt_duration(_pos.get('opened_at', ''))}", delta_color="off")
        pm2.metric("현재가", f"{_cur:,.6g}", _pnl_txt)
        pm3.metric("손절까지", f"{_to_stop:+.2f}%" if _to_stop is not None else "-")
        pm4.metric("목표까지", f"{_to_target:+.2f}%" if _to_target is not None else "-")
        pm5.metric("예상 청산가", f"{_liq:,.6g}" if _liq else "-",
                   f"{(_liq - _cur) / _cur * 100:+.1f}%" if _liq else None, delta_color="off")
        with pm6:
            if st.button("🔄", key=f"pos_refresh_{sess['id']}", help="현재가 즉시 갱신",
                         use_container_width=True):
                _current_price.clear()
                st.rerun()
            if st.button("✏️", key=f"pos_edit_btn_{sess['id']}", help="포지션 수정 패널 열기",
                         use_container_width=True):
                st.session_state[f"pos_edit_{sess['id']}"] = True
                st.rerun()
        if _pos.get("stop") and _pos.get("entry"):
            _hit = ((_pos["direction"] == "롱" and _cur <= _pos["stop"]) or
                    (_pos["direction"] == "숏" and _cur >= _pos["stop"]))
            _init_dist = abs(_pos["entry"] - _pos["stop"])
            if _hit:
                st.error("🚨 현재가가 손절가를 넘었습니다!")
            elif _init_dist > 0 and abs(_cur - _pos["stop"]) < _init_dist * 0.3:
                st.warning(f"⚠️ 손절가까지 {_to_stop:+.2f}% — 초기 손절거리의 70% 이상 소진됨")

with st.expander("🎯 포지션 기록 / 리스크 계산기"
                 + (f" — {_pos['direction']} 보유 중" if _pos else " — 미보유"),
                 expanded=st.session_state.get(f"pos_edit_{sess['id']}", False)):
    pc1, pc2, pc3, pc4 = st.columns([0.8, 1.2, 1.2, 1.2])
    with pc1:
        _dir = st.selectbox("방향", ["롱", "숏"],
                            index=0 if not _pos or _pos.get("direction") == "롱" else 1,
                            key=f"pos_dir_{sess['id']}")
    with pc2:
        _entry = st.number_input("진입가", min_value=0.0, format="%.6f",
                                 value=float((_pos or {}).get("entry") or 0),
                                 key=f"pos_entry_{sess['id']}")
    with pc3:
        _stop = st.number_input("손절가", min_value=0.0, format="%.6f",
                                value=float((_pos or {}).get("stop") or 0),
                                key=f"pos_stop_{sess['id']}")
    with pc4:
        _target = st.number_input("목표가", min_value=0.0, format="%.6f",
                                  value=float((_pos or {}).get("target") or 0),
                                  key=f"pos_target_{sess['id']}")

    pq1, pq2, pq3 = st.columns([1.2, 1.2, 0.8])
    with pq1:
        _qty = st.number_input("수량 (코인 개수)", min_value=0.0, format="%.6f",
                               value=float((_pos or {}).get("qty") or 0),
                               key=f"pos_qty_{sess['id']}")
    with pq2:
        _margin = st.number_input("증거금 (USDT)", min_value=0.0, format="%.2f",
                                  value=float((_pos or {}).get("margin") or 0),
                                  key=f"pos_margin_{sess['id']}")
    with pq3:
        _mode = st.selectbox("마진 모드", ["격리", "교차"],
                             index=0 if (_pos or {}).get("margin_mode", "격리") == "격리" else 1,
                             key=f"pos_mode_{sess['id']}")

    # 레버리지/예상 청산가 미리보기 (교차는 계좌 전체를 담보로 근사)
    if _entry > 0 and _qty > 0:
        _notional = _entry * _qty
        _eff_m = (st.session_state.account_size
                  if _mode == "교차" and st.session_state.account_size > 0 else _margin)
        if _eff_m > 0:
            _lev = _notional / _eff_m
            _liq_prev = _liq_price(_dir, _entry, _qty, _eff_m)
            _liq_txt = (f"예상 청산가 ≈ **{_liq_prev:,.6g}**" if _liq_prev
                        else "청산가 없음(저레버리지)")
            _mode_note = " · 교차=계좌 전체 담보 근사" if _mode == "교차" else ""
            st.caption(f"📊 명목 {_notional:,.1f} USDT · 레버리지 {_lev:.1f}x ({_mode}) · "
                       f"{_liq_txt} (유지증거금 0.5% 가정{_mode_note})")

    # 리스크 계산기 — 계좌×리스크% ÷ 손절거리 = 권장 사이즈
    if _entry > 0 and _stop > 0 and _entry != _stop:
        _stop_dist = abs(_entry - _stop) / _entry * 100
        _rr_txt = ""
        if _target > 0:
            _r = abs(_target - _entry) / abs(_entry - _stop)
            _rr_txt = f" · 손익비 {_r:.2f}R"
        if st.session_state.account_size > 0:
            _risk_amt = st.session_state.account_size * st.session_state.risk_pct / 100
            _notional = _risk_amt / (_stop_dist / 100)
            _qty = _notional / _entry
            st.info(f"📐 손절거리 {_stop_dist:.2f}%{_rr_txt} · "
                    f"리스크 {_risk_amt:,.1f} USDT({st.session_state.risk_pct}%) → "
                    f"권장 포지션 **{_notional:,.0f} USDT** (수량 ≈ {_qty:.6g}, "
                    f"계좌 대비 {_notional / st.session_state.account_size:.1f}x)")
        else:
            st.caption(f"📐 손절거리 {_stop_dist:.2f}%{_rr_txt} — "
                       f"⚙️ 설정에서 계좌 크기를 입력하면 권장 사이즈를 계산합니다.")

    pb1, pb2, pb3 = st.columns(3)
    with pb1:
        _save_label = "💾 진입 기록" if not _pos else "✏️ 수정 기록"
        if st.button(_save_label, key=f"pos_save_{sess['id']}", use_container_width=True):
            if _entry > 0:
                is_new = not _pos
                sess["position"] = {
                    "direction": _dir, "entry": _entry,
                    "stop": _stop or None, "target": _target or None,
                    "qty": _qty or None, "margin": _margin or None, "margin_mode": _mode,
                    "opened_at": (_pos or {}).get("opened_at") or datetime.now().isoformat(),
                }
                evt = (f"🧾 **포지션 {'오픈' if is_new else '수정'}** — {_dir} @{_entry:,.6g}"
                       + (f" · {_qty:,.6g}개" if _qty else "")
                       + (f" · 증거금 {_margin:,.1f}$ {_mode}" if _margin else "")
                       + (f" · 손절 {_stop:,.6g}" if _stop else "")
                       + (f" · 목표 {_target:,.6g}" if _target else ""))
                _append_trade_event(sess, evt)
                _safe_save_session(sess)
                st.session_state[f"pos_edit_{sess['id']}"] = False
                st.rerun()
            else:
                st.toast("⚠️ 진입가를 입력하세요")
    with pb2:
        if st.button("✅ 청산 기록", key=f"pos_close_{sess['id']}", use_container_width=True,
                     disabled=not _pos,
                     help="현재가 기준 실현 PnL을 계산해 이 방의 거래 기록으로 남기고 포지션을 정리합니다. 방은 유지됩니다."):
            try:
                exit_p = _cur or _current_price(sess.get("symbol") or "BTCUSDT")
            except Exception:
                exit_p = None
            pnl = None
            pnl_usdt = None
            if exit_p:
                diff = ((exit_p - _pos["entry"]) if _pos["direction"] == "롱"
                        else (_pos["entry"] - exit_p))
                pnl = diff / _pos["entry"] * 100
                if _pos.get("qty"):
                    pnl_usdt = diff * _pos["qty"]
            dur = _fmt_duration(_pos.get("opened_at", ""))
            icon = "🟢" if (pnl or 0) >= 0 else "🔴"
            evt = (f"🧾 {icon} **포지션 청산** — {_pos['direction']} {_pos['entry']:,.6g}"
                   + (f" → {exit_p:,.6g} (**{pnl:+.2f}%**" if pnl is not None else " (청산가 미확인")
                   + (f", {pnl_usdt:+,.1f}$" if pnl_usdt is not None else "")
                   + (")" if pnl is not None or pnl_usdt is not None else ")")
                   + f" · 보유 {dur}")
            _append_trade_event(sess, evt)
            sess.setdefault("trades", []).append({
                "direction": _pos["direction"], "entry": _pos["entry"],
                "stop": _pos.get("stop"), "target": _pos.get("target"),
                "qty": _pos.get("qty"), "margin": _pos.get("margin"),
                "margin_mode": _pos.get("margin_mode"),
                "exit": exit_p, "pnl_pct": round(pnl, 3) if pnl is not None else None,
                "pnl_usdt": round(pnl_usdt, 2) if pnl_usdt is not None else None,
                "opened_at": _pos.get("opened_at"), "closed_at": datetime.now().isoformat(),
            })
            sess.pop("position", None)
            _safe_save_session(sess)
            st.session_state[f"pos_edit_{sess['id']}"] = False
            st.rerun()
    with pb3:
        if st.button("✖ 취소 (기록 없음)", key=f"pos_clear_{sess['id']}", use_container_width=True,
                     disabled=not _pos, help="잘못 입력한 포지션을 기록 없이 제거합니다"):
            sess.pop("position", None)
            _safe_save_session(sess)
            st.session_state[f"pos_edit_{sess['id']}"] = False
            st.rerun()

    # 이 방의 거래 성적 (복기용)
    _trades = sess.get("trades") or []
    if _trades:
        _decided = [t for t in _trades if t.get("pnl_pct") is not None]
        _wins = sum(1 for t in _decided if t["pnl_pct"] > 0)
        _tot = sum(t["pnl_pct"] for t in _decided)
        _usd_list = [t["pnl_usdt"] for t in _decided if t.get("pnl_usdt") is not None]
        _usd_txt = f" ({sum(_usd_list):+,.1f}$)" if _usd_list else ""
        st.caption(f"🧾 이 방의 거래: {len(_trades)}건 · 승 {_wins}/{len(_decided)} · "
                   f"누적 **{_tot:+.2f}%**{_usd_txt}")
        for t in _trades[-5:]:
            _ic = "🟢" if (t.get("pnl_pct") or 0) >= 0 else "🔴"
            _tm = _format_time(t.get("closed_at", ""))
            st.caption(f"  {_ic} {t['direction']} {t['entry']:,.6g} → "
                       f"{t['exit']:,.6g} ({t['pnl_pct']:+.2f}%) · {_tm}"
                       if t.get("exit") and t.get("pnl_pct") is not None
                       else f"  {_ic} {t['direction']} {t['entry']:,.6g} · {_tm}")

# ── 채팅 입력 (자동 캡쳐 체크박스 + 입력) ──
st.session_state.auto_capture = st.checkbox(
    "📸 자동캡쳐 — 질문 시 현재 TradingView 차트를 자동으로 캡쳐합니다",
    value=st.session_state.auto_capture, key="auto_capture_cb")
user_input = st.chat_input("차트에 대해 질문하세요...")
prompt = pending or user_input

if prompt:
    if not _active_api_key():
        st.warning(_key_warning())
    else:
        # 이미지 수집: pending_captures 우선, 자동캡쳐 체크 시 캡쳐, 아니면 이미지 없이 진행
        all_images = []  # [(img, b64, label), ...]

        if st.session_state.pending_captures:
            all_images = list(st.session_state.pending_captures)
            st.session_state.pending_captures = []
        elif st.session_state.auto_capture:
            # 자동 캡쳐 체크박스가 켜져 있을 때만 자동으로 현재 차트 캡쳐
            with st.spinner("📸 TradingView 캡쳐 중..."):
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
                    label = f"{sym or '차트'} {tf or ''} ({size // 1024}KB)"
                    all_images.append((img, b64, label, tf or ''))
                else:
                    st.warning(f"⚠️ 캡쳐 실패: {title}")

        # 사용자 메시지 추가 — 퀵 분석은 짧은 라벨로 표시
        display_text = pending_display if pending_display else prompt
        sess["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(display_text)
            if all_images:
                img_cols = st.columns(min(len(all_images), 3))
                for ii, item in enumerate(all_images):
                    cap_img = item[0]
                    cap_label = item[2]
                    with img_cols[ii % len(img_cols)]:
                        st.image(cap_img, caption=cap_label, use_container_width=True)

        # 진행 피드백 — 화면 위치와 무관하게 보이는 토스트 + 하단 스크롤
        st.toast("📨 질문 접수 — 데이터 수집 후 답변을 생성합니다", icon="🤖")
        _scroll_chat_bottom()

        # 실시간 데이터 수집
        symbol = sess.get("symbol", "BTCUSDT")
        interval = sess.get("interval", "15분")

        if force_multi:
            requested_tfs = ["5분", "15분", "1시간", "4시간", "1일"]
        else:
            requested_tfs = parse_requested_timeframes(prompt, interval)

        # 캡쳐된 이미지들의 타임프레임도 데이터 수집에 포함
        if all_images:
            from core.market_data import INTERVAL_MAP
            detected_any = False
            for item in all_images:
                cap_tf = item[3] if len(item) > 3 else ''
                if cap_tf and cap_tf in INTERVAL_MAP:
                    detected_any = True
                    if cap_tf not in requested_tfs:
                        requested_tfs.append(cap_tf)
            # 타임프레임 감지 실패 시 (TradingView 타이틀에 tf 없음) → 주요 타임프레임 수집
            if not detected_any and len(all_images) >= 2:
                requested_tfs = ["5분", "15분", "1시간", "4시간", "1일"]
            # 정렬: 작은 타임프레임 → 큰 타임프레임
            tf_order = ["1분", "3분", "5분", "15분", "30분", "1시간", "2시간", "4시간", "1일", "1주", "1개월"]
            requested_tfs = [tf for tf in tf_order if tf in requested_tfs]

        tf_label = " / ".join(requested_tfs)
        with st.spinner(f"📊 데이터 수집 중 ({tf_label})..."):
            market_data = get_multi_timeframe_context(symbol, requested_tfs, interval)

        # 기계 판정 스냅샷(레짐·게이트·레벨 맵) + 보유 포지션 — 모든 대화의 공통 기준
        with st.spinner("🤖 기계 판정 동기화 중..."):
            market_data = market_data + _machine_context(symbol) + _position_context(sess)

        # AI에 보낼 이미지 (첫 번째 이미지 또는 없음)
        primary_b64 = all_images[0][1] if all_images else None
        primary_img = all_images[0][0] if all_images else None

        # 추가 이미지들의 b64 리스트
        extra_b64_list = [item[1] for item in all_images[1:]] if len(all_images) > 1 else []

        # AI 분석
        with st.chat_message("assistant", avatar="🤖"):
            if all_images:
                st.caption(f"✅ {len(all_images)}장 이미지 첨부됨 — {st.session_state.model}")
            else:
                st.caption("⚠️ 차트 이미지 없이 데이터만으로 분석합니다")

            try:
                response = st.write_stream(
                    analyze_chart(
                        api_key=_active_api_key(),
                        model=st.session_state.model,
                        messages=sess["messages"],
                        image_base64=primary_b64,
                        market_data=market_data,
                        extra_images=extra_b64_list,
                    )
                )
            except Exception as e:
                response = f"⚠️ API 오류: {str(e)}"
                st.error(response)

        sess["messages"].append({
            "role": "assistant",
            "content": str(response) if response else "",
            "image": primary_img,
        })
        _safe_save_session(sess)
        st.session_state["_toast_done"] = True
        st.rerun()
