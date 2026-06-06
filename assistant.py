"""
TradeAI Assistant — 트레이딩 세션 관리 + AI 분석 데스크탑 앱
ChatGPT 스타일 사이드바: 새 대화 / 대화 목록 / 설정
"""
import streamlit as st
import streamlit.components.v1 as components
import json
import copy
from datetime import datetime
from core.capture import capture_tradingview, image_to_base64, parse_window_title, detect_chart_info
from core.market_data import get_market_context, get_multi_timeframe_context, parse_requested_timeframes
from core.ai_client import analyze_chart, analyze_trade_summary
from core.rsi_wave import (
    analyze_rsi_wave, generate_wave_svg, generate_price_ladder_svg, generate_tf_cards,
    generate_summary_text, format_rsi_wave_for_ai,
    RSI_WAVE_SYSTEM_PROMPT, WAVE_TIMEFRAMES,
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
    st.session_state.auto_capture = False
if "pending_captures" not in st.session_state:
    st.session_state.pending_captures = []  # [(img, b64, label), ...]

# 다중 탭: tabs = { tab_id: { session_data } }
if "tabs" not in st.session_state:
    st.session_state.tabs = {}
if "active_tab" not in st.session_state:
    st.session_state.active_tab = None
if "viewing_history" not in st.session_state:
    st.session_state.viewing_history = None  # 히스토리 열람 중인 session_id

# 시작 시 탭이 없으면 → 기존 active 세션 복원 시도, 없으면 새로 생성
if not st.session_state.tabs and st.session_state.viewing_history is None:
    # DB/로컬에서 active 상태 세션 복원
    _restored = False
    try:
        all_sessions = list_sessions()
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
    """세션을 안전하게 저장 (Supabase + 로컬)"""
    save_session(session)


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
                    delete_session(tab_id)
                    del st.session_state.tabs[tab_id]
                    if st.session_state.active_tab == tab_id:
                        if st.session_state.tabs:
                            st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
                        else:
                            st.session_state.active_tab = None
                    st.rerun()

    # ── 모바일 분석 리포트 (읽기 전용 — 탭 복원 안 함) ──
    history = list_sessions()
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
                    delete_session(s["id"])
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

    # ── 신호 정확도 통계 (RSI 파동 가중치 검증) ──
    with st.expander("📈 신호 통계", expanded=False):
        st.caption("RSI 파동 신호의 실제 적중률을 추적합니다.")
        if st.button("평가 갱신 & 통계 보기", use_container_width=True, key="eval_stats"):
            try:
                from core.signal_logger import (
                    evaluate_pending_signals, get_signal_stats, get_weight_suggestions,
                )
                with st.spinner("성숙한 신호 평가 중..."):
                    n_eval = evaluate_pending_signals()
                    st.session_state._signal_stats = get_signal_stats()
                    st.session_state._signal_suggestions = get_weight_suggestions()
                    st.session_state._signal_eval_n = n_eval
            except Exception as e:
                st.session_state._signal_stats = {"error": str(e)}

        stats = st.session_state.get("_signal_stats")
        if stats and not stats.get("error"):
            n_eval = st.session_state.get("_signal_eval_n", 0)
            ov = stats.get("overall", {})
            st.markdown(
                f"**평가 완료: {stats.get('total_evaluated', 0)}건** "
                f"(이번에 {n_eval}건 신규 평가)"
            )
            if ov.get("n"):
                wr = ov.get("win_rate")
                ar = ov.get("avg_return")
                st.markdown(
                    f"- 전체 적중률: **{wr if wr is not None else '-'}%** "
                    f"(n={ov['n']})\n"
                    f"- 평균 실현수익: **{ar if ar is not None else '-'}%** "
                    f"(MFE {ov.get('avg_mfe')}% / MAE {ov.get('avg_mae')}%)"
                )

                def _render_group(title, group):
                    rows = [(k, v) for k, v in group.items() if v.get("n")]
                    if not rows:
                        return
                    lines = [f"\n**{title}**"]
                    lines.append("| 항목 | 적중률 | 평균수익 | n |")
                    lines.append("|---|---|---|---|")
                    for k, v in sorted(rows, key=lambda x: -(x[1].get("n") or 0)):
                        wr = v.get("win_rate")
                        ar = v.get("avg_return")
                        lines.append(
                            f"| {k} | {wr if wr is not None else '-'}% | "
                            f"{ar if ar is not None else '-'}% | {v['n']} |"
                        )
                    st.markdown("\n".join(lines))

                _render_group("신호유형별", stats.get("by_signal_type", {}))
                _render_group("확신등급별", stats.get("by_confidence", {}))
                _render_group("레짐별", stats.get("by_regime", {}))
                _render_group("타임프레임별", stats.get("by_timeframe", {}))
            else:
                st.info("아직 평가된 신호가 없습니다. 호라이즌(예: 1시간봉=24h)이 지나야 평가됩니다.")

            # ── 가중치 조정 제안 (반자동) ──
            sug = st.session_state.get("_signal_suggestions")
            if sug:
                st.markdown("---")
                st.markdown("**💡 가중치 조정 제안**")
                if not sug.get("ready"):
                    st.caption(
                        f"표본 수집 중 — {sug.get('total', 0)}/{sug.get('min_samples', 20)}건. "
                        f"{sug.get('needed', 0)}건 더 모이면 제안이 시작됩니다."
                    )
                elif not sug.get("suggestions"):
                    st.caption("✅ 모든 신호가 정상 범위 — 조정 제안 없음.")
                else:
                    icon_map = {"high": "🔴", "medium": "🟠", "info": "🟢"}
                    for s in sug["suggestions"]:
                        arrow = "⬇️ 하향" if s["direction"] == "DOWN" else "⬆️ 상향"
                        st.markdown(
                            f"{icon_map.get(s['severity'], '•')} **{s['target']}** {arrow}  \n"
                            f"<span style='font-size:12px;color:#9A8B78'>{s['message']}</span>",
                            unsafe_allow_html=True,
                        )
                    st.caption("제안은 참고용입니다. 적용은 직접 판단하세요.")
        elif stats and stats.get("error"):
            st.warning(f"통계 오류: {stats['error']}")


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
                delete_session(st.session_state.viewing_history)
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
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
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

with col_delete:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑 삭제", key="delete_current", use_container_width=True):
        tab_id = sess["id"]
        delete_session(tab_id)
        del st.session_state.tabs[tab_id]
        if st.session_state.tabs:
            st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
        else:
            st.session_state.active_tab = None
        st.rerun()


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
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
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
- 현재가 기준 가장 가까운 지지/저항"""),
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

# ── 하단 포지션 종료 버튼 ──
st.markdown('<div class="close-position-btn">', unsafe_allow_html=True)
if st.button("📤 포지션 종료", key="close_position_bottom", use_container_width=True):
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

        if st.session_state.tabs:
            st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
        else:
            st.session_state.active_tab = None

        st.rerun()
    else:
        tab_id = sess["id"]
        del st.session_state.tabs[tab_id]
        if st.session_state.tabs:
            st.session_state.active_tab = list(st.session_state.tabs.keys())[0]
        else:
            st.session_state.active_tab = None
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# Pending 메시지 처리
pending = st.session_state.pop("_pending_msg", None)
pending_display = st.session_state.pop("_pending_display", None)
force_multi = st.session_state.pop("_pending_force_multi", False)
pending_rsi_wave = st.session_state.pop("_pending_rsi_wave", False)

# ── 🌊 RSI 파동 분석 처리 ──
if pending_rsi_wave and st.session_state.api_key:
    symbol = sess.get("symbol", "BTCUSDT")

    # 사용자 메시지 추가
    user_prompt = "🌊 RSI 파동 분석"
    sess["messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

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

    # AI용 데이터 포맷팅
    ai_prompt_text = format_rsi_wave_for_ai(symbol, rsi_results)

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
                    api_key=st.session_state.api_key,
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
    st.rerun()

elif pending_rsi_wave and not st.session_state.api_key:
    st.warning("🔑 사이드바 설정에서 OpenAI API 키를 먼저 입력하세요.")

# ── 채팅 입력 (자동 캡쳐 체크박스 + 입력) ──
st.session_state.auto_capture = st.checkbox(
    "📸 자동캡쳐 — 질문 시 현재 TradingView 차트를 자동으로 캡쳐합니다",
    value=st.session_state.auto_capture, key="auto_capture_cb")
user_input = st.chat_input("차트에 대해 질문하세요...")
prompt = pending or user_input

if prompt:
    if not st.session_state.api_key:
        st.warning("🔑 사이드바 설정에서 OpenAI API 키를 먼저 입력하세요.")
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
                        api_key=st.session_state.api_key,
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
        st.rerun()
