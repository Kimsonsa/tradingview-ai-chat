"""
RSI 파동 분석 작업 워커 (모바일 → 데스크탑 트리거)

모바일은 Supabase `analysis_jobs` 테이블에 분석 요청을 INSERT만 한다.
데스크탑 앱이 켜져 있으면 이 워커(데몬 스레드)가 pending 작업을 원자적으로
점유하여 analyze_rsi_wave + AI 분석을 실행하고, 결과를 'report' 세션으로
저장한 뒤 작업을 done 처리한다. 모바일은 그 세션을 열어 결과를 본다.

핵심: 모든 분석 로직(파이썬)을 그대로 재사용. JS 포팅 불필요.
"""
import os
import json
import time
import threading
from datetime import datetime, timedelta

from core.session_manager import _get_conn, save_session, create_session
from core.rsi_wave import (
    analyze_rsi_wave, generate_wave_svg, generate_price_ladder_svg,
    generate_summary_text, format_rsi_wave_for_ai, RSI_WAVE_SYSTEM_PROMPT,
)
from core.ai_client import analyze_chart

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tradeai_config.json")
POLL_INTERVAL = 5     # 초
STALE_SECONDS = 180   # processing 점유 후 이 시간 지나면 죽은 워커로 보고 재점유

_worker_started = False
_worker_lock = threading.Lock()
_jobs_ready = False


def _load_config():
    """OpenAI 키/모델 — 로컬 설정파일(데스크탑) + 환경변수(클라우드) 병합.
    환경변수가 있으면 우선 적용."""
    cfg = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass
    if os.environ.get("OPENAI_API_KEY"):
        cfg["api_key"] = os.environ["OPENAI_API_KEY"]
    if os.environ.get("TRADEAI_MODEL"):
        cfg["model"] = os.environ["TRADEAI_MODEL"]
    return cfg


# ═══════════════════════════════════════════════
# 테이블 / 큐 조작
# ═══════════════════════════════════════════════

def ensure_jobs_table():
    """analysis_jobs 테이블 생성 + 모바일(anon) 접근 권한 부여 (최초 1회)"""
    global _jobs_ready
    if _jobs_ready:
        return True
    conn = _get_conn()
    if conn is None:
        return False
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS analysis_jobs (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                status TEXT DEFAULT 'pending',
                requested_at TEXT,
                started_at TEXT,
                finished_at TEXT,
                result_session_id TEXT,
                error TEXT,
                source TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # 채팅 지원 컬럼 (기존 테이블에도 추가 — 멱등)
        for coldef in ("job_type TEXT DEFAULT 'rsi_wave'", "prompt TEXT", "session_id TEXT"):
            try:
                c.execute(f"ALTER TABLE analysis_jobs ADD COLUMN IF NOT EXISTS {coldef}")
            except Exception:
                pass
        # 모바일은 anon 키로 INSERT/SELECT 해야 하므로 권한 부여
        try:
            c.execute("GRANT ALL ON analysis_jobs TO anon, authenticated")
        except Exception:
            pass
        # PostgREST 스키마 캐시 리로드 (신규 테이블이 REST API에 즉시 노출되도록)
        try:
            c.execute("NOTIFY pgrst, 'reload schema'")
        except Exception:
            pass
        conn.commit()
        _jobs_ready = True
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


def _claim_job():
    """pending 작업 1개를 원자적으로 processing 점유 → (id, symbol) 또는 None"""
    conn = _get_conn()
    if conn is None:
        return None
    try:
        c = conn.cursor()
        now = datetime.now()
        stale_iso = (now - timedelta(seconds=STALE_SECONDS)).isoformat()
        # pending 작업 OR 점유 후 멈춘(죽은 워커) stale 작업을 재점유 → self-healing
        c.execute("""
            UPDATE analysis_jobs
               SET status='processing', started_at=%s, updated_at=CURRENT_TIMESTAMP
             WHERE id = (
                   SELECT id FROM analysis_jobs
                    WHERE status='pending'
                       OR (status='processing' AND COALESCE(started_at, '') < %s)
                    ORDER BY requested_at
                    LIMIT 1 FOR UPDATE SKIP LOCKED)
            RETURNING id, symbol, COALESCE(job_type, 'rsi_wave'), prompt, session_id
        """, (now.isoformat(), stale_iso))
        row = c.fetchone()
        conn.commit()
        return row if row else None  # (id, symbol, job_type, prompt, session_id)
    except Exception:
        conn.rollback()
        return None
    finally:
        conn.close()


def _finish_job(job_id, session_id=None, error=None):
    conn = _get_conn()
    if conn is None:
        return
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE analysis_jobs
               SET status=%s, finished_at=%s, result_session_id=%s, error=%s,
                   updated_at=CURRENT_TIMESTAMP
             WHERE id=%s
        """, ('error' if error else 'done', datetime.now().isoformat(),
              session_id, error, job_id))
        conn.commit()
    except Exception:
        conn.rollback()
    finally:
        conn.close()


# ═══════════════════════════════════════════════
# 분석 실행 (데스크탑 RSI 파동 핸들러와 동일 로직)
# ═══════════════════════════════════════════════

def _build_rsi_report(symbol):
    """RSI 파동 분석 실행 → (본문 텍스트, 차트 HTML) 반환. 신규/이어붙이기 공용."""
    results = analyze_rsi_wave(symbol)

    try:
        from core.signal_logger import log_rsi_wave_signals
        log_rsi_wave_signals(symbol, results)
    except Exception:
        pass

    svg = generate_wave_svg(results)
    ladder = generate_price_ladder_svg(results)
    combined = svg.replace("</body>", ladder + "\n</body>") if ladder else svg
    summary = generate_summary_text(results)

    cfg = _load_config()
    api_key = cfg.get("api_key")
    model = cfg.get("model", "gpt-5.5")
    ai_text = ""
    if api_key:
        try:
            prompt = format_rsi_wave_for_ai(symbol, results)
            ai_text = "".join(analyze_chart(
                api_key=api_key, model=model,
                messages=[{"role": "user", "content": prompt}],
                system_prompt_override=RSI_WAVE_SYSTEM_PROMPT,
            ))
        except Exception as e:
            ai_text = f"⚠️ AI 분석 오류: {e}"

    content = summary + ("\n\n" + ai_text if ai_text else "")
    return content, combined


def run_analysis(symbol):
    """RSI 파동 분석 + (키 있으면) AI 코멘터리 → 'report' 세션 저장. 반환: session_id"""
    content, combined = _build_rsi_report(symbol)
    sess = create_session(symbol=symbol)
    sess["status"] = "report"  # 데스크탑 탭 자동복원에서 제외 (모바일 전용 리포트)
    sess["messages"] = [
        {"role": "user", "content": "🌊 RSI 파동 분석 (모바일 요청)"},
        {"role": "assistant", "content": content, "rsi_wave_html": combined},
    ]
    save_session(sess)
    return sess["id"]


def run_analysis_append(session_id, symbol):
    """기존 세션에 RSI 파동 분석 결과를 이어붙인다(새 세션 생성 안 함).
    모바일 상세화면에서 '같은 분석 안에서 RSI 재분석'을 누를 때 사용."""
    from core.session_manager import load_session
    sess = load_session(session_id) if session_id else None
    if not sess:
        # 세션을 못 찾으면 신규 분석으로 폴백
        return run_analysis(symbol or "BTCUSDT")
    sym = (symbol or sess.get("symbol") or "BTCUSDT")
    if not sess.get("symbol"):
        sess["symbol"] = sym
    content, combined = _build_rsi_report(sym)
    sess.setdefault("messages", [])
    sess["messages"].append({"role": "user", "content": "🌊 RSI 재분석"})
    sess["messages"].append({"role": "assistant", "content": content, "rsi_wave_html": combined})
    save_session(sess)
    return sess["id"]


def run_chat(session_id, prompt, symbol):
    """텍스트 AI 채팅 — 기존 세션에 사용자 메시지 추가 후 analyze_chart로 답변 생성.
    PC 채팅과 동일한 함수(analyze_chart) 사용. 차트 이미지는 없음(실시간 데이터 기반)."""
    from core.session_manager import load_session
    from core.market_data import get_multi_timeframe_context, parse_requested_timeframes

    sess = load_session(session_id) if session_id else None
    if not sess:
        sess = create_session(symbol=symbol or "BTCUSDT")
        sess["status"] = "report"
    sess.setdefault("messages", [])
    if not sess.get("symbol"):
        sess["symbol"] = symbol or "BTCUSDT"

    sym = sess.get("symbol") or "BTCUSDT"
    interval = sess.get("interval") or "15분"

    # 사용자 메시지 추가
    sess["messages"].append({"role": "user", "content": prompt})

    cfg = _load_config()
    api_key = cfg.get("api_key")
    model = cfg.get("model", "gpt-5.5")
    if not api_key:
        sess["messages"].append({
            "role": "assistant",
            "content": "⚠️ 서버에 OpenAI 키가 설정되지 않아 답변할 수 없습니다.",
        })
        save_session(sess)
        return sess["id"]

    # 실시간 시장 데이터 수집 (질문에서 타임프레임 감지)
    try:
        tfs = parse_requested_timeframes(prompt, interval)
        market_data = get_multi_timeframe_context(sym, tfs, interval)
    except Exception:
        market_data = ""

    try:
        resp = "".join(analyze_chart(
            api_key=api_key, model=model,
            messages=sess["messages"], market_data=market_data,
        ))
    except Exception as e:
        resp = f"⚠️ AI 오류: {e}"

    sess["messages"].append({"role": "assistant", "content": str(resp)})
    save_session(sess)
    return sess["id"]


def process_pending_jobs(max_jobs=3):
    """대기 중인 작업을 최대 max_jobs개 처리. 반환: 처리한 수"""
    if not ensure_jobs_table():
        return 0
    n = 0
    for _ in range(max_jobs):
        claim = _claim_job()
        if not claim:
            break
        job_id, symbol, job_type, prompt, session_id = claim
        try:
            if job_type == "chat":
                sid = run_chat(session_id, prompt or "", (symbol or "BTCUSDT").upper())
            elif job_type == "rsi_append":
                sid = run_analysis_append(session_id, (symbol or "BTCUSDT").upper())
            else:
                sid = run_analysis((symbol or "BTCUSDT").upper())
            _finish_job(job_id, session_id=sid)
        except Exception as e:
            _finish_job(job_id, error=str(e)[:500])
        n += 1
    return n


# ═══════════════════════════════════════════════
# 백그라운드 스레드 (앱에서 1회 기동)
# ═══════════════════════════════════════════════

EVAL_INTERVAL = 300   # 초 — 성숙 신호 자동 평가 주기(성적표 데이터 자동 숙성)
_last_eval = [0.0]


def _maybe_evaluate():
    """주기적으로 성숙한 RSI 신호를 평가해 성적표 데이터를 자동으로 익힌다.
    (예전엔 '신호 통계' 페이지를 열 때만 평가됐다.)"""
    now = time.time()
    if now - _last_eval[0] < EVAL_INTERVAL:
        return
    _last_eval[0] = now
    try:
        from core.signal_logger import evaluate_pending_signals
        evaluate_pending_signals(max_groups=14)
    except Exception:
        pass


def _loop():
    ensure_jobs_table()
    while True:
        try:
            process_pending_jobs()
        except Exception:
            pass
        try:
            _maybe_evaluate()
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)


def start_worker_thread():
    """싱글톤 데몬 워커 스레드 기동 (Streamlit 재실행에도 1개만 유지)"""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        t = threading.Thread(target=_loop, daemon=True, name="rsi-job-worker")
        t.start()
        _worker_started = True
