"""
트레이딩 세션 관리 — Supabase(PostgreSQL) 클라우드 동기화
어느 PC에서든 동일한 채팅 기록을 공유합니다.
로컬 JSON은 폴백/캐시용으로 유지합니다.
"""
import json
import os
import uuid
from datetime import datetime

import psycopg2
import psycopg2.extras

from core.db_config import get_conn as _db_get_conn

# 로컬 폴백 디렉토리
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")


# ═══════════════════════════════════════════════
# DB 연결
# ═══════════════════════════════════════════════

def _get_conn():
    """Supabase PostgreSQL 연결 (환경변수 우선, secrets.toml 폴백)"""
    return _db_get_conn()


def _init_table():
    """trade_sessions 테이블 생성 (최초 1회)"""
    conn = _get_conn()
    if conn is None:
        return False
    try:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_sessions (
                id TEXT PRIMARY KEY,
                symbol TEXT DEFAULT '',
                interval TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                created_at TEXT,
                closed_at TEXT,
                messages JSONB DEFAULT '[]'::jsonb,
                summary JSONB,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        return False
    finally:
        conn.close()


# 앱 시작 시 테이블 초기화
_DB_READY = False


def _ensure_db():
    global _DB_READY
    if not _DB_READY:
        _DB_READY = _init_table()
    return _DB_READY


# ═══════════════════════════════════════════════
# 로컬 유틸 (폴백)
# ═══════════════════════════════════════════════

def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _make_serializable(obj):
    """재귀적으로 모든 값을 JSON 직렬화 가능한 타입으로 변환"""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    # 그 외 (PIL Image, bytes 등) → 제거
    return None


def _clean_for_save(session):
    """저장용으로 직렬화 불가능한 객체 제거"""
    clean = dict(session)
    clean.pop("last_capture", None)
    clean.pop("last_capture_b64", None)
    clean_msgs = []
    for msg in clean.get("messages", []):
        m = dict(msg)
        m.pop("image", None)
        clean_msgs.append(m)
    clean["messages"] = clean_msgs
    return _make_serializable(clean)


# ═══════════════════════════════════════════════
# 공용 API
# ═══════════════════════════════════════════════

def create_session(symbol="", interval=""):
    """새 세션 생성 → session dict 반환 (아직 저장 안 됨)"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    return {
        "id": session_id,
        "symbol": symbol,
        "interval": interval,
        "created_at": datetime.now().isoformat(),
        "closed_at": None,
        "status": "active",
        "messages": [],
        "summary": None,
    }


def save_session(session):
    """세션 저장 — Supabase 우선, 로컬 폴백"""
    clean = _clean_for_save(session)

    # 1) Supabase 저장
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO trade_sessions (id, symbol, interval, status, created_at, closed_at, messages, summary, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (id) DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        interval = EXCLUDED.interval,
                        status = EXCLUDED.status,
                        closed_at = EXCLUDED.closed_at,
                        messages = EXCLUDED.messages,
                        summary = EXCLUDED.summary,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    clean["id"],
                    clean.get("symbol", ""),
                    clean.get("interval", ""),
                    clean.get("status", "active"),
                    clean.get("created_at", ""),
                    clean.get("closed_at"),
                    json.dumps(clean.get("messages", []), ensure_ascii=False),
                    json.dumps(clean.get("summary"), ensure_ascii=False) if clean.get("summary") else None,
                ))
                conn.commit()
            except Exception:
                conn.rollback()
            finally:
                conn.close()

    # 2) 로컬 폴백 저장
    _ensure_dir()
    path = os.path.join(SESSIONS_DIR, f"{clean['id']}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_session(session_id):
    """세션 로드 — Supabase 우선, 없으면 로컬"""
    # 1) Supabase
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                c.execute("SELECT * FROM trade_sessions WHERE id = %s", (session_id,))
                row = c.fetchone()
                if row:
                    data = dict(row)
                    # JSONB → Python
                    if isinstance(data.get("messages"), str):
                        data["messages"] = json.loads(data["messages"])
                    if isinstance(data.get("summary"), str):
                        data["summary"] = json.loads(data["summary"])
                    # updated_at은 반환에서 제외
                    data.pop("updated_at", None)
                    return data
            except Exception:
                pass
            finally:
                conn.close()

    # 2) 로컬 폴백
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def delete_session(session_id):
    """세션 삭제 — Supabase + 로컬 모두"""
    # 1) Supabase
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor()
                c.execute("DELETE FROM trade_sessions WHERE id = %s", (session_id,))
                conn.commit()
            except Exception:
                conn.rollback()
            finally:
                conn.close()

    # 2) 로컬
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def list_sessions():
    """전체 세션 목록 (최신순) — Supabase 우선, 로컬 폴백
    반환: [{ id, symbol, interval, created_at, closed_at, status, summary, msg_count }, ...]
    """
    # 1) Supabase
    if _ensure_db():
        conn = _get_conn()
        if conn:
            try:
                c = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                c.execute("""
                    SELECT id, symbol, interval, status, created_at, closed_at, summary,
                           jsonb_array_length(COALESCE(messages, '[]'::jsonb)) as msg_count
                    FROM trade_sessions
                    ORDER BY created_at DESC
                """)
                rows = c.fetchall()
                sessions = []
                for row in rows:
                    d = dict(row)
                    if isinstance(d.get("summary"), str):
                        d["summary"] = json.loads(d["summary"])
                    sessions.append(d)
                return sessions
            except Exception:
                pass
            finally:
                conn.close()

    # 2) 로컬 폴백
    _ensure_dir()
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            path = os.path.join(SESSIONS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data["id"],
                "symbol": data.get("symbol", ""),
                "interval": data.get("interval", ""),
                "created_at": data.get("created_at", ""),
                "closed_at": data.get("closed_at"),
                "status": data.get("status", "active"),
                "summary": data.get("summary"),
                "msg_count": len(data.get("messages", [])),
            })
        except Exception:
            continue

    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions
