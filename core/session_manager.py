"""
트레이딩 세션 관리 — 대화 기록 저장/로드/목록/삭제
저장 경로: {프로젝트}/sessions/{session_id}.json
"""
import json
import os
import uuid
from datetime import datetime

# 세션 저장 디렉토리
SESSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sessions")


def _ensure_dir():
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def create_session(symbol="", interval=""):
    """새 세션 생성 → session dict 반환 (아직 저장 안 됨)"""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    return {
        "id": session_id,
        "symbol": symbol,
        "interval": interval,
        "created_at": datetime.now().isoformat(),
        "closed_at": None,
        "status": "active",       # active / closed
        "messages": [],
        "summary": None,          # AI 분석 결과 (종료 시 채워짐)
    }


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


def save_session(session):
    """세션을 JSON 파일로 저장"""
    _ensure_dir()
    path = os.path.join(SESSIONS_DIR, f"{session['id']}.json")
    clean = _clean_for_save(session)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def _clean_for_save(session):
    """저장용으로 직렬화 불가능한 객체 제거"""
    clean = dict(session)
    # 저장 불필요한 대용량/비직렬화 키 제거
    clean.pop("last_capture", None)      # PIL Image
    clean.pop("last_capture_b64", None)  # base64 문자열 (용량 큼)
    # 메시지에서 image 키 제거
    clean_msgs = []
    for msg in clean.get("messages", []):
        m = dict(msg)
        m.pop("image", None)       # PIL Image 제거
        clean_msgs.append(m)
    clean["messages"] = clean_msgs
    # 최종 안전장치: 재귀적으로 직렬화 불가 객체 제거
    return _make_serializable(clean)


def load_session(session_id):
    """저장된 세션 로드"""
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def delete_session(session_id):
    """세션 삭제"""
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        os.remove(path)


def list_sessions():
    """저장된 전체 세션 목록 반환 (최신순)
    반환: [{ id, symbol, interval, created_at, closed_at, status, summary }, ...]
    """
    _ensure_dir()
    sessions = []
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            path = os.path.join(SESSIONS_DIR, fname)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 목록용 요약만 반환 (messages 제외)
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

    # 최신순 정렬
    sessions.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return sessions
