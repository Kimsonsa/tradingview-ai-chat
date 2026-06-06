"""
클라우드 분석 워커 (Render 등 배포용)

모바일이 Supabase analysis_jobs 에 작업을 INSERT한 뒤 이 서비스의 /process 를
호출하면, 대기 작업을 처리(analyze_rsi_wave + AI)하여 'report' 세션으로 저장한다.
PC 없이도 폰만으로 분석이 가능해진다.

필요한 환경변수:
  SUPABASE_HOST, SUPABASE_PORT, SUPABASE_DB, SUPABASE_USER, SUPABASE_PASSWORD
  OPENAI_API_KEY        (없으면 결정론적 리포트만 생성)
  TRADEAI_MODEL         (선택, 기본 gpt-5.5)
  WORKER_TOKEN          (선택, 설정 시 /process 호출에 X-Worker-Token 헤더 요구)

로컬 실행:  uvicorn cloud_worker:app --host 0.0.0.0 --port 8000
"""
import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.job_worker import ensure_jobs_table, process_pending_jobs, start_worker_thread

app = FastAPI(title="TradeAI Cloud Worker")


@app.on_event("startup")
def _boot_worker():
    """인스턴스 기동(콜드스타트 포함) 시 백그라운드 폴링 워커를 띄운다.

    무료 플랜은 미사용 시 잠들고, 깨우는 /process 요청은 콜드스타트
    로딩 페이지에 먹혀 핸들러까지 도달하지 못할 수 있다. 그 경우에도
    인스턴스가 일단 깨어나면 이 데몬 스레드가 pending 작업을 폴링해
    처리하므로 폰의 단발성 트리거 유실과 무관하게 분석이 완료된다.
    (/process 는 '즉시 처리' 가속기로 남는다.)
    """
    try:
        start_worker_thread()
    except Exception:
        pass

# 모바일(브라우저)에서 직접 호출하므로 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")


@app.get("/")
def health():
    return {"status": "ok", "service": "tradeai-cloud-worker"}


@app.get("/diag")
def diag():
    """진단: 이 서버에서 Binance 선물 API 접근 가능한지 확인 (지역차단 451 여부)."""
    import requests
    out = {}
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": "BTCUSDT", "interval": "1h", "limit": 2},
            timeout=10,
        )
        out["binance_status"] = r.status_code
        out["binance_body"] = r.text[:200]
        out["accessible"] = (r.status_code == 200)
    except Exception as e:
        out["binance_error"] = str(e)[:300]
        out["accessible"] = False
    return out


def _check_token(token):
    if _WORKER_TOKEN and token != _WORKER_TOKEN:
        raise HTTPException(status_code=401, detail="invalid worker token")


@app.post("/process")
@app.get("/process")
def process(x_worker_token: str = Header(default="")):
    """대기 중인 분석 작업을 처리. 모바일이 작업 INSERT 후 호출."""
    _check_token(x_worker_token)
    ensure_jobs_table()
    try:
        n = process_pending_jobs(max_jobs=5)
        return {"ok": True, "processed": n}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}
