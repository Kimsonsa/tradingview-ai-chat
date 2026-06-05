"""
Supabase 연결 설정 — 환경변수(클라우드) 우선, Streamlit secrets(데스크탑) 폴백.

데스크탑: .streamlit/secrets.toml 의 [supabase] 사용
클라우드(Render 등): SUPABASE_HOST/PORT/DB/USER/PASSWORD 환경변수 사용
→ 같은 코어 코드를 두 환경에서 그대로 재사용.
"""
import os


def get_supabase_config():
    """Supabase 접속 설정 dict 반환 (없으면 None)"""
    # 1) 환경변수 (클라우드 배포)
    if os.environ.get("SUPABASE_HOST"):
        return {
            "host": os.environ["SUPABASE_HOST"],
            "port": os.environ.get("SUPABASE_PORT", "5432"),
            "dbname": os.environ.get("SUPABASE_DB", "postgres"),
            "user": os.environ["SUPABASE_USER"],
            "password": os.environ["SUPABASE_PASSWORD"],
        }
    # 2) Streamlit secrets (데스크탑) — streamlit 미설치 환경에서도 안전하게
    try:
        import streamlit as st
        cfg = st.secrets["supabase"]
        return {
            "host": cfg["host"], "port": cfg["port"], "dbname": cfg["dbname"],
            "user": cfg["user"], "password": cfg["password"],
        }
    except Exception:
        return None


def get_conn():
    """psycopg2 커넥션 반환 (실패 시 None)"""
    cfg = get_supabase_config()
    if not cfg:
        return None
    try:
        import psycopg2
        return psycopg2.connect(connect_timeout=5, **cfg)
    except Exception:
        return None
