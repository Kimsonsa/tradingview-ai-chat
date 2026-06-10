"""
로컬 설정 파일(.tradeai_config.json) 공용 액세스 — 메인 앱 + 멀티페이지 공유.

저장 키:
  api_key         OpenAI API 키
  claude_api_key  Anthropic API 키
  model           선택된 AI 모델
  account_size    계좌 크기 (USDT) — 리스크 계산기
  risk_pct        거래당 리스크 % — 리스크 계산기
  watchlist       워치리스트 심볼 리스트
"""
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tradeai_config.json")

DEFAULT_WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(data):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def update_config(**kv):
    """기존 설정에 병합 저장 — 다른 키를 덮어쓰지 않는다"""
    cfg = load_config()
    cfg.update(kv)
    save_config(cfg)
    return cfg
