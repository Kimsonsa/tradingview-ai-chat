"""📊 워치리스트 — 관심 심볼들의 레짐/신호/점수를 한 표로"""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st

from core.app_config import load_config, update_config, DEFAULT_WATCHLIST
from core.market_data import fetch_klines, INTERVAL_MAP
from core.rsi_wave import (
    analyze_tf_snapshot, flow_gate, REGIME_LABELS, DIR_TF_WEIGHT, CONF_WEIGHT,
)

st.set_page_config(page_title="워치리스트", page_icon="📊", layout="wide")

DASH_TFS = ["15분", "1시간", "4시간", "1일"]

st.markdown("## 📊 워치리스트")
st.caption("관심 심볼들의 타임프레임별 레짐·포지션 판정을 한눈에. 1분마다 자동 갱신(캐시).")

# ── 워치리스트 편집 ──
cfg = load_config()
watchlist = cfg.get("watchlist") or DEFAULT_WATCHLIST
wc1, wc2 = st.columns([5, 1])
with wc1:
    wl_text = st.text_input("심볼 (쉼표 구분)", value=", ".join(watchlist),
                            label_visibility="collapsed")
with wc2:
    refresh = st.button("🔄 새로고침", use_container_width=True)

new_list = [s.strip().upper() for s in wl_text.split(",") if s.strip()][:12]
if new_list and new_list != watchlist:
    watchlist = new_list
    update_config(watchlist=watchlist)


@st.cache_data(ttl=60, show_spinner=False)
def _scan(symbols, tfs):
    """심볼×TF 전체를 병렬 수집·판정"""
    def one(sym, tf):
        candles = fetch_klines(sym, INTERVAL_MAP[tf], 500)
        return analyze_tf_snapshot(tf, candles)

    out = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {(s, t): ex.submit(one, s, t) for s in symbols for t in tfs}
        for key, fut in futs.items():
            try:
                out[key] = fut.result()
            except Exception as e:
                out[key] = {"error": str(e)[:80]}
    return out


if refresh:
    _scan.clear()

with st.spinner(f"{len(watchlist)}개 심볼 × {len(DASH_TFS)}개 TF 스캔 중..."):
    data = _scan(tuple(watchlist), tuple(DASH_TFS))


def _cell(r):
    if not r or r.get("error"):
        return "⚠️"
    regime = REGIME_LABELS.get(r.get("regime", ""), r.get("regime", ""))
    pos = r.get("position", "")
    conf = r.get("confidence", "")
    icon = "🟢" if pos == "롱" else "🔴" if pos == "숏" else "⚪"
    gate = " 🚦" if flow_gate(r) == "AGREE" else ""  # 주문흐름 동의(백테스트 검증 게이트)
    return f"{icon} {pos}·{conf}{gate} | {regime}"


def _bias_score(sym):
    """TF 가중 방향 점수 — 양수=롱 우위, 음수=숏 우위"""
    score = 0.0
    for tf in DASH_TFS:
        r = data.get((sym, tf))
        if not r or r.get("error"):
            continue
        w = DIR_TF_WEIGHT.get(tf, 1.0) * CONF_WEIGHT.get(r.get("confidence", ""), 0.3)
        if r.get("position") == "롱":
            score += w
        elif r.get("position") == "숏":
            score -= w
    return round(score, 2)


rows = []
for sym in watchlist:
    r15 = data.get((sym, "15분")) or {}
    price = r15.get("price")
    bias = _bias_score(sym)
    bias_label = ("🟢 롱 우위" if bias >= 1.5 else "🔴 숏 우위" if bias <= -1.5
                  else "↗ 롱 기울" if bias > 0.3 else "↘ 숏 기울" if bias < -0.3 else "↔ 중립")
    row = {
        "심볼": sym,
        "현재가": f"{price:,.4g}" if price else "-",
        "종합": f"{bias_label} ({bias:+.1f})",
    }
    for tf in DASH_TFS:
        row[tf] = _cell(data.get((sym, tf)))
    row["_bias_abs"] = abs(bias)
    rows.append(row)

df = pd.DataFrame(rows).sort_values("_bias_abs", ascending=False).drop(columns="_bias_abs")
st.dataframe(df, use_container_width=True, hide_index=True,
             height=42 + 36 * len(df))

st.caption("종합 = TF별 (방향 × TF가중치 × 확신가중치) 합산. "
           "🟢/🔴 셀 = 해당 TF의 포지션 판정·확신등급 | 레짐 · "
           "🚦 = 주문흐름(OBV/CVD) 동의 — 백테스트 검증 고승률 조건(15분 기준 ~73%)")

# ── 심볼 바로 분석 열기 ──
st.markdown("#### 💬 채팅으로 분석 열기")
cols = st.columns(min(len(watchlist), 6) or 1)
for i, sym in enumerate(watchlist):
    with cols[i % len(cols)]:
        if st.button(sym, key=f"open_{sym}", use_container_width=True):
            st.session_state["_open_symbol"] = sym
            st.switch_page("assistant.py")
