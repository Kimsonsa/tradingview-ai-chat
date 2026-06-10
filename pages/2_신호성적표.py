"""📈 신호 성적표 — 실시간 기록된 RSI 파동 신호의 실측 적중률"""
import pandas as pd
import streamlit as st

from core.signal_logger import (
    evaluate_pending_signals, get_signal_stats, get_weight_suggestions,
    get_attribution_stats, get_timing_stats, _all_evaluated, FEE_PCT,
)

st.set_page_config(page_title="신호 성적표", page_icon="📈", layout="wide")

_MIN_N = 20  # 표본부족 경고 기준

st.markdown("## 📈 신호 성적표")
st.caption("실거래 중 기록된 RSI 파동 신호를 호라이즌 경과 후 실제 가격으로 평가한 결과입니다. "
           "(과거 데이터 검증은 🧪 백테스트 페이지)")

c1, c2 = st.columns([1, 4])
with c1:
    if st.button("🔄 평가 갱신", type="primary", use_container_width=True):
        with st.spinner("성숙한 신호 평가 중..."):
            n = evaluate_pending_signals()
        st.toast(f"신규 평가 {n}건")
        st.cache_data.clear()
with c2:
    symbol_filter = st.text_input("심볼 필터 (비우면 전체)", value="",
                                  label_visibility="collapsed",
                                  placeholder="심볼 필터 — 예: BTCUSDT (비우면 전체)")

sym = symbol_filter.strip().upper() or None


@st.cache_data(ttl=120, show_spinner=False)
def _load(sym):
    return (get_signal_stats(sym), get_attribution_stats(sym),
            get_timing_stats(sym, max_groups=20), get_weight_suggestions(sym),
            _all_evaluated(sym))


with st.spinner("통계 로딩 중..."):
    try:
        stats, attr, timing, sug, raw_rows = _load(sym)
    except Exception as e:
        st.error(f"통계 오류: {e}")
        st.stop()

ov = stats.get("overall", {})
if not ov.get("n"):
    st.info("아직 평가된 신호가 없습니다. 신호는 RSI 파동 분석 시 자동 기록되고, "
            "호라이즌(예: 1시간봉=24h)이 지나면 평가됩니다.")
    st.stop()

# ── 핵심 메트릭 ──
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("평가 완료", f"{stats['total_evaluated']}건")
m2.metric("적중률", f"{ov.get('win_rate', '-')}%")
m3.metric("수수료 차감 적중률", f"{ov.get('win_rate_net', '-')}%")
m4.metric("평균 실현수익 (net)", f"{ov.get('avg_return_net', '-')}%")
m5.metric("MFE / MAE", f"{ov.get('avg_mfe')}% / {ov.get('avg_mae')}%")
st.caption(f"net = 왕복 비용 {FEE_PCT}% 차감 가정")

# ── 차트: 누적 적중률 추이 + 신호 분포 ──
df = pd.DataFrame([r for r in raw_rows if r.get("outcome") in ("WIN", "LOSS")])
if not df.empty:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).sort_values("created_at")
    df["win"] = (df["outcome"] == "WIN").astype(int)
    df["누적 적중률 %"] = (df["win"].cumsum() / range(1, len(df) + 1)) * 100

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("**누적 적중률 추이**")
        st.line_chart(df.set_index("created_at")["누적 적중률 %"], height=240)
    with ch2:
        st.markdown("**신호유형별 표본 분포**")
        st.bar_chart(df["signal_type"].value_counts(), height=240, horizontal=True)


# ── 그룹별 테이블 ──
def _group_df(d):
    rows = [{"항목": k, "n": f"{v['n']} ⚠️" if v["n"] < _MIN_N else str(v["n"]),
             "적중률%": v.get("win_rate"), "net적중률%": v.get("win_rate_net"),
             "net평균%": v.get("avg_return_net"),
             "MFE%": v.get("avg_mfe"), "MAE%": v.get("avg_mae")}
            for k, v in d.items() if v.get("n")]
    return pd.DataFrame(rows)


st.markdown("### 📊 그룹별 성적")
tabs = st.tabs(["롱/숏 방향", "신호유형", "확신등급", "레짐", "타임프레임", "종목"])
for tab, key in zip(tabs, ["by_position", "by_signal_type", "by_confidence",
                           "by_regime", "by_timeframe", "by_symbol"]):
    with tab:
        g = _group_df(stats.get(key, {}))
        if g.empty:
            st.caption("데이터 없음")
        else:
            st.dataframe(g, use_container_width=True, hide_index=True)
st.caption(f"⚠️ = 표본 {_MIN_N}건 미만 (우연일 수 있어 신뢰 낮음)")

# ── B: 추천 진입 타이밍 ──
if timing.get("n_analyses"):
    st.markdown("### 🎯 추천 진입 타이밍 (시나리오 시뮬레이션)")
    t1, t2, t3 = st.columns(3)
    t1.metric("분석 건수", timing["n_analyses"])
    t2.metric("시나리오 적중률", f"{timing.get('win_rate', '-')}%")
    t3.metric("승 / 패 / 미트리거",
              f"{timing.get('win', 0)} / {timing.get('loss', 0)} / {timing.get('not_triggered', 0)}")
    bg = timing.get("by_grade") or {}
    if bg:
        st.dataframe(pd.DataFrame(
            [{"등급": g, "적중률%": v.get("win_rate"), "승": v.get("win"), "패": v.get("loss")}
             for g, v in bg.items()]), hide_index=True)

# ── C: 지표별 귀인 ──
if attr.get("n"):
    st.markdown(f"### 🔍 지표별 귀인 — 어느 지표 해석이 맞았나 (n={attr['n']})")
    st.caption("값별 조건부 적중률. 낮을수록 그 지표 해석이 자주 틀렸다는 뜻 → 가중치 하향 후보.")
    _LBL = {"cvd_bias": "CVD 편향", "oi_quadrant": "OI 사분면",
            "div_v2": "RSI 다이버전스", "cvd_div": "CVD 다이버전스",
            "obv_div": "OBV 다이버전스", "squeeze": "스퀴즈",
            "failed_div": "다이버전스 실패", "regime": "레짐"}
    acols = st.columns(2)
    for i, (key, kv) in enumerate((attr.get("by_indicator") or {}).items()):
        rows = [{"값": v, "적중률%": d.get("win_rate"),
                 "n": f"{d['n']} ⚠️" if d["n"] < _MIN_N else str(d["n"])}
                for v, d in sorted(kv.items(), key=lambda x: -(x[1].get("n") or 0)) if d.get("n")]
        if not rows:
            continue
        with acols[i % 2]:
            st.markdown(f"**{_LBL.get(key, key)}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── 가중치 조정 제안 ──
st.markdown("### 💡 가중치 조정 제안")
if not sug.get("ready"):
    st.caption(f"표본 수집 중 — {sug.get('total', 0)}/{sug.get('min_samples', 20)}건. "
               f"{sug.get('needed', 0)}건 더 모이면 제안이 시작됩니다.")
elif not sug.get("suggestions"):
    st.caption("✅ 모든 신호가 정상 범위 — 조정 제안 없음.")
else:
    icon_map = {"high": "🔴", "medium": "🟠", "info": "🟢"}
    for s in sug["suggestions"]:
        arrow = "⬇️ 하향" if s["direction"] == "DOWN" else "⬆️ 상향"
        st.markdown(f"{icon_map.get(s['severity'], '•')} **{s['target']}** {arrow} — {s['message']}")
    st.caption("제안은 참고용입니다. 적용은 직접 판단하세요.")
