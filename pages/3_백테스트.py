"""🧪 백테스트 — 과거 캔들로 RSI 파동 신호의 기대값을 즉시 측정"""
import pandas as pd
import streamlit as st

from core.backtest import run_backtest, verdict_table, BACKTEST_TFS, WARMUP_BARS

st.set_page_config(page_title="백테스트", page_icon="🧪", layout="wide")

st.markdown("## 🧪 신호 백테스트")
st.caption(
    "과거 캔들에 실거래와 동일한 신호 로직(analyze_tf_snapshot)을 돌려 "
    "신호 유형별 기대값을 측정합니다. OI/펀딩 가점은 과거 데이터가 없어 제외됩니다."
)

# ── 입력 ──
c1, c2, c3, c4 = st.columns([2, 1.2, 1.5, 1])
with c1:
    symbol = st.text_input("종목", value="BTCUSDT").strip().upper()
with c2:
    tf = st.selectbox("타임프레임", BACKTEST_TFS, index=3)  # 기본 1시간
with c3:
    total_bars = st.slider("캔들 수", 800, 5000, 2000, step=200,
                           help=f"워밍업 {WARMUP_BARS}봉 + 평가 호라이즌 제외 후 나머지가 판정 구간")
with c4:
    dedupe = st.checkbox("상태변화만", value=True,
                         help="같은 (방향, 신호유형)이 연속되면 첫 봉만 기록")

if st.button("▶️ 백테스트 실행", type="primary", use_container_width=True):
    prog = st.progress(0.0, text="과거 캔들 수집 + 신호 재생 중...")
    try:
        bt = run_backtest(symbol, tf, total_bars=total_bars, dedupe=dedupe,
                          progress_cb=lambda p: prog.progress(min(p, 1.0)))
        st.session_state["_bt_result"] = bt
    except Exception as e:
        st.error(f"백테스트 실패: {e}")
    finally:
        prog.empty()

bt = st.session_state.get("_bt_result")
if not bt:
    st.info("종목·타임프레임을 정하고 실행을 눌러주세요. 1시간봉 2,000개 기준 약 10초 걸립니다.")
    st.stop()

# ── 요약 메트릭 ──
ov = bt["overall"]
st.markdown(f"### {bt['symbol']} · {bt['timeframe']} · {bt['period'][0]} ~ {bt['period'][1]}")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("평가 신호", f"{bt['n_signals']}건")
m2.metric("적중률 (gross)", f"{ov['win_rate'] if ov['win_rate'] is not None else '-'}%")
m3.metric("적중률 (수수료 차감)", f"{ov['win_rate_net'] if ov['win_rate_net'] is not None else '-'}%")
m4.metric("평균수익 (net)", f"{ov['avg_return_net'] if ov['avg_return_net'] is not None else '-'}%")
m5.metric("MFE / MAE", f"{ov['avg_mfe']}% / {ov['avg_mae']}%")
st.caption(f"net = 왕복 비용 {bt['fee_pct']}% 차감 · 호라이즌 {bt['horizon_bars']}봉 보유 가정 · "
           f"신호 시점 종가 즉시 진입 가정")

# ── 버려/살려 판정 ──
st.markdown("### 🎯 신호유형별 판정")
vt = verdict_table(bt["by_signal_type"])
if vt:
    df_v = pd.DataFrame(vt)
    df_v.columns = ["신호유형", "판정", "n", "적중률%", "net적중률%", "평균수익%", "net평균%", "MFE%", "MAE%"]
    st.dataframe(df_v, use_container_width=True, hide_index=True)
else:
    st.info("평가된 신호가 없습니다.")

# ── 누적 수익 곡선 ──
sig = bt["signals"]
if sig:
    df = pd.DataFrame(sig)
    df["time"] = pd.to_datetime(df["time"])
    st.markdown("### 📈 누적 수익 곡선 (신호별 실현수익 합산, net)")
    df["cum_net"] = (df["return_pct"] - bt["fee_pct"]).cumsum()
    st.line_chart(df.set_index("time")["cum_net"], height=260)

    # ── 그룹별 표 ──
    st.markdown("### 📊 세부 분해")
    g1, g2 = st.columns(2)

    def _group_df(d):
        rows = [{"항목": k, "n": v["n"], "적중률%": v["win_rate"],
                 "net평균%": v["avg_return_net"], "MFE%": v["avg_mfe"], "MAE%": v["avg_mae"]}
                for k, v in d.items() if v.get("n")]
        return pd.DataFrame(rows).sort_values("n", ascending=False)

    with g1:
        st.markdown("**레짐별**")
        st.dataframe(_group_df(bt["by_regime"]), use_container_width=True, hide_index=True)
        st.markdown("**방향별**")
        st.dataframe(_group_df(bt["by_position"]), use_container_width=True, hide_index=True)
    with g2:
        st.markdown("**확신등급별**")
        st.dataframe(_group_df(bt["by_confidence"]), use_container_width=True, hide_index=True)
        st.markdown("**수익 분포**")
        st.bar_chart(df["return_pct"].round(0).value_counts().sort_index(), height=200)

    with st.expander(f"🧾 신호 전체 목록 ({len(sig)}건)", expanded=False):
        df_all = df[["time", "price", "position", "confidence", "signal_type",
                     "regime", "rsi", "return_pct", "mfe_pct", "mae_pct", "outcome"]]
        st.dataframe(df_all, use_container_width=True, hide_index=True, height=400)

st.caption("⚠️ 백테스트는 신호 시점 종가 진입·호라이즌 종료 청산 가정의 단순화 모델입니다. "
           "슬리피지·체결 실패·자금조달료는 반영되지 않으니 기대값의 '부호와 순위' 비교 용도로 쓰세요.")
