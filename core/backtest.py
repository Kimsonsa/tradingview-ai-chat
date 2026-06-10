"""
백테스트 엔진 — 과거 캔들에 RSI 파동 신호 로직을 그대로 돌려 기대값 측정

성적표(signal_logger)가 실시간으로 표본이 쌓이길 기다리는 것과 달리,
과거 수천 봉을 한 번에 재생해 "이 신호 유형은 버려/살려"를 즉시 판정한다.

파이프라인:
  1. fetch_history — endTime 페이지네이션으로 과거 캔들 수집
  2. 각 시점 i 에서 직전 500봉으로 analyze_tf_snapshot (실거래와 동일 로직)
  3. 호라이즌(EVAL_HORIZON_MIN) 뒤 캔들로 MFE/MAE/실현수익 평가 (signal_logger 와 동일 기준)
  4. 신호유형/레짐/방향/확신등급별 집계 + 수수료 차감 기대값

주의: OI/펀딩 히스토리는 정렬 불가(바이낸스 30일 제한)라 백테스트에선 제외 —
실시간 판정과 이 부분만 다르다(OI/펀딩 가점이 빠진 보수적 신호).
"""
from datetime import datetime

from core.market_data import fetch_klines, INTERVAL_MAP, KLINE_WARMUP
from core.rsi_wave import analyze_tf_snapshot
from core.signal_logger import (
    _evaluate_window, _agg, _group_agg, EVAL_HORIZON_MIN, TF_MINUTES, FEE_PCT,
)

# 지표 워밍업 — EMA200 수렴을 위해 판정 시작 전 확보할 최소 봉 수
WARMUP_BARS = 300
# 판정에 사용할 룩백 윈도 (실시간과 동일하게 KLINE_WARMUP=500봉)
SNAPSHOT_WINDOW = KLINE_WARMUP

BACKTEST_TFS = ["5분", "15분", "30분", "1시간", "4시간", "1일"]


def fetch_history(symbol, interval, total=3000):
    """endTime 페이지네이션으로 과거 캔들 total개 수집 (오래된→최신 정렬)"""
    out = []
    end_ms = None
    while len(out) < total:
        batch = fetch_klines(symbol, interval,
                             limit=min(1500, total - len(out)),
                             end_time_ms=end_ms)
        if not batch:
            break
        out = batch + out
        end_ms = batch[0]["time"] - 1
        if len(batch) < 100:  # 더 이상 과거 데이터 없음
            break
    return out[-total:]


def run_backtest(symbol, tf_label, total_bars=2000, dedupe=True, progress_cb=None):
    """심볼+TF 하나에 대한 신호 백테스트.

    Args:
        symbol: 예 "BTCUSDT"
        tf_label: 한글 TF 라벨 (예 "1시간")
        total_bars: 수집할 캔들 수 (워밍업+호라이즌 포함)
        dedupe: 같은 (방향, 신호유형) 상태가 연속되면 첫 봉만 기록
        progress_cb: 진행률 콜백 fn(0.0~1.0)

    Returns:
        dict: {symbol, timeframe, n_bars, n_signals, period,
               overall, by_signal_type, by_regime, by_position, by_confidence,
               signals: [행...], fee_pct}
    """
    bi = INTERVAL_MAP.get(tf_label)
    if not bi:
        raise ValueError(f"지원하지 않는 타임프레임: {tf_label}")

    candles = fetch_history(symbol, bi, total_bars)
    if len(candles) < WARMUP_BARS + 50:
        raise ValueError(f"캔들 부족: {len(candles)}개 (최소 {WARMUP_BARS + 50}개 필요)")

    tf_min = TF_MINUTES.get(tf_label, 60)
    horizon_bars = max(1, int(EVAL_HORIZON_MIN.get(tf_label, 1440) / tf_min))

    rows = []
    prev_key = None
    start_i = WARMUP_BARS
    end_i = len(candles) - horizon_bars
    n_steps = max(1, end_i - start_i)

    for n, i in enumerate(range(start_i, end_i)):
        if progress_cb and n % 25 == 0:
            progress_cb(n / n_steps)

        window = candles[max(0, i - SNAPSHOT_WINDOW + 1):i + 1]
        try:
            r = analyze_tf_snapshot(tf_label, window)
        except Exception:
            continue

        position = r.get("position")
        key = (position, r.get("signal_type"))
        is_new_state = key != prev_key
        prev_key = key

        if dedupe and not is_new_state:
            continue
        if position not in ("롱", "숏"):
            continue  # 중립/관망은 평가 제외

        entry = candles[i]["close"]
        fwd = candles[i + 1:i + 1 + horizon_bars]
        res = _evaluate_window(position, entry, fwd)
        if res is None:
            continue
        mfe, mae, ret, outcome = res

        rows.append({
            "time": datetime.fromtimestamp(candles[i]["time"] / 1000).isoformat(),
            "price": round(entry, 4),
            "position": position,
            "confidence": r.get("confidence", ""),
            "signal_type": r.get("signal_type", ""),
            "regime": r.get("regime", ""),
            "rsi": r.get("rsi"),
            "long_score": r.get("long_score", 0),
            "short_score": r.get("short_score", 0),
            "mfe_pct": mfe,
            "mae_pct": mae,
            "return_pct": ret,
            "outcome": outcome,
            # 컨플루언스 게이트/시나리오 시뮬용 추가 필드
            "atr": r.get("atr"),
            "obv_div": (r.get("obv_div") or {}).get("type"),
            "cvd_div": (r.get("cvd_div") or {}).get("type"),
            "div_v2": (r.get("div_v2") or {}).get("type"),
            "squeeze": (r.get("squeeze_expansion") or {}).get("type"),
            "_i": i,  # candles 내 신호 봉 인덱스
        })

    if progress_cb:
        progress_cb(1.0)

    period = (
        datetime.fromtimestamp(candles[0]["time"] / 1000).strftime("%Y-%m-%d"),
        datetime.fromtimestamp(candles[-1]["time"] / 1000).strftime("%Y-%m-%d"),
    )

    return {
        "symbol": symbol,
        "timeframe": tf_label,
        "n_bars": len(candles),
        "n_signals": len(rows),
        "horizon_bars": horizon_bars,
        "period": period,
        "fee_pct": FEE_PCT,
        "overall": _agg(rows),
        "by_signal_type": _group_agg(rows, "signal_type"),
        "by_regime": _group_agg(rows, "regime"),
        "by_position": _group_agg(rows, "position"),
        "by_confidence": _group_agg(rows, "confidence"),
        "signals": rows,
        "candles": candles,  # 시나리오 청산 시뮬용
    }


# ═══════════════════════════════════════════════
# 컨플루언스 게이트 — 측정된 고승률 조각(주문흐름 동의)만 통과
# ═══════════════════════════════════════════════

def _flow_agree_one(r):
    """OBV 또는 CVD 다이버전스가 신호 방향에 동의"""
    if r["position"] == "롱":
        return r.get("obv_div") == "OBV_BULL_DIV" or r.get("cvd_div") == "CVD_BULL_DIV"
    if r["position"] == "숏":
        return r.get("obv_div") == "OBV_BEAR_DIV" or r.get("cvd_div") == "CVD_BEAR_DIV"
    return False


def _flow_agree_both(r):
    """OBV와 CVD 다이버전스가 모두 신호 방향에 동의"""
    if r["position"] == "롱":
        return r.get("obv_div") == "OBV_BULL_DIV" and r.get("cvd_div") == "CVD_BULL_DIV"
    if r["position"] == "숏":
        return r.get("obv_div") == "OBV_BEAR_DIV" and r.get("cvd_div") == "CVD_BEAR_DIV"
    return False


def _regime_ok(r):
    """신호 방향이 레짐과 정면충돌하지 않음"""
    if r["position"] == "롱":
        return r.get("regime") not in ("DOWN_TREND", "DOWN_BIAS")
    if r["position"] == "숏":
        return r.get("regime") not in ("UP_TREND", "UP_BIAS")
    return True


GATES = {
    "없음": lambda r: True,
    "주문흐름 동의(OBV/CVD 중 1)": _flow_agree_one,
    "주문흐름 강동의(둘 다)": _flow_agree_both,
    "주문흐름 동의 + 레짐 비충돌": lambda r: _flow_agree_one(r) and _regime_ok(r),
}


# ═══════════════════════════════════════════════
# 시나리오 청산 시뮬 — 손절/목표 중 먼저 닿는 쪽 (호라이즌 보유와 대비)
# ═══════════════════════════════════════════════

def simulate_scenario_exits(candles, rows, stop_atr=1.0, target_ratio=1.0,
                            max_bars=96, fee_pct=FEE_PCT):
    """각 신호에 손절·목표 기반 청산을 시뮬레이션한 새 행 리스트 반환.

    진입: 신호 봉 종가. 손절: ATR×stop_atr, 목표: 손절거리×target_ratio.
    같은 봉에서 둘 다 닿으면 보수적으로 손절 처리. max_bars 안에 둘 다
    못 닿으면 마지막 종가로 정산. return_pct는 수수료 차감 전(gross),
    집계(_agg)에서 fee_pct 차감 지표가 함께 계산된다.
    """
    out = []
    for r in rows:
        i = r.get("_i")
        atr = r.get("atr") or 0
        if i is None or atr <= 0 or r["position"] not in ("롱", "숏"):
            continue
        entry = r["price"]
        stop_d = atr * stop_atr
        tgt_d = stop_d * target_ratio
        is_long = r["position"] == "롱"
        stop = entry - stop_d if is_long else entry + stop_d
        tgt = entry + tgt_d if is_long else entry - tgt_d

        end = min(len(candles), i + 1 + max_bars)
        outcome, ret = None, None
        for c in candles[i + 1:end]:
            hit_stop = (c["low"] <= stop) if is_long else (c["high"] >= stop)
            hit_tgt = (c["high"] >= tgt) if is_long else (c["low"] <= tgt)
            if hit_stop:  # 동시 도달 시 손절 우선 (보수적)
                outcome, ret = "LOSS", -stop_d / entry * 100
                break
            if hit_tgt:
                outcome, ret = "WIN", tgt_d / entry * 100
                break
        if outcome is None:  # 만기 정산
            close = candles[end - 1]["close"]
            ret = ((close - entry) if is_long else (entry - close)) / entry * 100
            outcome = "WIN" if ret > 0 else "LOSS"
        out.append({**r, "outcome": outcome, "return_pct": round(ret, 3),
                    "exit_mode": "scenario"})
    return out


def verdict_table(by_signal_type, min_n=10):
    """신호유형별 '살려/버려/보류' 판정 리스트.

    기준(수수료 차감 후):
      살려  — net 평균수익 > 0 이고 net 적중률 ≥ 50%
      버려  — net 평균수익 < -0.1% 또는 net 적중률 < 40%
      보류  — 그 외 또는 표본 부족(min_n 미만)
    """
    out = []
    for sig, v in by_signal_type.items():
        n = v.get("n") or 0
        ar_net = v.get("avg_return_net")
        wr_net = v.get("win_rate_net")
        if n < min_n or ar_net is None:
            verdict = "⏸️ 보류(표본부족)" if n < min_n else "⏸️ 보류"
        elif ar_net > 0 and (wr_net or 0) >= 50:
            verdict = "✅ 살려"
        elif ar_net < -0.1 or (wr_net or 0) < 40:
            verdict = "🗑️ 버려"
        else:
            verdict = "⏸️ 보류"
        out.append({
            "signal_type": sig, "verdict": verdict, "n": n,
            "win_rate": v.get("win_rate"), "win_rate_net": wr_net,
            "avg_return": v.get("avg_return"), "avg_return_net": ar_net,
            "avg_mfe": v.get("avg_mfe"), "avg_mae": v.get("avg_mae"),
        })
    out.sort(key=lambda x: (x["avg_return_net"] is None, -(x["avg_return_net"] or 0)))
    return out
