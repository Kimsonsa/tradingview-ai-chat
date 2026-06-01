"""
Binance Futures 실시간 데이터 + 기술적 지표 계산
"""
import requests
import numpy as np
from datetime import datetime


INTERVAL_MAP = {
    "1분": "1m", "3분": "3m", "5분": "5m", "15분": "15m", "30분": "30m",
    "1시간": "1h", "2시간": "2h", "4시간": "4h",
    "1일": "1d", "1주": "1w", "1개월": "1M",
}

INTERVAL_OPTIONS = list(INTERVAL_MAP.keys())


# ── 사용자 메시지에서 타임프레임 감지 ──

import re

# 메시지에서 감지할 타임프레임 키워드 매핑
_TF_KEYWORDS = {
    # 한글 표현
    "1분": "1분", "1분봉": "1분",
    "3분": "3분", "3분봉": "3분",
    "5분": "5분", "5분봉": "5분",
    "15분": "15분", "15분봉": "15분",
    "30분": "30분", "30분봉": "30분",
    "1시간": "1시간", "1시간봉": "1시간",
    "2시간": "2시간", "2시간봉": "2시간",
    "4시간": "4시간", "4시간봉": "4시간",
    "일봉": "1일", "1일봉": "1일", "1일": "1일", "데일리": "1일", "daily": "1일",
    "주봉": "1주", "1주봉": "1주", "1주": "1주", "weekly": "1주",
    "월봉": "1개월", "1개월봉": "1개월", "1개월": "1개월", "monthly": "1개월",
}


def parse_requested_timeframes(user_message, current_interval="15분"):
    """사용자 메시지에서 언급된 타임프레임을 감지하여 리스트로 반환.
    현재 차트 타임프레임은 항상 포함됩니다.
    타임프레임이 추가로 언급되지 않으면 현재 타임프레임만 반환합니다.
    """
    found = set()
    msg_lower = user_message.lower()

    # 긴 키워드부터 매칭 (예: "15분봉"이 "5분봉"보다 먼저)
    sorted_keywords = sorted(_TF_KEYWORDS.keys(), key=len, reverse=True)

    for keyword in sorted_keywords:
        if keyword in msg_lower:
            found.add(_TF_KEYWORDS[keyword])

    # 현재 차트 타임프레임은 항상 포함
    found.add(current_interval)

    # 정렬: 작은 타임프레임 → 큰 타임프레임
    tf_order = ["1분", "3분", "5분", "15분", "30분", "1시간", "2시간", "4시간", "1일", "1주", "1개월"]
    result = [tf for tf in tf_order if tf in found]

    return result


def fetch_klines(symbol="BTCUSDT", interval="1h", limit=210):
    """Binance Futures에서 캔들 데이터 가져오기"""
    url = f"https://fapi.binance.com/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    raw = res.json()
    return [{
        "time": r[0], "open": float(r[1]), "high": float(r[2]),
        "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])
    } for r in raw]


def calc_ema(closes, period):
    k = 2 / (period + 1)
    ema = [closes[0]]
    for c in closes[1:]:
        ema.append(c * k + ema[-1] * (1 - k))
    return ema


def calc_rsi(closes, period=14):
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    rsi_values = []
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi_values.append(round(100 - 100 / (1 + rs), 2))

    return rsi_values


def calc_macd(closes, fast=12, slow=26, signal=9):
    ema_fast = calc_ema(closes, fast)
    ema_slow = calc_ema(closes, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = calc_ema(macd_line[slow-1:], signal)
    histogram = [m - s for m, s in zip(macd_line[slow-1+signal-1:], signal_line)]
    return macd_line[-1], signal_line[-1], histogram[-1] if histogram else 0


def calc_bollinger(closes, period=20, std_dev=2):
    if len(closes) < period:
        return None, None, None, None
    recent = closes[-period:]
    mid = np.mean(recent)
    std = np.std(recent)
    upper = round(mid + std_dev * std, 1)
    lower = round(mid - std_dev * std, 1)
    # 밴드폭: (상단-하단)/중간 × 100 → 수축/확장 판단
    bandwidth = round((upper - lower) / mid * 100, 2) if mid != 0 else 0
    return upper, round(mid, 1), lower, bandwidth


def calc_stoch_rsi(closes, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """스토캐스틱 RSI — RSI에 스토캐스틱을 적용, 더 민감한 과매수/과매도 감지"""
    rsi_vals = calc_rsi(closes, rsi_period)
    if len(rsi_vals) < stoch_period:
        return None, None

    stoch_rsi = []
    for i in range(stoch_period - 1, len(rsi_vals)):
        window = rsi_vals[i - stoch_period + 1:i + 1]
        min_rsi = min(window)
        max_rsi = max(window)
        if max_rsi - min_rsi == 0:
            stoch_rsi.append(50.0)
        else:
            stoch_rsi.append((rsi_vals[i] - min_rsi) / (max_rsi - min_rsi) * 100)

    # %K = SMA of Stoch RSI
    if len(stoch_rsi) < k_smooth:
        return None, None
    k_vals = [np.mean(stoch_rsi[i - k_smooth + 1:i + 1]) for i in range(k_smooth - 1, len(stoch_rsi))]

    # %D = SMA of %K
    if len(k_vals) < d_smooth:
        return round(k_vals[-1], 2), None
    d_vals = [np.mean(k_vals[i - d_smooth + 1:i + 1]) for i in range(d_smooth - 1, len(k_vals))]

    return round(k_vals[-1], 2), round(d_vals[-1], 2)


def calc_atr(candles, period=14):
    """ATR (Average True Range) — 변동성 측정, 손절 설정 기준"""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i - 1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)

    # 초기 ATR = 단순평균
    atr = np.mean(trs[:period])
    # 이후 지수평균
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return round(atr, 2)


def calc_adx(candles, period=14):
    """ADX (Average Directional Index) — 추세 강도 (25 이상 = 추세, 미만 = 횡보)"""
    if len(candles) < period * 2 + 1:
        return None, None, None  # adx, +di, -di

    plus_dm_list = []
    minus_dm_list = []
    tr_list = []

    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_high = candles[i - 1]["high"]
        prev_low = candles[i - 1]["low"]
        prev_close = candles[i - 1]["close"]

        # True Range
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    # Smoothed averages (Wilder's smoothing)
    atr_s = np.mean(tr_list[:period])
    plus_dm_s = np.mean(plus_dm_list[:period])
    minus_dm_s = np.mean(minus_dm_list[:period])

    dx_list = []
    for i in range(period, len(tr_list)):
        atr_s = atr_s - (atr_s / period) + tr_list[i]
        plus_dm_s = plus_dm_s - (plus_dm_s / period) + plus_dm_list[i]
        minus_dm_s = minus_dm_s - (minus_dm_s / period) + minus_dm_list[i]

        plus_di = (plus_dm_s / atr_s) * 100 if atr_s != 0 else 0
        minus_di = (minus_dm_s / atr_s) * 100 if atr_s != 0 else 0

        di_sum = plus_di + minus_di
        dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum != 0 else 0
        dx_list.append(dx)

    if len(dx_list) < period:
        return None, None, None

    # ADX = 이동평균 of DX
    adx = np.mean(dx_list[:period])
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period

    # 최종 DI 값
    final_plus_di = (plus_dm_s / atr_s) * 100 if atr_s != 0 else 0
    final_minus_di = (minus_dm_s / atr_s) * 100 if atr_s != 0 else 0

    return round(adx, 1), round(final_plus_di, 1), round(final_minus_di, 1)


def calc_obv(candles, return_series=False):
    """OBV (On Balance Volume) — 거래량 누적으로 매집/분산 감지

    Args:
        candles: 캔들 데이터 리스트
        return_series: True이면 (obv, obv_ema, obv_list) 반환 (다이버전스 분석용)

    Returns:
        return_series=False: (obv, obv_ema) — 기존 호환
        return_series=True:  (obv, obv_ema, obv_list) — 전체 시계열 포함
    """
    if len(candles) < 2:
        return (None, None, []) if return_series else (None, None)

    obv = 0
    obv_list = [0]
    for i in range(1, len(candles)):
        if candles[i]["close"] > candles[i - 1]["close"]:
            obv += candles[i]["volume"]
        elif candles[i]["close"] < candles[i - 1]["close"]:
            obv -= candles[i]["volume"]
        obv_list.append(obv)

    # OBV의 20일 EMA (추세선)
    obv_ema = calc_ema(obv_list, 20)[-1] if len(obv_list) >= 20 else obv_list[-1]
    if return_series:
        return obv, obv_ema, obv_list
    return obv, obv_ema


def calc_vwap(candles, period=20):
    """VWAP (Volume Weighted Average Price) — 기관 기준 가격"""
    if len(candles) < period:
        period = len(candles)

    recent = candles[-period:]
    total_vol = sum(c["volume"] for c in recent)
    if total_vol == 0:
        return None

    vwap = sum(
        ((c["high"] + c["low"] + c["close"]) / 3) * c["volume"]
        for c in recent
    ) / total_vol
    return round(vwap, 1)


def calc_fibonacci(candles, lookback=50):
    """최근 N봉 고저 기반 피보나치 되돌림 레벨 계산"""
    if len(candles) < lookback:
        lookback = len(candles)
    recent = candles[-lookback:]
    high = max(c["high"] for c in recent)
    low = min(c["low"] for c in recent)
    diff = high - low

    if diff == 0:
        return None

    levels = {
        "0.0% (고점)": round(high, 1),
        "23.6%": round(high - diff * 0.236, 1),
        "38.2%": round(high - diff * 0.382, 1),
        "50.0%": round(high - diff * 0.5, 1),
        "61.8%": round(high - diff * 0.618, 1),
        "78.6%": round(high - diff * 0.786, 1),
        "100.0% (저점)": round(low, 1),
    }
    return levels


def get_market_context(symbol="BTCUSDT", interval_label="1시간"):
    """종합 시장 데이터 컨텍스트 문자열 반환"""
    bi = INTERVAL_MAP.get(interval_label, "1h")

    try:
        candles = fetch_klines(symbol, bi, 210)
    except Exception as e:
        return f"⚠️ 데이터 수집 실패: {e}"

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]
    last = candles[-1]
    cur = last["close"]

    # EMA
    e20 = calc_ema(closes, 20)[-1]
    e50 = calc_ema(closes, 50)[-1]
    e200 = calc_ema(closes, 200)[-1]

    # RSI
    rsi_vals = calc_rsi(closes)
    cur_rsi = rsi_vals[-1] if rsi_vals else 0

    # Stochastic RSI
    stoch_k, stoch_d = calc_stoch_rsi(closes)
    stoch_str = f"%K={stoch_k} | %D={stoch_d}" if stoch_k is not None else "데이터 부족"
    if stoch_k is not None:
        if stoch_k > 80:
            stoch_str += " ⚠️과매수"
        elif stoch_k < 20:
            stoch_str += " ⚠️과매도"
        if stoch_k is not None and stoch_d is not None:
            if stoch_k > stoch_d:
                stoch_str += " (상승 크로스)"
            else:
                stoch_str += " (하락 크로스)"

    # MACD
    macd, macd_sig, macd_hist = calc_macd(closes)

    # 볼린저밴드
    bb_upper, bb_mid, bb_lower, bb_bw = calc_bollinger(closes)
    bb_squeeze = ""
    if bb_bw is not None:
        if bb_bw < 3:
            bb_squeeze = " 🔴스퀴즈(극단적 수축→폭발 임박)"
        elif bb_bw < 5:
            bb_squeeze = " 🟡수축 중"
        elif bb_bw > 10:
            bb_squeeze = " 🟢확장(변동성 높음)"

    # ATR
    atr = calc_atr(candles)
    atr_str = f"{atr}" if atr is not None else "데이터 부족"
    if atr is not None:
        atr_pct = round(atr / cur * 100, 2)
        atr_str += f" ({atr_pct}%)"

    # ADX
    adx, plus_di, minus_di = calc_adx(candles)
    adx_str = "데이터 부족"
    if adx is not None:
        if adx >= 40:
            strength = "매우 강한 추세 🔥"
        elif adx >= 25:
            strength = "추세 진행 중 📈"
        elif adx >= 20:
            strength = "약한 추세"
        else:
            strength = "횡보/추세 없음 ↔"
        di_direction = "상승 우세 ↑" if plus_di > minus_di else "하락 우세 ↓"
        adx_str = f"ADX={adx} ({strength}) | +DI={plus_di} | -DI={minus_di} → {di_direction}"

    # OBV
    obv, obv_ema = calc_obv(candles)
    obv_str = "데이터 부족"
    if obv is not None:
        obv_trend = "매집 신호 ↑" if obv > obv_ema else "분산 신호 ↓"
        obv_str = f"OBV={obv:,.0f} | EMA={obv_ema:,.0f} → {obv_trend}"

    # VWAP
    vwap = calc_vwap(candles)
    vwap_str = "데이터 부족"
    if vwap is not None:
        vwap_pos = "현재가 VWAP 위 (매수 우위)" if cur > vwap else "현재가 VWAP 아래 (매도 우위)"
        vwap_str = f"{vwap:.1f} → {vwap_pos}"



    # 거래량
    avg_vol5 = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
    vol_ratio = int(last["volume"] / avg_vol5 * 100) if avg_vol5 > 0 else 100

    # 고저
    recent20 = candles[-20:]
    high20 = max(c["high"] for c in recent20)
    low20 = min(c["low"] for c in recent20)

    # EMA 배열
    if cur > e20 > e50 > e200:
        trend = "강한 상승 정배열 ↑"
    elif cur > e20 > e50:
        trend = "상승 추세 ↑"
    elif cur < e20 < e50 < e200:
        trend = "강한 하락 역배열 ↓"
    elif cur < e20 < e50:
        trend = "하락 추세 ↓"
    else:
        trend = "횡보/혼조 ↔"

    # 최근 5봉
    recent5 = candles[-5:]
    candle_str = "\n".join(
        f"  {datetime.fromtimestamp(c['time']/1000).strftime('%m/%d %H:%M')} "
        f"O:{c['open']:.1f} H:{c['high']:.1f} L:{c['low']:.1f} C:{c['close']:.1f} V:{c['volume']:.0f}"
        for c in recent5
    )

    return f"""📊 실시간 데이터 ({symbol} {interval_label}, Binance Futures)
━━━━━━━━━━━━━━━━━━━━━━━
현재가: {cur:.1f} USDT
20봉 고가: {high20:.1f} | 저가: {low20:.1f}

📈 EMA: 20={e20:.1f} | 50={e50:.1f} | 200={e200:.1f}
배열: {trend}
현재가 위치: {'EMA20 위' if cur > e20 else 'EMA20 아래'}

📉 RSI(14): {cur_rsi} {'⚠️과매수' if cur_rsi > 70 else '⚠️과매도' if cur_rsi < 30 else '중립'}
🔄 스토캐스틱 RSI: {stoch_str}

📊 MACD: {macd:.1f} | Signal: {macd_sig:.1f} | Hist: {macd_hist:.1f} {'🟢' if macd_hist > 0 else '🔴'}

📏 볼린저밴드(20,2): 상단={bb_upper} | 중간={bb_mid} | 하단={bb_lower} | 밴드폭={bb_bw}%{bb_squeeze}

📐 ATR(14): {atr_str}
🧭 ADX(14): {adx_str}
📊 OBV: {obv_str}
💰 VWAP(20): {vwap_str}

📊 거래량: 현재봉 {last['volume']:.0f} (5봉평균 대비 {vol_ratio}%)

📋 최근 5봉:
{candle_str}
━━━━━━━━━━━━━━━━━━━━━━━"""


# ── 멀티 타임프레임 (동적) ──

def get_multi_timeframe_context(symbol="BTCUSDT", intervals=None, primary_interval="15분"):
    """요청된 타임프레임들의 실시간 데이터를 가져와서 AI 컨텍스트 문자열 반환.

    Args:
        symbol: 종목 심볼
        intervals: 가져올 타임프레임 리스트 (예: ["15분", "1시간", "4시간"])
                   None이면 primary_interval만 가져옴
        primary_interval: 현재 차트에 열려있는 타임프레임
    """
    if intervals is None:
        intervals = [primary_interval]

    sections = []

    for label in intervals:
        bi = INTERVAL_MAP.get(label)
        if not bi:
            continue

        try:
            candles = fetch_klines(symbol, bi, 210)
        except Exception as e:
            sections.append(f"📊 {label} — ⚠️ 데이터 수집 실패: {e}")
            continue

        closes = [c["close"] for c in candles]
        volumes = [c["volume"] for c in candles]
        last = candles[-1]
        cur = last["close"]

        # EMA
        e20 = calc_ema(closes, 20)[-1]
        e50 = calc_ema(closes, 50)[-1]
        e200 = calc_ema(closes, 200)[-1]

        # RSI
        rsi_vals = calc_rsi(closes)
        cur_rsi = rsi_vals[-1] if rsi_vals else 0

        # Stochastic RSI
        stoch_k, stoch_d = calc_stoch_rsi(closes)
        stoch_str = f"%K={stoch_k} | %D={stoch_d}" if stoch_k is not None else "N/A"
        if stoch_k is not None:
            if stoch_k > 80:
                stoch_str += " ⚠️과매수"
            elif stoch_k < 20:
                stoch_str += " ⚠️과매도"
            if stoch_d is not None:
                stoch_str += " (↑)" if stoch_k > stoch_d else " (↓)"

        # MACD
        macd, macd_sig, macd_hist = calc_macd(closes)

        # 볼린저밴드
        bb_upper, bb_mid, bb_lower, bb_bw = calc_bollinger(closes)
        bb_extra = ""
        if bb_bw is not None:
            if bb_bw < 3:
                bb_extra = f" 밴드폭={bb_bw}% 🔴스퀴즈"
            elif bb_bw < 5:
                bb_extra = f" 밴드폭={bb_bw}% 🟡수축"
            elif bb_bw > 10:
                bb_extra = f" 밴드폭={bb_bw}% 🟢확장"
            else:
                bb_extra = f" 밴드폭={bb_bw}%"

        # ATR
        atr = calc_atr(candles)
        atr_str = f"{atr} ({round(atr/cur*100, 2)}%)" if atr else "N/A"

        # ADX
        adx, plus_di, minus_di = calc_adx(candles)
        adx_str = "N/A"
        if adx is not None:
            if adx >= 40:
                strength = "🔥강추세"
            elif adx >= 25:
                strength = "📈추세"
            elif adx >= 20:
                strength = "약추세"
            else:
                strength = "↔횡보"
            di_dir = "↑" if plus_di > minus_di else "↓"
            adx_str = f"{adx}({strength}) +DI={plus_di} -DI={minus_di}{di_dir}"

        # OBV
        obv, obv_ema = calc_obv(candles)
        obv_str = "N/A"
        if obv is not None:
            obv_str = f"{'매집↑' if obv > obv_ema else '분산↓'}"

        # VWAP
        vwap = calc_vwap(candles)
        vwap_str = f"{vwap:.1f} ({'위' if cur > vwap else '아래'})" if vwap else "N/A"



        # 거래량
        avg_vol5 = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
        vol_ratio = int(last["volume"] / avg_vol5 * 100) if avg_vol5 > 0 else 100

        # 고저
        recent20 = candles[-20:]
        high20 = max(c["high"] for c in recent20)
        low20 = min(c["low"] for c in recent20)

        # 추세 판단
        if cur > e20 > e50 > e200:
            trend = "강한 상승 정배열 ↑"
        elif cur > e20 > e50:
            trend = "상승 추세 ↑"
        elif cur < e20 < e50 < e200:
            trend = "강한 하락 역배열 ↓"
        elif cur < e20 < e50:
            trend = "하락 추세 ↓"
        else:
            trend = "횡보/혼조 ↔"

        # 최근 5봉
        recent5 = candles[-5:]
        candle_str = "\n".join(
            f"  {datetime.fromtimestamp(c['time']/1000).strftime('%m/%d %H:%M')} "
            f"O:{c['open']:.1f} H:{c['high']:.1f} L:{c['low']:.1f} C:{c['close']:.1f} V:{c['volume']:.0f}"
            for c in recent5
        )

        is_primary = "(📸 차트 캡쳐 중)" if label == primary_interval else ""

        sections.append(f"""📊 [{label}] 실시간 데이터 {is_primary} ({symbol}, Binance Futures)
━━━━━━━━━━━━━━━━━━━━━━━
현재가: {cur:.1f} USDT | 20봉 고가: {high20:.1f} | 저가: {low20:.1f}
📈 EMA: 20={e20:.1f} | 50={e50:.1f} | 200={e200:.1f} → {trend}
📉 RSI(14): {cur_rsi} {'⚠️과매수' if cur_rsi > 70 else '⚠️과매도' if cur_rsi < 30 else '중립'} | StochRSI: {stoch_str}
📊 MACD: {macd:.1f} | Sig: {macd_sig:.1f} | Hist: {macd_hist:.1f} {'🟢' if macd_hist > 0 else '🔴'}
📏 볼린저(20,2): 상={bb_upper} | 중={bb_mid} | 하={bb_lower}{bb_extra}
📐 ATR(14): {atr_str} | ADX(14): {adx_str}
💰 VWAP: {vwap_str} | OBV: {obv_str}
📊 거래량: {last['volume']:.0f} (5봉평균 대비 {vol_ratio}%)
📋 최근 5봉:
{candle_str}""")

    # 헤더 생성
    tf_list_str = " / ".join(intervals)
    if len(intervals) == 1:
        header = f"📊 실시간 데이터 ({symbol} {intervals[0]}, Binance Futures)\n══════════════════════════════════════════════"
    else:
        header = f"""📊 멀티 타임프레임 실시간 데이터 ({symbol}, Binance Futures)
현재 차트: {primary_interval} | 조회 타임프레임: {tf_list_str} (모두 실시간)
══════════════════════════════════════════════"""

    return header + "\n\n" + "\n\n".join(sections) + "\n══════════════════════════════════════════════"


