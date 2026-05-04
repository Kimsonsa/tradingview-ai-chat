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
        return None, None, None
    recent = closes[-period:]
    mid = np.mean(recent)
    std = np.std(recent)
    return round(mid + std_dev * std, 1), round(mid, 1), round(mid - std_dev * std, 1)


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

    # MACD
    macd, macd_sig, macd_hist = calc_macd(closes)

    # 볼린저밴드
    bb_upper, bb_mid, bb_lower = calc_bollinger(closes)

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

📊 MACD: {macd:.1f} | Signal: {macd_sig:.1f} | Hist: {macd_hist:.1f} {'🟢' if macd_hist > 0 else '🔴'}

📏 볼린저밴드(20,2): 상단={bb_upper} | 중간={bb_mid} | 하단={bb_lower}

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

        # MACD
        macd, macd_sig, macd_hist = calc_macd(closes)

        # 볼린저밴드
        bb_upper, bb_mid, bb_lower = calc_bollinger(closes)

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
📉 RSI(14): {cur_rsi} {'⚠️과매수' if cur_rsi > 70 else '⚠️과매도' if cur_rsi < 30 else '중립'}
📊 MACD: {macd:.1f} | Signal: {macd_sig:.1f} | Hist: {macd_hist:.1f} {'🟢' if macd_hist > 0 else '🔴'}
📏 볼린저(20,2): 상단={bb_upper} | 중간={bb_mid} | 하단={bb_lower}
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


