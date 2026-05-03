"""
Binance Futures 실시간 데이터 + 기술적 지표 계산
"""
import requests
import numpy as np
from datetime import datetime


INTERVAL_MAP = {
    "1분": "1m", "5분": "5m", "15분": "15m",
    "1시간": "1h", "4시간": "4h", "1일": "1d",
}

INTERVAL_OPTIONS = list(INTERVAL_MAP.keys())


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
