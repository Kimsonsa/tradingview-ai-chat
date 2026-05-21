"""
RSI 파동 분석 모듈 — 멀티 타임프레임 RSI 사이클 이론 기반

사용자 이론 핵심:
- 과매수(RSI≥80) → 과매도(RSI≤20)까지 하락/횡보 예상
- 과매도(RSI≤20) → 반대 방향 전환 예상
- 큰 타임프레임 한 파동 안에 작은 타임프레임 여러 파동 존재
- 최대 3번 고점/저점 갱신 (3-push divergence)
- ADX 낮을 때 RSI 사이클 전략이 더 유효
"""
import math
import numpy as np
from core.market_data import (
    fetch_klines, calc_rsi, calc_ema, calc_macd, calc_bollinger,
    calc_stoch_rsi, calc_atr, calc_adx, calc_obv, calc_vwap,
    INTERVAL_MAP,
)

# ═══════════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════════

OVERBOUGHT = 80
OVERSOLD = 20

# 화살표 방향 판정용 (직전 과매수/과매도 탐색)
ARROW_OB = 75
ARROW_OS = 25

WAVE_TIMEFRAMES = ["1분", "5분", "15분", "1시간", "4시간", "1일", "1주"]

TF_LABELS_SHORT = {
    "1분": "1m", "5분": "5m", "15분": "15m",
    "1시간": "1H", "4시간": "4H", "1일": "1D", "1주": "1W",
}

# ═══════════════════════════════════════════════
# AI 프롬프트
# ═══════════════════════════════════════════════

RSI_WAVE_SYSTEM_PROMPT = """당신은 암호화폐 기술적 분석 전문가입니다. 사용자의 **멀티 타임프레임 RSI 사이클 이론**을 기반으로 분석합니다.

━━━ RSI 사이클 이론 ━━━
1. 과매수 = RSI 80 이상 / 과매도 = RSI 20 이하
2. 과매수 도달 후 → 과매도까지 하락 또는 횡보 예상
3. 과매도 도달 후 → 방향 전환 예상
4. 큰 타임프레임의 한 파동 안에 작은 타임프레임의 여러 파동이 존재
5. 작은 프레임 사이클 완료 ≠ 큰 프레임 추세 전환
6. 최대 3번까지 고점/저점 갱신 가능 (3-push divergence)
7. 3번째 갱신 시 RSI+거래량 동반 돌파 → 다이버전스 실패 → 손절
8. ADX 낮을 때 (횡보장) RSI 사이클 전략이 더 유효
9. VWAP/EMA 구조 + 거래량으로 반드시 필터링

━━━ 멀티 타임프레임 역할 ━━━
• 방향 (일봉/4시간/1시간): RSI 50 기준 + EMA/VWAP → 큰 추세
• 셋업 (15분/5분): 진입 구간 식별 (과매수/과매도 + 지지저항)
• 트리거 (1분): 정확한 진입 타이밍 (RSI 극단값 + 구조 이탈)

━━━ 응답 규칙 ━━━
• 원론적/교과서적 설명 금지
• 실시간 수치 기반 구체적 판단만 제공
• 각 관점(스캘핑/데이트레이딩/스윙/장기)별 구체적 조언
• 상위-하위 프레임 간 충돌이 있으면 반드시 언급
• 진입/청산 타점이 있으면 구체적 가격 제시
• 마크다운 취소선(~~텍스트~~) 절대 사용 금지
• ⚠️ 경계선(borderline) RSI가 감지된 타임프레임이 있으면:
  - 파동 맵의 화살표 방향이 실제와 다를 수 있음을 지적
  - RSI 피크/트로프 값과 75/25 기준을 비교하여 실제 방향을 AI 관점에서 판단
  - 예: "맵에는 상승으로 표시되었으나, RSI 최고점 74.2가 과매수(75)에 매우 근접하여 실제로는 하락 국면일 수 있습니다"

⚠️ 투자 조언이 아닌 기술적 분석 의견입니다."""


# ═══════════════════════════════════════════════
# 핵심 분석 함수
# ═══════════════════════════════════════════════

def analyze_rsi_wave(symbol="BTCUSDT"):
    """7개 타임프레임 데이터 수집 → RSI 사이클 분석 결과 반환

    Returns:
        dict: { "1분": {...}, "5분": {...}, ... }
    """
    results = {}

    for tf_label in WAVE_TIMEFRAMES:
        bi = INTERVAL_MAP.get(tf_label)
        if not bi:
            continue

        try:
            candles = fetch_klines(symbol, bi, 210)
            closes = [c["close"] for c in candles]
            volumes = [c["volume"] for c in candles]
            last = candles[-1]
            cur = last["close"]

            # ── RSI ──
            rsi_vals = calc_rsi(closes)
            cur_rsi = rsi_vals[-1] if rsi_vals else 50
            prev_rsi = rsi_vals[-2] if len(rsi_vals) >= 2 else cur_rsi

            # ── EMA ──
            e20 = calc_ema(closes, 20)[-1]
            e50 = calc_ema(closes, 50)[-1]
            e200 = calc_ema(closes, 200)[-1]

            if cur > e20 > e50 > e200:
                ema_trend = "강한 상승 정배열 ↑"
            elif cur > e20 > e50:
                ema_trend = "상승 추세 ↑"
            elif cur < e20 < e50 < e200:
                ema_trend = "강한 하락 역배열 ↓"
            elif cur < e20 < e50:
                ema_trend = "하락 추세 ↓"
            else:
                ema_trend = "횡보/혼조 ↔"

            # ── MACD ──
            macd, macd_sig, macd_hist = calc_macd(closes)

            # ── 볼린저밴드 ──
            bb_upper, bb_mid, bb_lower, bb_bw = calc_bollinger(closes)

            # ── StochRSI ──
            stoch_k, stoch_d = calc_stoch_rsi(closes)

            # ── ATR ──
            atr = calc_atr(candles)
            atr_pct = round(atr / cur * 100, 2) if atr and cur else 0

            # ── ADX ──
            adx, plus_di, minus_di = calc_adx(candles)

            # ── OBV ──
            obv, obv_ema = calc_obv(candles)

            # ── VWAP ──
            vwap = calc_vwap(candles)

            # ── 거래량 ──
            avg_vol5 = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
            vol_ratio = int(last["volume"] / avg_vol5 * 100) if avg_vol5 > 0 else 100

            # ── 사이클 판정 ──
            cycle_pos, cycle_desc = determine_cycle_position(
                cur_rsi, prev_rsi, adx, ema_trend
            )
            arrow_dir, borderline = determine_arrow_direction(rsi_vals)

            # ── 다이버전스 ──
            div_type = detect_divergence(closes, rsi_vals)

            # ── 추세장/횡보장 판별 ──
            if adx is not None and adx >= 25:
                market_type = "추세장"
                rsi_strategy_valid = "⚠️ RSI 사이클 신뢰도 낮음"
            elif adx is not None and adx < 20:
                market_type = "횡보장"
                rsi_strategy_valid = "✅ RSI 사이클 전략 유효"
            else:
                market_type = "약추세"
                rsi_strategy_valid = "🟡 RSI 사이클 보통"

            results[tf_label] = {
                "price": cur,
                "rsi": cur_rsi,
                "prev_rsi": prev_rsi,
                "ema20": e20,
                "ema50": e50,
                "ema200": e200,
                "ema_trend": ema_trend,
                "macd": macd,
                "macd_sig": macd_sig,
                "macd_hist": macd_hist,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "bb_bw": bb_bw,
                "stoch_k": stoch_k,
                "stoch_d": stoch_d,
                "atr": atr,
                "atr_pct": atr_pct,
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "obv": obv,
                "obv_ema": obv_ema,
                "vwap": vwap,
                "vol_ratio": vol_ratio,
                "cycle_pos": cycle_pos,
                "cycle_desc": cycle_desc,
                "arrow_dir": arrow_dir,
                "borderline": borderline,
                "divergence": div_type,
                "market_type": market_type,
                "rsi_strategy_valid": rsi_strategy_valid,
            }
        except Exception as e:
            results[tf_label] = {"error": str(e)}

    return results


def determine_cycle_position(rsi, prev_rsi, adx, ema_trend):
    """RSI 사이클 위치 판정 (80/20 기준)

    Returns:
        (position_label, description)
    """
    if rsi >= OVERBOUGHT:
        return "🔴 과매수", "하락/횡보 전환 관찰"
    elif rsi <= OVERSOLD:
        return "🟢 과매도", "반등 전환 관찰"
    elif 65 <= rsi < OVERBOUGHT:
        if rsi > prev_rsi:
            return "🟠 상승 후반", f"RSI 상승 중 ({rsi:.1f}→80)"
        else:
            return "🟠 상승 둔화", f"RSI 하락 전환 중 ({rsi:.1f})"
    elif OVERSOLD < rsi <= 35:
        if rsi < prev_rsi:
            return "🟠 하락 후반", f"RSI 하락 중 ({rsi:.1f}→20)"
        else:
            return "🟠 하락 둔화", f"RSI 반등 시작 ({rsi:.1f})"
    elif rsi >= 50:
        if rsi > prev_rsi:
            return "🟡 상승 진행", "RSI 50 위 상승 중"
        else:
            return "🟡 상승 조정", "RSI 50 위 하락 중"
    else:
        if rsi < prev_rsi:
            return "🟡 하락 진행", "RSI 50 하회 하락 중"
        else:
            return "🟡 하락 반등", "RSI 50 하회 반등 중"


def determine_arrow_direction(rsi_values):
    """RSI 히스토리에서 직전 과매수/과매도를 찾아 방향 결정

    1. 최근 과매수(≥75) AND 과매도(≤25) 모두 탐색
    2. 더 최근인 것이 방향 결정
    3. 경계선(70~74.9 또는 25.1~30) 감지 → AI 코멘트용

    Returns:
        tuple: (direction: 'up'|'down', borderline: dict|None)
    """
    if not rsi_values or len(rsi_values) < 5:
        return "down", None

    # ── 양쪽 극단값을 모두 탐색 ──
    last_ob_idx = -1   # 최근 과매수 위치
    last_os_idx = -1   # 최근 과매도 위치

    for i in range(len(rsi_values) - 1, -1, -1):
        if rsi_values[i] >= ARROW_OB and last_ob_idx == -1:
            last_ob_idx = i
        if rsi_values[i] <= ARROW_OS and last_os_idx == -1:
            last_os_idx = i
        if last_ob_idx != -1 and last_os_idx != -1:
            break

    # ── 방향 판정 ──
    if last_ob_idx != -1 and last_os_idx != -1:
        # 둘 다 있으면 더 최근인 것이 승리
        direction = "down" if last_ob_idx > last_os_idx else "up"
    elif last_ob_idx != -1:
        direction = "down"
    elif last_os_idx != -1:
        direction = "up"
    else:
        # 극단값 없음 → RSI 50 기준
        direction = "up" if rsi_values[-1] >= 50 else "down"

    # ── 경계선(borderline) 감지 ──
    max_rsi = max(rsi_values)
    min_rsi = min(rsi_values)
    borderline = None

    # 과매수 경계: 피크가 70~74.9인데 75 미달
    if 70 <= max_rsi < ARROW_OB and last_ob_idx == -1:
        borderline = {
            "type": "near_overbought",
            "value": round(max_rsi, 1),
            "msg": f"RSI 최고점 {max_rsi:.1f} — 과매수(75) 근접, 하락 전환 가능성"
        }
    # 과매도 경계: 트러프가 25.1~30인데 25 미달
    if ARROW_OS < min_rsi <= 30 and last_os_idx == -1:
        bl = {
            "type": "near_oversold",
            "value": round(min_rsi, 1),
            "msg": f"RSI 최저점 {min_rsi:.1f} — 과매도(25) 근접, 반등 가능성"
        }
        if borderline:  # 양쪽 모두 경계
            borderline = {
                "type": "both_near",
                "near_ob": round(max_rsi, 1),
                "near_os": round(min_rsi, 1),
                "msg": f"RSI 범위 {min_rsi:.1f}~{max_rsi:.1f} — 양쪽 경계 근접, AI 판단 필요"
            }
        else:
            borderline = bl

    return direction, borderline


def detect_divergence(closes, rsi_values, lookback=30):
    """최근 N봉에서 다이버전스 감지

    Returns:
        str | None: 다이버전스 유형 또는 None
    """
    if len(closes) < lookback or len(rsi_values) < lookback:
        return None

    recent_closes = closes[-lookback:]
    recent_rsi = rsi_values[-lookback:]

    # 고점/저점 탐색 (5봉 기준 로컬 극값)
    peaks_price = []
    peaks_rsi = []
    troughs_price = []
    troughs_rsi = []

    for i in range(2, len(recent_closes) - 2):
        # 고점
        if (recent_closes[i] > recent_closes[i - 1] and
            recent_closes[i] > recent_closes[i + 1] and
            recent_closes[i] > recent_closes[i - 2] and
            recent_closes[i] > recent_closes[i + 2]):
            peaks_price.append((i, recent_closes[i]))
            peaks_rsi.append((i, recent_rsi[i]))
        # 저점
        if (recent_closes[i] < recent_closes[i - 1] and
            recent_closes[i] < recent_closes[i + 1] and
            recent_closes[i] < recent_closes[i - 2] and
            recent_closes[i] < recent_closes[i + 2]):
            troughs_price.append((i, recent_closes[i]))
            troughs_rsi.append((i, recent_rsi[i]))

    # 하락 다이버전스: 가격 더 높은 고점 + RSI 더 낮은 고점
    if len(peaks_price) >= 2 and len(peaks_rsi) >= 2:
        if (peaks_price[-1][1] > peaks_price[-2][1] and
                peaks_rsi[-1][1] < peaks_rsi[-2][1]):
            return "🔻 하락 다이버전스"

    # 상승 다이버전스: 가격 더 낮은 저점 + RSI 더 높은 저점
    if len(troughs_price) >= 2 and len(troughs_rsi) >= 2:
        if (troughs_price[-1][1] < troughs_price[-2][1] and
                troughs_rsi[-1][1] > troughs_rsi[-2][1]):
            return "🔺 상승 다이버전스"

    return None


# ═══════════════════════════════════════════════
# SVG 시각화 생성
# ═══════════════════════════════════════════════

def _get_arrow_color(direction):
    """화살표 방향에 따른 색상: 상승=초록, 하락=빨강"""
    if direction == "up":
        return "#22C55E"
    else:
        return "#EF4444"


def _svg_arrow(cx, cy, direction, color, adx=None):
    """SVG 화살표 요소 생성 — 순수 위/아래 방향

    Args:
        cx, cy: 중심 좌표
        direction: 'up', 'down', 'right'
        color: 화살표 색상
        adx: 추세 강도 (화살표 크기 결정)
    """
    # ADX에 따른 화살표 크기
    if adx is not None and adx >= 40:
        length = 32
        width = 3.5
        head_w = 8
    elif adx is not None and adx >= 25:
        length = 26
        width = 3
        head_w = 7
    else:
        length = 20
        width = 2.5
        head_w = 6

    half = length / 2

    if direction == "up":
        # 순수 위쪽 화살표
        x1, y1 = cx, cy + half
        x2, y2 = cx, cy - half
        # 화살머리: 위쪽 삼각형
        p1 = f"{cx},{cy - half}"          # 꼭짓점 (위)
        p2 = f"{cx - head_w},{cy - half + 10}"  # 왼쪽
        p3 = f"{cx + head_w},{cy - half + 10}"  # 오른쪽
        line_y2 = cy - half + 10
    elif direction == "down":
        # 순수 아래쪽 화살표
        x1, y1 = cx, cy - half
        x2, y2 = cx, cy + half
        # 화살머리: 아래쪽 삼각형
        p1 = f"{cx},{cy + half}"          # 꼭짓점 (아래)
        p2 = f"{cx - head_w},{cy + half - 10}"  # 왼쪽
        p3 = f"{cx + head_w},{cy + half - 10}"  # 오른쪽
        line_y2 = cy + half - 10
    else:
        # fallback: 하락으로 처리
        x1, y1 = cx, cy - half
        x2, y2 = cx, cy + half
        p1 = f"{cx},{cy + half}"
        p2 = f"{cx - head_w},{cy + half - 10}"
        p3 = f"{cx + head_w},{cy + half - 10}"
        line_y2 = cy + half - 10

    return f"""
        <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{line_y2:.1f}"
              stroke="{color}" stroke-width="{width}" stroke-linecap="round"
              filter="url(#glow)"/>
        <polygon points="{p1} {p2} {p3}" fill="{color}" filter="url(#glow)"/>
    """


def generate_wave_svg(results):
    """분석 결과를 SVG 파동 위치 맵으로 변환

    Returns:
        str: HTML 문자열 (div + inline SVG)
    """
    # ── 레이아웃 상수 ──
    W, H = 720, 430
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 25, 45, 70
    PLOT_W = W - PAD_L - PAD_R   # 635
    PLOT_H = H - PAD_T - PAD_B   # 315

    def rsi_to_y(rsi):
        return PAD_T + PLOT_H - (rsi / 100 * PLOT_H)

    # X 위치 계산
    n = len(WAVE_TIMEFRAMES)
    x_margin = 50
    x_start = PAD_L + x_margin
    x_end = W - PAD_R - x_margin
    x_spacing = (x_end - x_start) / (n - 1) if n > 1 else 0
    x_positions = [x_start + i * x_spacing for i in range(n)]

    # Y 기준선
    y_100 = rsi_to_y(100)
    y_80 = rsi_to_y(80)
    y_50 = rsi_to_y(50)
    y_20 = rsi_to_y(20)
    y_0 = rsi_to_y(0)

    svg_parts = []

    # ── HTML 래퍼 시작 (components.html iframe용) ──
    svg_parts.append(f"""<!DOCTYPE html>
<html><head>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>body{{margin:0;padding:0;background:transparent;font-family:'Inter',sans-serif;}}</style>
</head><body>
<div style="width:100%;max-width:720px;margin:0 auto;border-radius:16px;overflow:hidden;
            box-shadow:0 4px 24px rgba(0,0,0,0.25);">
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;display:block;">
  <defs>
    <linearGradient id="bg_grad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#1a1a2e"/>
      <stop offset="100%" stop-color="#16213e"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="2" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <!-- 배경 -->
  <rect width="{W}" height="{H}" fill="url(#bg_grad)" rx="16"/>

  <!-- 과매수 구간 배경 -->
  <rect x="{PAD_L}" y="{y_100:.0f}" width="{PLOT_W}" height="{y_80 - y_100:.0f}"
        fill="rgba(239,68,68,0.07)" rx="4"/>

  <!-- 과매도 구간 배경 -->
  <rect x="{PAD_L}" y="{y_20:.0f}" width="{PLOT_W}" height="{y_0 - y_20:.0f}"
        fill="rgba(34,197,94,0.07)" rx="4"/>
""")

    # ── 수평 기준선 ──
    grid_lines = [
        (100, y_100, "#444466", "4,6", "100"),
        (80,  y_80,  "#EF4444", "6,4", "80"),
        (50,  y_50,  "#555577", "4,6", "50"),
        (20,  y_20,  "#22C55E", "6,4", "20"),
        (0,   y_0,   "#444466", "4,6", "0"),
    ]
    for rsi_val, y, color, dash, label in grid_lines:
        opacity = "0.6" if rsi_val in (80, 20) else "0.3"
        svg_parts.append(
            f'  <line x1="{PAD_L}" y1="{y:.0f}" x2="{W - PAD_R}" y2="{y:.0f}" '
            f'stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" opacity="{opacity}"/>'
        )
        # Y축 라벨
        font_color = color if rsi_val in (80, 20) else "#888899"
        font_weight = "600" if rsi_val in (80, 20) else "400"
        svg_parts.append(
            f'  <text x="{PAD_L - 8}" y="{y + 4:.0f}" fill="{font_color}" '
            f'font-size="11" font-family="Inter,sans-serif" text-anchor="end" '
            f'font-weight="{font_weight}">{label}</text>'
        )

    # ── 과매수/과매도 라벨 ──
    svg_parts.append(
        f'  <text x="{W - PAD_R - 4}" y="{y_80 - 6:.0f}" fill="#EF4444" '
        f'font-size="10" font-family="Inter,sans-serif" text-anchor="end" opacity="0.7">과매수</text>'
    )
    svg_parts.append(
        f'  <text x="{W - PAD_R - 4}" y="{y_20 + 14:.0f}" fill="#22C55E" '
        f'font-size="10" font-family="Inter,sans-serif" text-anchor="end" opacity="0.7">과매도</text>'
    )

    # ── 타이틀 ──
    svg_parts.append(
        f'  <text x="{W / 2}" y="28" fill="#E0E0F0" font-size="15" '
        f'font-family="Inter,sans-serif" text-anchor="middle" font-weight="600">'
        f'🌊 RSI 파동 위치 맵</text>'
    )

    # ── 데이터 포인트 수집 ──
    points = []  # (x, y, rsi, tf_label, arrow_dir, color, adx)
    for i, tf in enumerate(WAVE_TIMEFRAMES):
        r = results.get(tf)
        if not r or r.get("error"):
            continue
        rsi = r["rsi"]
        x = x_positions[i]
        y = rsi_to_y(rsi)
        arrow = r.get("arrow_dir", "right")
        color = _get_arrow_color(arrow)
        adx = r.get("adx")
        points.append((x, y, rsi, tf, arrow, color, adx))

    # ── 연결선 (RSI 프로파일) ──
    if len(points) >= 2:
        polyline_pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points)
        svg_parts.append(
            f'  <polyline points="{polyline_pts}" fill="none" '
            f'stroke="rgba(255,255,255,0.12)" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )

    # ── 각 타임프레임 화살표 + 라벨 ──
    for x, y, rsi, tf, arrow, color, adx in points:
        # 배경 원 (글로우 효과)
        svg_parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" opacity="0.2" '
            f'filter="url(#glow)"/>'
        )
        # 화살표
        svg_parts.append(_svg_arrow(x, y, arrow, color, adx))
        # RSI 값 라벨
        label_y = y - 22 if rsi >= 50 else y + 28
        svg_parts.append(
            f'  <text x="{x:.1f}" y="{label_y:.0f}" fill="{color}" '
            f'font-size="12" font-family="Inter,sans-serif" text-anchor="middle" '
            f'font-weight="600">{rsi:.1f}</text>'
        )

    # ── X축 타임프레임 라벨 ──
    for i, tf in enumerate(WAVE_TIMEFRAMES):
        x = x_positions[i]
        short = TF_LABELS_SHORT.get(tf, tf)
        r = results.get(tf)

        # 아이콘 (사이클 상태)
        if r and not r.get("error"):
            rsi = r["rsi"]
            if rsi >= 80:
                icon = "🔴"
            elif rsi <= 20:
                icon = "🟢"
            elif rsi >= 65 or rsi <= 35:
                icon = "🟠"
            else:
                icon = "🟡"
        else:
            icon = "⚪"

        svg_parts.append(
            f'  <text x="{x:.1f}" y="{H - PAD_B + 22:.0f}" fill="#BBBBCC" '
            f'font-size="12" font-family="Inter,sans-serif" text-anchor="middle" '
            f'font-weight="500">{short}</text>'
        )
        svg_parts.append(
            f'  <text x="{x:.1f}" y="{H - PAD_B + 40:.0f}" fill="#CCCCDD" '
            f'font-size="13" text-anchor="middle">{icon}</text>'
        )

    # ── 범례 ──
    legend_y = H - 12
    legends = [
        ("#22C55E", "▲ 상승 중 (직전 과매도 후)"),
        ("#EF4444", "▼ 하락 중 (직전 과매수 후)"),
    ]
    legend_start = W / 2 - len(legends) * 80 / 2
    for j, (lc, lt) in enumerate(legends):
        lx = legend_start + j * 160
        svg_parts.append(
            f'  <circle cx="{lx:.0f}" cy="{legend_y - 3}" r="4" fill="{lc}"/>'
        )
        svg_parts.append(
            f'  <text x="{lx + 8:.0f}" y="{legend_y:.0f}" fill="#888899" '
            f'font-size="9" font-family="Inter,sans-serif">{lt}</text>'
        )

    # ── 닫기 ──
    svg_parts.append("</svg>\n</div>\n</body></html>")

    return "\n".join(svg_parts)


# ═══════════════════════════════════════════════
# 텍스트 생성
# ═══════════════════════════════════════════════

def generate_tf_cards(results):
    """타임프레임별 상세 카드 마크다운 생성"""
    cards = []

    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if not r or r.get("error"):
            err = r.get("error", "데이터 없음") if r else "데이터 없음"
            cards.append(f"**⏱ {tf}** — ⚠️ {err}\n")
            continue

        # 기본 데이터
        price = r["price"]
        rsi = r["rsi"]
        adx = r.get("adx")
        adx_str = f"{adx}" if adx is not None else "N/A"

        # EMA
        ema_str = f"20={r['ema20']:.1f} | 50={r['ema50']:.1f} | 200={r['ema200']:.1f}"

        # VWAP
        if r.get("vwap"):
            vwap_pos = "위" if price > r["vwap"] else "아래"
            vwap_str = f"{r['vwap']:.1f} ({vwap_pos})"
        else:
            vwap_str = "N/A"

        # 볼밴
        bb_str = "N/A"
        if r.get("bb_bw") is not None:
            bw = r["bb_bw"]
            if bw < 3:
                bb_status = "🔴스퀴즈"
            elif bw < 5:
                bb_status = "🟡수축"
            elif bw > 10:
                bb_status = "🟢확장"
            else:
                bb_status = ""
            bb_str = f"{bw}% {bb_status}"

        # MACD
        macd_dir = "🟢" if r["macd_hist"] > 0 else "🔴"

        # 다이버전스
        div_str = f"\n> ⚠️ **{r['divergence']}** 감지" if r.get("divergence") else ""

        card = f"""**⏱ {tf}** — {r['cycle_pos']}
| 항목 | 값 |
|------|-----|
| 현재가 | {price:,.1f} USDT |
| RSI(14) | **{rsi:.1f}** (이전: {r['prev_rsi']:.1f}) |
| EMA | {ema_str} → {r['ema_trend']} |
| ADX | {adx_str} ({r['market_type']}) |
| VWAP | {vwap_str} |
| 볼밴폭 | {bb_str} |
| MACD Hist | {r['macd_hist']:.2f} {macd_dir} |
| 거래량 | 5봉평균 대비 {r['vol_ratio']}% |

📍 **{r['cycle_desc']}** — {r['rsi_strategy_valid']}{div_str}

---
"""
        cards.append(card)

    return "\n".join(cards)


def generate_summary_text(results):
    """스캘핑/데이/스윙/장기 종합 판정"""
    sections = []

    # 관점별 타임프레임 그룹
    groups = {
        "스캘핑 (1분/5분)": ["1분", "5분"],
        "데이트레이딩 (15분/1시간)": ["15분", "1시간"],
        "스윙 (4시간/일봉)": ["4시간", "1일"],
        "장기 (주봉)": ["1주"],
    }

    for group_name, tfs in groups.items():
        summaries = []
        for tf in tfs:
            r = results.get(tf)
            if r and not r.get("error"):
                summaries.append(f"{tf}: {r['cycle_pos']} ({r['cycle_desc']})")
        if summaries:
            sections.append(f"• **{group_name}**: {' / '.join(summaries)}")
        else:
            sections.append(f"• **{group_name}**: 데이터 없음")

    # 상위-하위 프레임 충돌 확인
    conflicts = []
    big_tfs = ["1일", "4시간", "1시간"]
    small_tfs = ["15분", "5분", "1분"]

    big_direction = None
    small_direction = None

    for tf in big_tfs:
        r = results.get(tf)
        if r and not r.get("error"):
            if r["rsi"] >= 50:
                big_direction = "상승"
            else:
                big_direction = "하락"
            break

    for tf in small_tfs:
        r = results.get(tf)
        if r and not r.get("error"):
            if r["rsi"] >= 50:
                small_direction = "상승"
            else:
                small_direction = "하락"
            break

    if big_direction and small_direction and big_direction != small_direction:
        conflicts.append(
            f"⚠️ **프레임 간 충돌**: 상위({big_direction}) ↔ 하위({small_direction}) — "
            f"하위 프레임의 RSI 사이클이 상위 추세 내 반등/눌림일 수 있음"
        )

    # 다이버전스 경고
    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if r and r.get("divergence"):
            conflicts.append(f"⚠️ **{tf}**: {r['divergence']} — 추세 전환 가능성 주시")
        if r and r.get("borderline"):
            conflicts.append(f"⚠️ **{tf}**: {r['borderline']['msg']} — AI 판단 필요")

    summary = "### 📊 종합 판정\n\n" + "\n".join(sections)
    if conflicts:
        summary += "\n\n" + "\n".join(conflicts)

    return summary


def format_rsi_wave_for_ai(symbol, results):
    """분석 결과를 AI에게 보낼 텍스트로 포맷팅"""
    lines = [
        f"[🌊 RSI 파동 분석] {symbol} — 멀티 타임프레임 RSI 사이클 분석\n",
        "아래 7개 타임프레임의 RSI 사이클 상태를 분석해주세요.\n",
    ]

    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if not r or r.get("error"):
            continue

        lines.append(f"━━━ {tf} ━━━")
        lines.append(f"현재가: {r['price']:,.1f} | RSI: {r['rsi']:.1f} (이전: {r['prev_rsi']:.1f})")
        lines.append(f"EMA: 20={r['ema20']:.1f} 50={r['ema50']:.1f} 200={r['ema200']:.1f} → {r['ema_trend']}")
        lines.append(f"MACD: {r['macd']:.1f} Sig: {r['macd_sig']:.1f} Hist: {r['macd_hist']:.1f}")

        if r.get("bb_upper"):
            lines.append(f"볼밴: 상={r['bb_upper']} 중={r['bb_mid']} 하={r['bb_lower']} 폭={r['bb_bw']}%")
        if r.get("stoch_k") is not None:
            lines.append(f"StochRSI: K={r['stoch_k']} D={r['stoch_d']}")
        if r.get("adx") is not None:
            lines.append(f"ADX: {r['adx']} (+DI={r['plus_di']} -DI={r['minus_di']}) → {r['market_type']}")
        if r.get("vwap"):
            pos = "위" if r["price"] > r["vwap"] else "아래"
            lines.append(f"VWAP: {r['vwap']:.1f} ({pos})")
        lines.append(f"거래량: 5봉평균 대비 {r['vol_ratio']}%")
        lines.append(f"📍 판정: {r['cycle_pos']} — {r['cycle_desc']} | {r['rsi_strategy_valid']}")
        if r.get("divergence"):
            lines.append(f"⚠️ {r['divergence']} 감지")
        if r.get("borderline"):
            lines.append(f"⚠️ 경계선 판단 필요: {r['borderline']['msg']}")
        lines.append("")

    lines.append("위 데이터를 RSI 사이클 이론(과매수 80/과매도 20 기준)에 따라 분석해주세요.")
    lines.append("각 관점(스캘핑/데이트레이딩/스윙/장기)별 현재 사이클 위치와 매매 방향을 구체적으로 판단하세요.")
    lines.append("상위-하위 프레임 간 충돌이나 다이버전스가 있으면 반드시 언급하세요.")
    lines.append("진입/청산 타점이 보이면 구체적 가격을 제시하세요.")

    return "\n".join(lines)
