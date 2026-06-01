"""
RSI 파동 분석 모듈 v2 — 시장 레짐 기반 멀티 타임프레임 RSI 사이클 분석

v2 업그레이드 핵심:
- 시장 레짐(RANGE/UP_TREND/DOWN_TREND 등) 분류 후 RSI 해석 분기
- 하락장에서는 RSI 과매도→과매수 한 파동 가정 금지
  → RSI 반등 한계를 45~55로 제한, EMA20/VWAP 회복 여부로 판단
- 다이버전스 3단계: 후보(CANDIDATE) → 확정(CONFIRMED) → 실패(FAILED)
- RSI 회복 강도, 베어 플래그, 거래량 흡수/지속 패턴
- 12요소 롱/숏 점수화 시스템
- 레짐별 목표가, 상위 프레임 필터

기존 이론 유지:
- 과매수(RSI≥80) → 과매도(RSI≤20)까지 하락/횡보 예상 (횡보장에서)
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

# 레짐 한글 라벨
REGIME_LABELS = {
    "UP_TREND": "↑강세", "UP_BIAS": "↑약상승",
    "DOWN_TREND": "↓강하락", "DOWN_BIAS": "↓약하락",
    "RANGE": "↔횡보", "MIXED": "혼조",
}

REGIME_COLORS = {
    "UP_TREND": "#22C55E", "UP_BIAS": "#86EFAC",
    "DOWN_TREND": "#EF4444", "DOWN_BIAS": "#FCA5A5",
    "RANGE": "#F59E0B", "MIXED": "#94A3B8",
}

# 신호 유형 한글 라벨
SIGNAL_LABELS = {
    "STRONG_LONG_REVERSAL": "🟢 강한 추세전환 롱",
    "VALID_LONG": "🟢 유효 롱",
    "SCALP_LONG_ONLY": "🟡 스캘핑 반등만",
    "COUNTER_TREND_SCALP": "🟡 역추세 스캘핑",
    "WEAK_LONG": "⚪ 약한 롱",
    "BEARISH_EXPANSION": "🔴💥 하방 스퀴즈 확장",
    "BULLISH_EXPANSION": "🟢💥 상방 스퀴즈 확장",
    "BEARISH_CONTINUATION": "🔴 하락 지속 숏",
    "SHORT_BIAS": "🔴 숏 우위",
    "WEAK_SHORT": "⚪ 약한 숏",
    "NEUTRAL": "⚪ 중립/관망",
    "NO_SIGNAL": "⚪ 신호 없음",
}


# ═══════════════════════════════════════════════
# AI 프롬프트 (v2)
# ═══════════════════════════════════════════════

RSI_WAVE_SYSTEM_PROMPT = """당신은 암호화폐 기술적 분석 전문가입니다. 사용자의 **멀티 타임프레임 RSI 사이클 이론 v2**를 기반으로 분석합니다.

━━━ RSI 사이클 이론 (v2 업그레이드) ━━━
1. 과매수 = RSI 80 이상 / 과매도 = RSI 20 이하 (절대 기준)
2. **시장 레짐에 따라 RSI 파동 범위가 달라짐** ← 핵심 변경점
   - 상승장: RSI 40~45가 매수 구간, 70~80까지 가능
   - 횡보장: RSI 30 → 70 클래식 파동 작동
   - 하락장: RSI 과매도 반등해도 45~55에서 막히는 게 정상. "과매수까지 간다" 가정 금지
3. 하락 추세에서는 과매도 반등 → EMA20/VWAP에서 저항 → 재하락이 기본 패턴
4. 큰 타임프레임의 한 파동 안에 작은 타임프레임의 여러 파동이 존재
5. 최대 3번까지 고점/저점 갱신 가능 (3-push divergence)
6. 3번째 갱신 시 RSI+거래량 동반 돌파 → 다이버전스 실패 → 손절
7. ADX 낮을 때 (횡보장) RSI 사이클 전략이 더 유효
8. VWAP/EMA 구조 + 거래량으로 반드시 필터링

━━━ RSI 다이버전스 4유형 분류 ━━━
• **정규 하락 다이버전스**: 가격 HH + RSI LH → 상승 모멘텀 약화, 고점 반전 경고
• **히든 하락 다이버전스**: 가격 LH + RSI HH → 하락 추세 지속 신호 (반전이 아닌 지속!)
• **정규 상승 다이버전스**: 가격 LL + RSI HL → 하락 모멘텀 약화, 저점 반전 경고
• **히든 상승 다이버전스**: 가격 HL + RSI LL → 상승 추세 지속 신호 (반전이 아닌 지속!)

핵심 구분:
- 정규 다이버전스 = 추세 반전 경고
- 히든 다이버전스 = 추세 지속 신호

━━━ 다이버전스 3단계 분류 ━━━
• **후보(CANDIDATE)**: 가격 LL + RSI HL (또는 가격 HH + RSI LH) → 반등/하락 가능성
• **확정(CONFIRMED)**: 후보 + EMA20 회복 + VWAP 회복 + RSI 50 회복 → 추세 전환 가능
• **실패(FAILED)**: 후보 후 RSI 50~55 실패 + VWAP 실패 + 저점 재이탈 → 하락 지속

⚠️ 다이버전스 후보만으로 롱/숏 확정 금지!
반드시 VWAP/EMA20/RSI50 회복 여부를 확인 후 판정.

━━━ 거래량 다이버전스 (종합 분석) ━━━
거래량은 방향성이 없는 원시 데이터이므로 OBV와 함께 종합 분석.

■ 고점 기준:
  가격 HH + 거래량↑ = 상승 확인, 추세 지속
  가격 HH + 거래량↓ = 🔻 하락 다이버전스 (상승 추진력 약화) ★핵심★
  가격 LH + 거래량↑ = 🔻 약세 흡수/분산 (매도 우위, 저항 흡수 실패)
  가격 LH + 거래량↓ = 약한 반등, 관심 감소

■ 저점 기준:
  가격 LL + 거래량↑ = 🔻 하락 확인 (매도 압력 증가). 단, 투매 클라이맥스 가능성 별도
  가격 LL + 거래량↓ = 🔺 상승 다이버전스 (매도세 약화) ★핵심★
  가격 HL + 거래량↓ = 🔺 건강한 눌림 (매도 압력 감소)
  가격 HL + 거래량↑ = ⚠️ 매수 방어 강함 or 변동성 증가 (캔들 확인 필요)

⚠️ 주의: 저점 갱신 + 거래량 증가는 상승 다이버전스가 아님! 기본은 하락 확인.
  투매 클라이맥스(긴 아래꼬리 + 종가 회복 + 이후 저점 방어)일 때만 반등 후보.

━━━ OBV 다이버전스 (최우선) ━━━
OBV는 매수/매도 방향이 포함되어 raw volume보다 다이버전스 판단이 정확.
  가격 HH + OBV LH = 하락 다이버전스 (매수 주도 약화)
  가격 LL + OBV HL = 상승 다이버전스 (매도 주도 약화)

다이버전스 우선순위: OBV > RSI > raw volume
동일 방향 신호가 겹치면 확신도 상승, 충돌하면 OBV를 우선.

━━━ 시장 레짐 ━━━
• UP_TREND: 강한 상승 (ADX≥20, EMA 정배열, +DI 우위)
• UP_BIAS: 약한 상승
• RANGE: 횡보 (ADX<18, BB 수축)
• DOWN_TREND: 강한 하락 (ADX≥20, EMA 역배열, -DI 우위)
• DOWN_BIAS: 약한 하락
• MIXED: 혼조

━━━ 핵심 패턴 ━━━
• **베어 플래그**: 급락 후 약한 반등(되돌림<38.2%) → 재하락. RSI 다이버전스가 있어도 무시
• **실패한 상승 다이버전스**: 다이버전스 후보 → RSI 55 미만으로 반등 → VWAP/EMA20 실패 → 저점 재이탈 → 하락 지속 신호
• **RSI 회복 강도**: 과매도 후 RSI 반등 높이가 핵심
  - VERY_WEAK(<45): 하락 지속 / WEAK(<50): 약반등 / NORMAL(<60): 정상 / STRONG(≥60): 전환 가능
• **거래량 흡수**: 큰 거래량 + 긴 아래꼬리 + 종가 위쪽 = 바닥 흡수 (롱 긍정)
• **거래량 지속**: 큰 거래량 + 저가 마감 + 회복 실패 = 하락 지속
• **스퀴즈 확장(BEARISH/BULLISH_EXPANSION)**: BB 이탈 + 거래량 300%↑ + RSI 극단 + VWAP/EMA 이탈 → 기존 롱/숏 즉시 무효화
  - 하방 확장: price < BB하단 + 거래량폭발 + RSI<30 + VWAP아래 + EMA역배열 → 롱 완전 무효, 숏 확실
  - 상방 확장: price > BB상단 + 거래량폭발 + RSI>70 + VWAP위 + EMA정배열 → 숏 완전 무효, 롱 확실
  - RANGE에서 갑자기 BB 이탈하면서 거래량 터지면 그 전의 "롱 우위" 판정은 즉시 뒤집힘

━━━ .00 라운드넘버 변곡점 보조지표 (현물차트 기준) ━━━
사용자는 TradingView에 커스텀 보조지표를 사용합니다.
이 보조지표의 핵심 원리:
• **가격이 .00 (라운드넘버)을 찍을 때 + 거래량이 급증하면** → 그 위치가 **변곡점(고점 H 또는 저점 L)**이 되는 확률이 높음
• 차트 스크린샷에 다음과 같은 라벨이 표시됨:
  - **H .00** (빨간 라벨): 해당 라운드넘버에서 거래량 급증과 함께 고점 변곡점 형성 → 저항선 역할
  - **L .00** (초록 라벨): 해당 라운드넘버에서 거래량 급증과 함께 저점 변곡점 형성 → 지지선 역할
  - **S .00** (매도 신호): 거래량 급등과 함께 매도 변곡점
• 같은 가격대에서 여러 타임프레임에 걸쳐 반복적으로 H .00 또는 L .00이 나타나면 → 매우 강한 지지/저항
• EMA 리턴드(EMA에서 반등), EMA 닿기(EMA 도달 후 반전) 등의 추가 조건이 라벨에 표시되기도 함

RSI 파동 분석에서의 적용 규칙:
1. RSI 극단값과 .00 변곡점이 동시에 나타나면 → 변곡점 신뢰도 크게 상승
2. RSI 과매도 구간에서 L .00이 형성되면 → 반등 가능성 높음 (거래량 확인)
3. RSI 과매수 구간에서 H .00이 형성되면 → 하락 전환 가능성 높음
4. 목표가 설정 시 .00 변곡점 가격대를 우선 참조 (H .00 = 롱 목표/숏 진입, L .00 = 숏 목표/롱 진입)
5. .00 변곡점과 EMA/VWAP/피보나치 레벨이 겹치는 구간은 컨플루언스 존으로 특별 강조
6. 다중 타임프레임에서 같은 가격에 변곡점이 클러스터를 형성하면 → 핵심 지지/저항 레벨로 최우선 언급

━━━ 멀티 타임프레임 역할 ━━━
• 방향 (일봉/4시간/1시간): RSI 50 기준 + EMA/VWAP → 큰 추세
• 셋업 (15분/5분): 진입 구간 식별 (과매수/과매도 + 지지저항)
• 트리거 (1분): 정확한 진입 타이밍 (RSI 극단값 + 구조 이탈)
• ⚠️ 하위 프레임 다이버전스가 있어도 상위 프레임이 강한 하락이면 신뢰도 50% 감소

━━━ 응답 규칙 ━━━
• 원론적/교과서적 설명 금지
• 실시간 수치 기반 구체적 판단만 제공
• 각 관점(스캘핑/데이트레이딩/스윙/장기)별 구체적 조언
• 상위-하위 프레임 간 충돌이 있으면 반드시 언급
• 진입/청산 타점이 있으면 구체적 가격 제시
• 마크다운 취소선(~~텍스트~~) 절대 사용 금지
• 레짐에 따른 목표가 차이를 반드시 언급
  - 하락장 롱 목표: EMA20/VWAP (과매수 목표 금지)
  - 횡보장 롱 목표: BB 상단/RSI 70
  - 상승장 롱 목표: 전고/과매수
• 다이버전스 상태(후보/확정/실패)를 반드시 명시
• ⚠️ 경계선(borderline) RSI가 감지된 타임프레임이 있으면:
  - 파동 맵의 화살표 방향이 실제와 다를 수 있음을 지적
  - RSI 피크/트로프 값과 75/25 기준을 비교하여 실제 방향을 AI 관점에서 판단
• 신호 유형(STRONG_LONG/SCALP_ONLY/BEARISH_CONTINUATION 등)에 따라 매매 전략 차별화

⚠️ 투자 조언이 아닌 기술적 분석 의견입니다."""


# ═══════════════════════════════════════════════
# 시장 레짐 분류
# ═══════════════════════════════════════════════

def determine_market_regime(price, ema20, ema50, vwap, adx, plus_di, minus_di, bb_bw, ema20_slope):
    """시장 레짐(상태) 분류

    Args:
        price: 현재가
        ema20, ema50: EMA 값
        vwap: VWAP (None 가능)
        adx, plus_di, minus_di: ADX 지표 (None 가능)
        bb_bw: 볼린저 밴드폭 (None 가능)
        ema20_slope: EMA20의 최근 5봉 기울기 (%)

    Returns:
        str: 'UP_TREND' | 'UP_BIAS' | 'RANGE' | 'DOWN_TREND' | 'DOWN_BIAS' | 'MIXED'
    """
    adx = adx or 0
    plus_di = plus_di or 0
    minus_di = minus_di or 0
    bb_bw = bb_bw or 5
    vwap = vwap or price  # VWAP 없으면 현재가 기준

    # 횡보장 판정: ADX 낮고, EMA20 기울기 작고, BB 수축
    if adx < 18 and abs(ema20_slope) < 0.3 and bb_bw < 4:
        return "RANGE"

    # 상승 추세 판정
    if price > vwap and ema20 > ema50 and plus_di > minus_di:
        if adx >= 20:
            return "UP_TREND"
        else:
            return "UP_BIAS"

    # 하락 추세 판정
    if price < vwap and ema20 < ema50 and minus_di > plus_di:
        if adx >= 20:
            return "DOWN_TREND"
        else:
            return "DOWN_BIAS"

    return "MIXED"


def get_regime_rsi_params(regime):
    """레짐별 RSI 파라미터 반환

    Returns:
        dict: oversold_zone, overbought_zone, expected_target, bounce_cap 등
    """
    params = {
        "UP_TREND": {
            "oversold_zone": (35, 45),
            "mid_support": 40,
            "overbought_zone": (70, 80),
            "expected_target": 70,
            "bounce_cap": None,
            "desc": "RSI 40 매수, 70~80 도달 가능",
        },
        "UP_BIAS": {
            "oversold_zone": (30, 40),
            "mid_support": 38,
            "overbought_zone": (65, 75),
            "expected_target": 65,
            "bounce_cap": None,
            "desc": "RSI 30~40 매수, 65~75 목표",
        },
        "RANGE": {
            "oversold_zone": (25, 35),
            "mid_support": 50,
            "overbought_zone": (65, 75),
            "expected_target": 65,
            "bounce_cap": None,
            "desc": "RSI 30→70 클래식 파동",
        },
        "DOWN_TREND": {
            "oversold_zone": (20, 30),
            "mid_support": None,
            "overbought_zone": (60, 65),
            "expected_target": 50,
            "bounce_cap": (45, 55),
            "desc": "RSI 반등 45~55 한계, 과매수 기대 금지",
        },
        "DOWN_BIAS": {
            "oversold_zone": (22, 32),
            "mid_support": 45,
            "overbought_zone": (60, 70),
            "expected_target": 55,
            "bounce_cap": (50, 60),
            "desc": "RSI 반등 50~60 한계",
        },
        "MIXED": {
            "oversold_zone": (25, 35),
            "mid_support": 50,
            "overbought_zone": (65, 75),
            "expected_target": 55,
            "bounce_cap": None,
            "desc": "혼조 — 구조 확인 필요",
        },
    }
    return params.get(regime, params["MIXED"])


# ═══════════════════════════════════════════════
# RSI 사이클 판정
# ═══════════════════════════════════════════════

def determine_cycle_position(rsi, prev_rsi, adx, ema_trend, regime=None):
    """RSI 사이클 위치 판정 — 레짐 반영

    Returns:
        (position_label, description)
    """
    regime = regime or "MIXED"
    params = get_regime_rsi_params(regime)
    ob_zone = params["overbought_zone"]
    os_zone = params["oversold_zone"]
    bounce_cap = params.get("bounce_cap")

    # 절대 과매수/과매도 (80/20)
    if rsi >= OVERBOUGHT:
        return "🔴 과매수", "하락/횡보 전환 관찰"
    elif rsi <= OVERSOLD:
        return "🟢 과매도", "반등 전환 관찰"

    # 레짐별 과매수 접근 구간
    if rsi >= ob_zone[0]:
        if rsi > prev_rsi:
            return "🟠 상승 후반", f"RSI 상승 중 ({rsi:.1f}→{ob_zone[1]})"
        else:
            return "🟠 상승 둔화", f"RSI 하락 전환 중 ({rsi:.1f})"

    # 레짐별 과매도 접근 구간
    if rsi <= os_zone[1]:
        if rsi < prev_rsi:
            return "🟠 하락 후반", f"RSI 하락 중 ({rsi:.1f}→{os_zone[0]})"
        else:
            return "🟠 하락 둔화", f"RSI 반등 시작 ({rsi:.1f})"

    # 하락 추세에서 반등 한계 영역
    if bounce_cap and bounce_cap[0] <= rsi <= bounce_cap[1]:
        if regime in ("DOWN_TREND", "DOWN_BIAS"):
            if rsi < prev_rsi:
                return "🟠 반등 한계", f"RSI 재하락 중 — 반등 한계 구간 ({rsi:.1f})"
            else:
                return "🟠 반등 한계 접근", f"RSI {bounce_cap[1]} 근처 — 반등 한계 ({rsi:.1f})"

    # 중립 영역
    if rsi >= 50:
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
        direction = "down" if last_ob_idx > last_os_idx else "up"
    elif last_ob_idx != -1:
        direction = "down"
    elif last_os_idx != -1:
        direction = "up"
    else:
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
        if borderline:
            borderline = {
                "type": "both_near",
                "near_ob": round(max_rsi, 1),
                "near_os": round(min_rsi, 1),
                "msg": f"RSI 범위 {min_rsi:.1f}~{max_rsi:.1f} — 양쪽 경계 근접, AI 판단 필요"
            }
        else:
            borderline = bl

    return direction, borderline


# ═══════════════════════════════════════════════
# 다이버전스 분석 (v2)
# ═══════════════════════════════════════════

def detect_divergence(closes, rsi_values, lookback=30):
    """최근 N봉에서 다이버전스 감지 (기존 호환 — 히든 다이버전스 포함)

    Returns:
        str | None: 다이버전스 유형 문자열 또는 None
    """
    result = detect_divergence_v2(closes, rsi_values, lookback)
    if result is None:
        return None
    type_labels = {
        "BEAR_DIV_CANDIDATE": "🔻 하락 다이버전스",
        "BULL_DIV_CANDIDATE": "🔺 상승 다이버전스",
        "HIDDEN_BEAR_DIV": "🔻 히든 하락 다이버전스",
        "HIDDEN_BULL_DIV": "🔺 히든 상승 다이버전스",
    }
    return type_labels.get(result["type"])


def detect_divergence_v2(closes, rsi_values, lookback=30):
    """다이버전스 감지 v2 — 구조화된 데이터 반환

    Returns:
        dict | None: {type, price/rsi 값들, label} 또는 None
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
            return {
                "type": "BEAR_DIV_CANDIDATE",
                "price_high_1": peaks_price[-2][1],
                "price_high_2": peaks_price[-1][1],
                "rsi_high_1": round(peaks_rsi[-2][1], 1),
                "rsi_high_2": round(peaks_rsi[-1][1], 1),
                "label": "🔻 하락 다이버전스 후보",
            }

    # 상승 다이버전스: 가격 더 낮은 저점 + RSI 더 높은 저점
    if len(troughs_price) >= 2 and len(troughs_rsi) >= 2:
        if (troughs_price[-1][1] < troughs_price[-2][1] and
                troughs_rsi[-1][1] > troughs_rsi[-2][1]):
            return {
                "type": "BULL_DIV_CANDIDATE",
                "price_low_1": troughs_price[-2][1],
                "price_low_2": troughs_price[-1][1],
                "rsi_low_1": round(troughs_rsi[-2][1], 1),
                "rsi_low_2": round(troughs_rsi[-1][1], 1),
                "label": "🔺 상승 다이버전스 후보",
            }

    # 히든 하락 다이버전스: 가격 더 낮은 고점 + RSI 더 높은 고점 (하락 추세 지속)
    if len(peaks_price) >= 2 and len(peaks_rsi) >= 2:
        if (peaks_price[-1][1] < peaks_price[-2][1] and
                peaks_rsi[-1][1] > peaks_rsi[-2][1]):
            return {
                "type": "HIDDEN_BEAR_DIV",
                "price_high_1": peaks_price[-2][1],
                "price_high_2": peaks_price[-1][1],
                "rsi_high_1": round(peaks_rsi[-2][1], 1),
                "rsi_high_2": round(peaks_rsi[-1][1], 1),
                "label": "🔻 히든 하락 다이버전스 (하락 지속)",
            }

    # 히든 상승 다이버전스: 가격 더 높은 저점 + RSI 더 낮은 저점 (상승 추세 지속)
    if len(troughs_price) >= 2 and len(troughs_rsi) >= 2:
        if (troughs_price[-1][1] > troughs_price[-2][1] and
                troughs_rsi[-1][1] < troughs_rsi[-2][1]):
            return {
                "type": "HIDDEN_BULL_DIV",
                "price_low_1": troughs_price[-2][1],
                "price_low_2": troughs_price[-1][1],
                "rsi_low_1": round(troughs_rsi[-2][1], 1),
                "rsi_low_2": round(troughs_rsi[-1][1], 1),
                "label": "🔺 히든 상승 다이버전스 (상승 지속)",
            }

    return None


def _find_local_extremes(values, lookback=30, pivot_range=2):
    """시계열에서 로컬 고점/저점 피봇 찾기 (범용 헬퍼)

    Args:
        values: 시계열 데이터
        lookback: 최근 N개 데이터만 사용
        pivot_range: 좌우 비교 범위

    Returns:
        (peaks, troughs): [(index, value), ...]
    """
    if len(values) < lookback:
        recent = list(values)
    else:
        recent = values[-lookback:]

    peaks = []
    troughs = []

    for i in range(pivot_range, len(recent) - pivot_range):
        is_peak = all(recent[i] > recent[i - j] and recent[i] > recent[i + j]
                      for j in range(1, pivot_range + 1))
        is_trough = all(recent[i] < recent[i - j] and recent[i] < recent[i + j]
                        for j in range(1, pivot_range + 1))
        if is_peak:
            peaks.append((i, recent[i]))
        if is_trough:
            troughs.append((i, recent[i]))

    return peaks, troughs


def detect_volume_divergence(closes, volumes, lookback=30):
    """거래량 다이버전스 감지 — 가격과 거래량의 관계 8가지 패턴 분류

    거래량 다이버전스 규칙:
    고점 기준:
      가격 HH + 거래량↑ = 상승 확인 (추세 지속)
      가격 HH + 거래량↓ = 하락 다이버전스 (상승 힘 약화) ★
      가격 LH + 거래량↑ = 약세 흡수/분산 (매도 우위)
      가격 LH + 거래량↓ = 약한 반등, 관심 감소
    저점 기준:
      가격 LL + 거래량↑ = 하락 확인 (매도 압력 증가)
      가격 LL + 거래량↓ = 상승 다이버전스 (매도세 약화) ★
      가격 HL + 거래량↓ = 건강한 눌림 (매도 압력 감소)
      가격 HL + 거래량↑ = 매수 방어 강함 or 변동성 증가

    Returns:
        dict | None: {type, pattern, label, detail, bias, strength}
    """
    if len(closes) < lookback or len(volumes) < lookback:
        return None

    price_peaks, price_troughs = _find_local_extremes(closes, lookback)

    recent_closes = closes[-lookback:] if len(closes) >= lookback else list(closes)
    recent_vols = volumes[-lookback:] if len(volumes) >= lookback else list(volumes)

    def _get_vol_at_pivot(idx, data, window=2):
        start = max(0, idx - window)
        end = min(len(data), idx + window + 1)
        return np.mean(data[start:end])

    results = []

    # ═══ 고점 기준 분석 ═══
    if len(price_peaks) >= 2:
        p1_idx, p1_val = price_peaks[-2]
        p2_idx, p2_val = price_peaks[-1]
        v1 = _get_vol_at_pivot(p1_idx, recent_vols)
        v2 = _get_vol_at_pivot(p2_idx, recent_vols)

        price_hh = p2_val > p1_val  # 고점 상승
        vol_up = v2 > v1            # 거래량 상승

        if price_hh and not vol_up:
            # 가격 HH + 거래량↓ = 하락 다이버전스 (상승 힘 약화)
            results.append({
                "type": "VOL_BEAR_DIV",
                "pattern": "HH_VOL_DOWN",
                "label": "🔻 거래량 하락 다이버전스",
                "detail": f"가격 고점↑({p1_val:.1f}→{p2_val:.1f}) + 거래량↓ — 상승 추진력 약화",
                "bias": "BEARISH",
                "strength": 8,
                "price_1": p1_val, "price_2": p2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })
        elif price_hh and vol_up:
            # 가격 HH + 거래량↑ = 상승 확인
            results.append({
                "type": "VOL_BULL_CONFIRM",
                "pattern": "HH_VOL_UP",
                "label": "✅ 거래량 상승 확인",
                "detail": f"가격 고점↑({p1_val:.1f}→{p2_val:.1f}) + 거래량↑ — 추세 지속",
                "bias": "BULLISH",
                "strength": 5,
                "price_1": p1_val, "price_2": p2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })
        elif not price_hh and vol_up:
            # 가격 LH + 거래량↑ = 약세 흡수/분산
            results.append({
                "type": "VOL_BEAR_ABSORPTION",
                "pattern": "LH_VOL_UP",
                "label": "🔻 약세 흡수/분산",
                "detail": f"가격 고점↓({p1_val:.1f}→{p2_val:.1f}) + 거래량↑ — 매도 우위/저항 흡수 실패",
                "bias": "BEARISH",
                "strength": 6,
                "price_1": p1_val, "price_2": p2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })
        else:
            # 가격 LH + 거래량↓ = 약한 반등, 관심 감소
            results.append({
                "type": "VOL_WEAK_BOUNCE",
                "pattern": "LH_VOL_DOWN",
                "label": "⚪ 약한 반등",
                "detail": f"가격 고점↓({p1_val:.1f}→{p2_val:.1f}) + 거래량↓ — 관심 감소",
                "bias": "NEUTRAL",
                "strength": 3,
                "price_1": p1_val, "price_2": p2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })

    # ═══ 저점 기준 분석 ═══
    if len(price_troughs) >= 2:
        t1_idx, t1_val = price_troughs[-2]
        t2_idx, t2_val = price_troughs[-1]
        v1 = _get_vol_at_pivot(t1_idx, recent_vols)
        v2 = _get_vol_at_pivot(t2_idx, recent_vols)

        price_ll = t2_val < t1_val  # 저점 하락
        vol_up = v2 > v1            # 거래량 상승

        if price_ll and vol_up:
            # 가격 LL + 거래량↑ = 하락 확인 (매도 압력 증가)
            results.append({
                "type": "VOL_BEAR_CONFIRM",
                "pattern": "LL_VOL_UP",
                "label": "🔻 하락 확인 (매도 압력 증가)",
                "detail": f"가격 저점↓({t1_val:.1f}→{t2_val:.1f}) + 거래량↑ — 하락 지속. 투매 클라이맥스 가능성 별도 확인",
                "bias": "BEARISH",
                "strength": 7,
                "price_1": t1_val, "price_2": t2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })
        elif price_ll and not vol_up:
            # 가격 LL + 거래량↓ = 상승 다이버전스 (매도세 약화)
            results.append({
                "type": "VOL_BULL_DIV",
                "pattern": "LL_VOL_DOWN",
                "label": "🔺 거래량 상승 다이버전스",
                "detail": f"가격 저점↓({t1_val:.1f}→{t2_val:.1f}) + 거래량↓ — 매도세 약화",
                "bias": "BULLISH",
                "strength": 8,
                "price_1": t1_val, "price_2": t2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })
        elif not price_ll and not vol_up:
            # 가격 HL + 거래량↓ = 건강한 눌림
            results.append({
                "type": "VOL_HEALTHY_PULLBACK",
                "pattern": "HL_VOL_DOWN",
                "label": "🔺 건강한 눌림",
                "detail": f"가격 저점↑({t1_val:.1f}→{t2_val:.1f}) + 거래량↓ — 매도 압력 감소",
                "bias": "BULLISH",
                "strength": 5,
                "price_1": t1_val, "price_2": t2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })
        else:
            # 가격 HL + 거래량↑ = 매수 방어 강함 or 변동성 증가
            results.append({
                "type": "VOL_DEFENSE_OR_VOLATILITY",
                "pattern": "HL_VOL_UP",
                "label": "⚠️ 매수 방어/변동성",
                "detail": f"가격 저점↑({t1_val:.1f}→{t2_val:.1f}) + 거래량↑ — 매수 방어 강함 or 변동성 증가, 캔들 확인 필요",
                "bias": "NEUTRAL",
                "strength": 4,
                "price_1": t1_val, "price_2": t2_val,
                "vol_1": round(v1, 0), "vol_2": round(v2, 0),
            })

    # 가장 강한 신호 반환 (bias가 있는 것 우선)
    if not results:
        return None

    # BEARISH/BULLISH 중 가장 강한 것 반환
    biased = [r for r in results if r["bias"] != "NEUTRAL"]
    if biased:
        return max(biased, key=lambda x: x["strength"])
    return results[0]


def detect_obv_divergence(closes, obv_series, lookback=30):
    """OBV 다이버전스 감지 — raw volume보다 정확한 방향성 거래량 분석

    OBV는 매수/매도 방향성이 포함되어 있어 다이버전스 판단이 더 정확함.
      가격 HH + OBV LH = 하락 다이버전스 (매수 주도 약화)
      가격 LL + OBV HL = 상승 다이버전스 (매도 주도 약화)

    Returns:
        dict | None: {type, label, detail, price/obv 값들}
    """
    if len(closes) < lookback or len(obv_series) < lookback:
        return None

    price_peaks, price_troughs = _find_local_extremes(closes, lookback)
    obv_peaks, obv_troughs = _find_local_extremes(obv_series, lookback)

    recent_obv = obv_series[-lookback:] if len(obv_series) >= lookback else list(obv_series)

    # OBV 값을 가격 피봇 인덱스에서 가져오기
    def _obv_at(idx):
        if idx < len(recent_obv):
            return recent_obv[idx]
        return recent_obv[-1]

    # ═══ 하락 다이버전스: 가격 HH + OBV LH ═══
    if len(price_peaks) >= 2:
        p1_idx, p1_val = price_peaks[-2]
        p2_idx, p2_val = price_peaks[-1]
        obv1 = _obv_at(p1_idx)
        obv2 = _obv_at(p2_idx)

        if p2_val > p1_val and obv2 < obv1:
            return {
                "type": "OBV_BEAR_DIV",
                "label": "🔻 OBV 하락 다이버전스",
                "detail": f"가격 HH({p1_val:.1f}→{p2_val:.1f}) + OBV LH — 매수 주도 약화",
                "bias": "BEARISH",
                "price_1": p1_val, "price_2": p2_val,
                "obv_1": round(obv1, 0), "obv_2": round(obv2, 0),
            }

    # ═══ 상승 다이버전스: 가격 LL + OBV HL ═══
    if len(price_troughs) >= 2:
        t1_idx, t1_val = price_troughs[-2]
        t2_idx, t2_val = price_troughs[-1]
        obv1 = _obv_at(t1_idx)
        obv2 = _obv_at(t2_idx)

        if t2_val < t1_val and obv2 > obv1:
            return {
                "type": "OBV_BULL_DIV",
                "label": "🔺 OBV 상승 다이버전스",
                "detail": f"가격 LL({t1_val:.1f}→{t2_val:.1f}) + OBV HL — 매도 주도 약화",
                "bias": "BULLISH",
                "price_1": t1_val, "price_2": t2_val,
                "obv_1": round(obv1, 0), "obv_2": round(obv2, 0),
            }

    return None


def synthesize_divergence(rsi_div, vol_div, obv_div):
    """RSI + 거래량 + OBV 다이버전스 종합 판정

    우선순위: OBV > RSI > raw volume
    동일 방향 신호가 겹치면 확신도 상승, 충돌하면 우선순위 적용.

    Returns:
        dict: {
            overall_bias: 'BULLISH' | 'BEARISH' | 'NEUTRAL',
            confidence: 'HIGH' | 'MEDIUM' | 'LOW',
            rsi_div, vol_div, obv_div,  # 개별 결과
            signals: [일치하는 신호들],
            conflicts: [충돌하는 신호들],
            summary: str  # 한줄 요약
        }
    """
    bullish_signals = []
    bearish_signals = []
    all_signals = []

    # RSI 다이버전스 분류
    if rsi_div:
        rsi_type = rsi_div.get("type", "")
        if rsi_type in ("BULL_DIV_CANDIDATE", "HIDDEN_BULL_DIV"):
            bullish_signals.append(("RSI", rsi_div["label"], 10))
        elif rsi_type in ("BEAR_DIV_CANDIDATE", "HIDDEN_BEAR_DIV"):
            bearish_signals.append(("RSI", rsi_div["label"], 10))
        all_signals.append(f"RSI: {rsi_div['label']}")

    # OBV 다이버전스 분류 (우선순위 최고)
    if obv_div:
        obv_bias = obv_div.get("bias", "NEUTRAL")
        if obv_bias == "BULLISH":
            bullish_signals.append(("OBV", obv_div["label"], 12))
        elif obv_bias == "BEARISH":
            bearish_signals.append(("OBV", obv_div["label"], 12))
        all_signals.append(f"OBV: {obv_div['label']}")

    # 거래량 다이버전스 분류
    if vol_div:
        vol_bias = vol_div.get("bias", "NEUTRAL")
        if vol_bias == "BULLISH":
            bullish_signals.append(("VOL", vol_div["label"], vol_div.get("strength", 5)))
        elif vol_bias == "BEARISH":
            bearish_signals.append(("VOL", vol_div["label"], vol_div.get("strength", 5)))
        all_signals.append(f"거래량: {vol_div['label']}")

    # 점수 합산
    bull_score = sum(s[2] for s in bullish_signals)
    bear_score = sum(s[2] for s in bearish_signals)

    # 충돌 감지
    conflicts = []
    if bullish_signals and bearish_signals:
        conflicts = [
            f"상승: {', '.join(s[1] for s in bullish_signals)}",
            f"하락: {', '.join(s[1] for s in bearish_signals)}",
        ]

    # 종합 판정
    if bull_score > bear_score:
        overall_bias = "BULLISH"
        dominant = bullish_signals
    elif bear_score > bull_score:
        overall_bias = "BEARISH"
        dominant = bearish_signals
    else:
        overall_bias = "NEUTRAL"
        dominant = []

    # 확신도
    total_sources = len(bullish_signals) + len(bearish_signals)
    if total_sources >= 3 and not conflicts:
        confidence = "HIGH"
    elif total_sources >= 2 and len(dominant) >= 2:
        confidence = "MEDIUM"
    elif total_sources >= 1:
        confidence = "LOW"
    else:
        confidence = "NONE"

    # 요약
    if not all_signals:
        summary = "다이버전스 미감지"
    elif conflicts:
        summary = f"⚠️ 신호 충돌 — {' vs '.join(conflicts)}"
    else:
        bias_label = {"BULLISH": "🔺 상승", "BEARISH": "🔻 하락", "NEUTRAL": "⚪ 중립"}
        summary = f"{bias_label[overall_bias]} 종합 다이버전스 ({confidence}) — {' + '.join(all_signals)}"

    return {
        "overall_bias": overall_bias,
        "confidence": confidence,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "rsi_div": rsi_div,
        "vol_div": vol_div,
        "obv_div": obv_div,
        "signals": all_signals,
        "conflicts": conflicts,
        "summary": summary,
    }


def evaluate_divergence_confirmation(div_info, close, ema20, vwap, rsi, macd_hist, regime):
    """다이버전스 후보의 확정/미확정 판정

    Args:
        div_info: detect_divergence_v2 결과
        close: 현재가
        ema20: EMA20 값
        vwap: VWAP (None 가능)
        rsi: 현재 RSI
        macd_hist: MACD Histogram
        regime: 시장 레짐

    Returns:
        str: 'CONFIRMED' | 'UNCONFIRMED'
    """
    if not div_info:
        return None

    if div_info["type"] == "BULL_DIV_CANDIDATE":
        conditions_met = 0
        conditions_total = 4

        if close > ema20:
            conditions_met += 1
        if vwap and close > vwap:
            conditions_met += 1
        elif not vwap:
            conditions_met += 1  # VWAP 없으면 패스
        if rsi > 50:
            conditions_met += 1
        if macd_hist > 0:
            conditions_met += 1

        # 하락 추세에서는 4/4 필요, 그 외에는 3/4
        threshold = 4 if regime in ("DOWN_TREND", "DOWN_BIAS") else 3
        return "CONFIRMED" if conditions_met >= threshold else "UNCONFIRMED"

    elif div_info["type"] == "BEAR_DIV_CANDIDATE":
        conditions_met = 0

        if close < ema20:
            conditions_met += 1
        if vwap and close < vwap:
            conditions_met += 1
        elif not vwap:
            conditions_met += 1
        if rsi < 50:
            conditions_met += 1
        if macd_hist < 0:
            conditions_met += 1

        return "CONFIRMED" if conditions_met >= 3 else "UNCONFIRMED"

    return None


# ═══════════════════════════════════════════════
# RSI 회복 강도
# ═══════════════════════════════════════════════

def calc_rsi_recovery_strength(rsi_values, lookback=50):
    """과매도 이후 RSI 회복 강도 측정

    Returns:
        dict: {strength, rebound_high, os_low, detail}
    """
    if not rsi_values or len(rsi_values) < 10:
        return {"strength": "NEUTRAL", "rebound_high": None, "os_low": None, "detail": "데이터 부족"}

    recent = rsi_values[-lookback:] if len(rsi_values) >= lookback else list(rsi_values)

    # 최근 과매도 (RSI <= 30) 찾기
    os_idx = -1
    for i in range(len(recent) - 1, -1, -1):
        if recent[i] <= 30:
            os_idx = i
            break

    if os_idx == -1:
        return {"strength": "NEUTRAL", "rebound_high": None, "os_low": None, "detail": "최근 과매도 없음"}

    # 과매도 이후 RSI 최고점
    after_os = recent[os_idx:]
    if len(after_os) < 3:
        return {"strength": "NEUTRAL", "rebound_high": round(recent[-1], 1), "os_low": round(recent[os_idx], 1),
                "detail": "반등 데이터 부족"}

    rebound_high = max(after_os)
    os_low = min(recent[max(0, os_idx - 2):os_idx + 3])

    if rebound_high < 45:
        strength = "VERY_WEAK"
    elif rebound_high < 50:
        strength = "WEAK"
    elif rebound_high < 60:
        strength = "NORMAL"
    else:
        strength = "STRONG"

    return {
        "strength": strength,
        "rebound_high": round(rebound_high, 1),
        "os_low": round(os_low, 1),
        "detail": f"과매도 {os_low:.1f} → 반등 {rebound_high:.1f} ({strength})"
    }


# ═══════════════════════════════════════════════
# 거래량 패턴 분석
# ═══════════════════════════════════════════════

def analyze_volume_pattern(candles, lookback=10):
    """거래량 흡수/지속 패턴 분석

    Returns:
        dict: {pattern: 'ABSORPTION'|'CONTINUATION'|'NEUTRAL', detail, ...}
    """
    if len(candles) < lookback + 1:
        return {"pattern": "NEUTRAL", "detail": "데이터 부족"}

    recent = candles[-lookback:]
    volumes = [c["volume"] for c in recent]
    vol_avg = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]

    # 최근 10봉 중 가장 거래량이 높은 봉 찾기
    max_vol_idx = -1
    max_vol = 0
    for i in range(len(recent)):
        if recent[i]["volume"] > max_vol:
            max_vol = recent[i]["volume"]
            max_vol_idx = i

    if max_vol_idx < 0 or vol_avg == 0:
        return {"pattern": "NEUTRAL", "detail": "거래량 데이터 부족"}

    target = recent[max_vol_idx]
    vol = target["volume"]
    total_range = target["high"] - target["low"]

    if total_range == 0:
        return {"pattern": "NEUTRAL", "detail": "변동 없음"}

    lower_wick = min(target["open"], target["close"]) - target["low"]
    lower_wick_ratio = lower_wick / total_range
    close_position = (target["close"] - target["low"]) / total_range

    # 매수 흡수형 저점 (큰 거래량 + 긴 아래꼬리 + 종가 위쪽)
    if (vol > vol_avg * 1.5 and
        lower_wick_ratio > 0.40 and
        close_position > 0.50):
        # 이후 봉에서 저점 방어 확인
        low_defended = True
        if max_vol_idx < len(recent) - 1:
            for j in range(max_vol_idx + 1, len(recent)):
                if recent[j]["low"] < target["low"]:
                    low_defended = False
                    break

        if low_defended:
            return {
                "pattern": "ABSORPTION",
                "detail": f"거래량 {vol / vol_avg:.1f}x, 아래꼬리 {lower_wick_ratio:.0%}, 종가위치 {close_position:.0%}",
                "vol_ratio": round(vol / vol_avg, 1),
                "wick_ratio": round(lower_wick_ratio, 2),
                "close_pos": round(close_position, 2),
            }

    # 하락 지속형 (큰 거래량 + 저가 마감)
    if (vol > vol_avg * 1.2 and
        close_position < 0.25):
        # 이후 회복 실패 확인
        recovery_failed = True
        if max_vol_idx < len(recent) - 1:
            mid_price = (target["high"] + target["low"]) / 2
            for j in range(max_vol_idx + 1, len(recent)):
                if recent[j]["close"] > mid_price:
                    recovery_failed = False
                    break

        if recovery_failed:
            return {
                "pattern": "CONTINUATION",
                "detail": f"거래량 {vol / vol_avg:.1f}x, 종가 저가근처 {close_position:.0%}, 회복 실패",
                "vol_ratio": round(vol / vol_avg, 1),
                "close_pos": round(close_position, 2),
            }

    return {"pattern": "NEUTRAL", "detail": "특별 패턴 없음"}


# ═══════════════════════════════════════════════
# 패턴 감지
# ═══════════════════════════════════════════════

def detect_bear_flag(candles, rsi_values, atr, ema20, vwap):
    """베어 플래그 패턴 감지

    급락 후 약한 반등(되돌림 < 38.2%) + RSI 과매도 + VWAP/EMA20 아래

    Returns:
        dict | None: 감지 시 상세 정보
    """
    if len(candles) < 30 or atr is None or atr == 0:
        return None

    closes = [c["close"] for c in candles]
    volumes = [c["volume"] for c in candles]

    # 최근 30봉에서 최고점, 최근 15봉 최저점
    recent_high = max(c["high"] for c in candles[-30:])
    recent_low = min(c["low"] for c in candles[-15:])
    drop = recent_high - recent_low

    # 급락 판정: ATR × 2 이상
    if drop < atr * 2:
        return None

    # RSI 최저점 확인
    recent_rsi = rsi_values[-15:] if len(rsi_values) >= 15 else list(rsi_values)
    min_rsi = min(recent_rsi) if recent_rsi else 50

    if min_rsi >= 35:
        return None  # RSI가 충분히 낮지 않았음

    cur = closes[-1]
    vwap_val = vwap or cur

    # 현재가가 VWAP/EMA20 아래여야 함
    if cur > vwap_val and cur > ema20:
        return None

    # 반등 되돌림 측정
    rebound_high = max(closes[-10:]) if len(closes) >= 10 else cur
    retracement = (rebound_high - recent_low) / drop if drop > 0 else 0

    if retracement >= 0.50:
        return None  # 50% 이상 되돌리면 베어플래그 아님

    # RSI 반등 최고점
    rsi_rebound = max(rsi_values[-10:]) if len(rsi_values) >= 10 else 50
    if rsi_rebound >= 60:
        return None

    # 반등 거래량 vs 하락 거래량
    vol_drop = np.mean(volumes[-30:-15]) if len(volumes) >= 30 else np.mean(volumes)
    vol_rebound = np.mean(volumes[-10:]) if len(volumes) >= 10 else np.mean(volumes)

    if vol_drop > 0 and vol_rebound > vol_drop * 1.5:
        return None  # 반등 거래량이 너무 강하면 아님

    return {
        "detected": True,
        "drop": round(drop, 1),
        "retracement": round(retracement, 3),
        "rsi_rebound_high": round(rsi_rebound, 1),
        "vol_ratio": round(vol_rebound / vol_drop, 2) if vol_drop > 0 else 1.0,
        "detail": f"급락 {drop:.1f} ({drop / atr:.1f}×ATR), 되돌림 {retracement:.0%}, RSI반등 {rsi_rebound:.1f}",
    }


def detect_failed_bull_div(div_info, rsi_recovery, close, ema20, vwap, closes):
    """실패한 상승 다이버전스 감지

    조건: 다이버전스 후보 + RSI 회복 약함 + VWAP/EMA20 실패 + 저점 재이탈

    Returns:
        dict | None: 감지 시 상세 정보
    """
    if not div_info or div_info["type"] != "BULL_DIV_CANDIDATE":
        return None

    recovery_strength = rsi_recovery.get("strength", "NEUTRAL")
    rebound_high = rsi_recovery.get("rebound_high") or 50

    conditions = {}
    conditions["weak_recovery"] = recovery_strength in ("VERY_WEAK", "WEAK")
    conditions["rsi_capped"] = rebound_high < 55
    conditions["below_ema20"] = close < ema20
    conditions["below_vwap"] = close < vwap if vwap else True
    conditions["price_below_div_low"] = close < div_info.get("price_low_2", close)

    met = sum(1 for v in conditions.values() if v)

    if met >= 3:
        return {
            "detected": True,
            "conditions_met": met,
            "conditions": conditions,
            "detail": f"다이버전스 실패 — RSI 반등 {rebound_high:.1f}, 회복 {recovery_strength}",
            "signal": "BEAR_CONTINUATION",
        }

    return None


# ═══════════════════════════════════════════════
# 스퀴즈 확장 감지 (Bearish / Bullish Expansion)
# ═══════════════════════════════════════════════

def detect_squeeze_expansion(price, rsi, bb_upper, bb_lower, vwap, ema20, ema50, ema200,
                             vol_ratio, macd_hist, minus_di, plus_di):
    """볼린저밴드 이탈 + 거래량 폭발 = 스퀴즈 확장 감지

    RANGE/약추세에서 갑자기 BB 밖으로 이탈하면서 거래량이 폭발하면
    기존 롱/숏 판정을 즉시 무효화하고 확장 방향으로 전환.

    하방 확장 조건 (BEARISH_EXPANSION):
      - price < BB하단
      - 거래량 > 평균 × 3 (vol_ratio >= 300%)
      - RSI < 30
      - close < VWAP
      - EMA20 < EMA50 (역배열)

    상방 확장 조건 (BULLISH_EXPANSION):
      - price > BB상단
      - 거래량 > 평균 × 3 (vol_ratio >= 300%)
      - RSI > 70
      - close > VWAP
      - EMA20 > EMA50 (정배열)

    Returns:
        dict | None: {type, conditions, detail} 또는 None
    """
    vwap = vwap or price
    minus_di = minus_di or 0
    plus_di = plus_di or 0

    # ── 하방 스퀴즈 확장 ──
    if bb_lower is not None and price < bb_lower:
        conditions = {
            "bb_break": True,
            "vol_spike": vol_ratio >= 300,
            "rsi_oversold": rsi < 30,
            "below_vwap": price < vwap,
            "ema_bearish": ema20 < ema50,
            "ema_full_bearish": ema20 < ema50 < ema200 if ema200 else False,
            "di_bearish": minus_di > plus_di,
            "macd_negative": macd_hist < 0 if macd_hist is not None else False,
        }
        # 핵심 5개 중 4개 이상 충족
        core = [conditions["bb_break"], conditions["vol_spike"],
                conditions["rsi_oversold"], conditions["below_vwap"],
                conditions["ema_bearish"]]
        core_met = sum(1 for v in core if v)
        total_met = sum(1 for v in conditions.values() if v)

        if core_met >= 4:
            return {
                "type": "BEARISH_EXPANSION",
                "core_met": core_met,
                "total_met": total_met,
                "conditions": conditions,
                "detail": (
                    f"하방 확장 — BB하단 이탈, 거래량 {vol_ratio}%, RSI {rsi:.1f}, "
                    f"VWAP/EMA 아래, {core_met}/5 핵심조건 충족"
                ),
            }

    # ── 상방 스퀴즈 확장 ──
    if bb_upper is not None and price > bb_upper:
        conditions = {
            "bb_break": True,
            "vol_spike": vol_ratio >= 300,
            "rsi_overbought": rsi > 70,
            "above_vwap": price > vwap,
            "ema_bullish": ema20 > ema50,
            "ema_full_bullish": ema20 > ema50 > ema200 if ema200 else False,
            "di_bullish": plus_di > minus_di,
            "macd_positive": macd_hist > 0 if macd_hist is not None else False,
        }
        core = [conditions["bb_break"], conditions["vol_spike"],
                conditions["rsi_overbought"], conditions["above_vwap"],
                conditions["ema_bullish"]]
        core_met = sum(1 for v in core if v)
        total_met = sum(1 for v in conditions.values() if v)

        if core_met >= 4:
            return {
                "type": "BULLISH_EXPANSION",
                "core_met": core_met,
                "total_met": total_met,
                "conditions": conditions,
                "detail": (
                    f"상방 확장 — BB상단 돌파, 거래량 {vol_ratio}%, RSI {rsi:.1f}, "
                    f"VWAP/EMA 위, {core_met}/5 핵심조건 충족"
                ),
            }

    return None


# ═══════════════════════════════════════════════
# 목표가 산출
# ═══════════════════════════════════════════════

def calc_regime_targets(regime, price, ema20, ema50, vwap, bb_upper, bb_mid, bb_lower):
    """레짐별 목표가 산출

    Returns:
        dict: {"long": [(label, price, basis), ...], "short": [...], "rsi_target": str}
    """
    targets = {"long": [], "short": [], "rsi_target": ""}
    params = get_regime_rsi_params(regime)
    targets["rsi_target"] = f"RSI {params['expected_target']}"

    if regime in ("DOWN_TREND", "DOWN_BIAS"):
        # 하락 추세 롱 목표 (보수적: EMA20/VWAP까지만)
        if ema20 and ema20 > price:
            targets["long"].append(("TP1", round(ema20, 1), "EMA20"))
        if vwap and vwap > price:
            targets["long"].append(("TP2", round(vwap, 1), "VWAP"))
        if ema50 and ema50 > price:
            targets["long"].append(("TP3", round(ema50, 1), "EMA50"))
        # 숏 목표
        if bb_lower and bb_lower < price:
            targets["short"].append(("TP1", round(bb_lower, 1), "BB하단"))
        bounce_cap = params.get("bounce_cap")
        if bounce_cap:
            targets["rsi_target"] = f"RSI {bounce_cap[0]}~{bounce_cap[1]} (반등 한계)"

    elif regime == "RANGE":
        # 횡보장: BB 기반
        if bb_mid and bb_mid > price:
            targets["long"].append(("TP1", round(bb_mid, 1), "BB중간"))
        if vwap and vwap > price:
            targets["long"].append(("TP2", round(vwap, 1), "VWAP"))
        if bb_upper and bb_upper > price:
            targets["long"].append(("TP3", round(bb_upper, 1), "BB상단"))
        if bb_mid and bb_mid < price:
            targets["short"].append(("TP1", round(bb_mid, 1), "BB중간"))
        if bb_lower and bb_lower < price:
            targets["short"].append(("TP2", round(bb_lower, 1), "BB하단"))
        targets["rsi_target"] = "RSI 65~70"

    elif regime in ("UP_TREND", "UP_BIAS"):
        # 상승장: 과매수까지 홀딩 가능
        if bb_upper and bb_upper > price:
            targets["long"].append(("TP1", round(bb_upper, 1), "BB상단"))
        targets["rsi_target"] = "RSI 70~80"
        # 숏 목표
        if ema20 and ema20 < price:
            targets["short"].append(("TP1", round(ema20, 1), "EMA20"))
        if vwap and vwap < price:
            targets["short"].append(("TP2", round(vwap, 1), "VWAP"))

    else:  # MIXED
        if vwap and vwap > price:
            targets["long"].append(("TP1", round(vwap, 1), "VWAP"))
        if bb_upper and bb_upper > price:
            targets["long"].append(("TP2", round(bb_upper, 1), "BB상단"))
        if vwap and vwap < price:
            targets["short"].append(("TP1", round(vwap, 1), "VWAP"))
        if bb_lower and bb_lower < price:
            targets["short"].append(("TP2", round(bb_lower, 1), "BB하단"))

    return targets


# ═══════════════════════════════════════════════
# 포지션 판정 (v2 — 12요소 점수화)
# ═══════════════════════════════════════════════

def determine_position(r):
    """타임프레임별 롱/숏 포지션 + 확신 등급 판정 (v2 점수 모델)

    Returns:
        dict: {position, confidence, long_score, short_score, signal_type}
    """
    if r.get("error"):
        return {
            "position": "중립", "confidence": "관망",
            "long_score": 0, "short_score": 0, "signal_type": "NO_SIGNAL"
        }

    regime = r.get("regime", "MIXED")
    arrow = r["arrow_dir"]
    rsi = r["rsi"]
    prev_rsi = r["prev_rsi"]
    adx = r.get("adx") or 0
    ema_trend = r.get("ema_trend", "")
    borderline = r.get("borderline")
    div_v2 = r.get("div_v2")
    div_status = r.get("div_status")
    failed_div = r.get("failed_div")
    rsi_recovery = r.get("rsi_recovery") or {}
    vol_pattern = r.get("vol_pattern") or {}
    bear_flag = r.get("bear_flag")
    price = r["price"]
    ema20 = r["ema20"]
    ema50 = r["ema50"]
    vwap = r.get("vwap")
    macd_hist = r.get("macd_hist", 0)

    # ════════════════════════
    # 롱 점수 (0~100+)
    # ════════════════════════
    long_score = 0

    # 1) RSI 과매도 진입
    if rsi <= 25:
        long_score += 15
    elif rsi <= 30:
        long_score += 10
    elif rsi <= 35:
        long_score += 5

    # 2) 상승 다이버전스 후보
    if div_v2 and div_v2["type"] == "BULL_DIV_CANDIDATE":
        long_score += 20

    # 3) 다이버전스 확정
    if div_status == "CONFIRMED":
        long_score += 15
    elif div_status == "UNCONFIRMED":
        long_score -= 5

    # 4) EMA20 회복
    if price > ema20:
        long_score += 10

    # 5) VWAP 회복
    if vwap and price > vwap:
        long_score += 15
    elif not vwap:
        long_score += 5  # VWAP 없으면 중립 가산

    # 6) RSI 50 상회
    if rsi > 50:
        long_score += 10

    # 7) 거래량 흡수
    if vol_pattern.get("pattern") == "ABSORPTION":
        long_score += 15

    # 8) MACD 양전환
    if macd_hist > 0:
        long_score += 5

    # 9) EMA 정배열 일치
    if "상승" in ema_trend:
        long_score += 10
    elif "하락" in ema_trend:
        long_score -= 10

    # 10) 레짐 보정
    if regime == "DOWN_TREND":
        long_score -= 15
    elif regime == "DOWN_BIAS":
        long_score -= 8
    elif regime == "UP_TREND":
        long_score += 10
    elif regime == "UP_BIAS":
        long_score += 5

    # 11) RSI 회복 강도
    recovery_strength = rsi_recovery.get("strength", "NEUTRAL")
    if recovery_strength == "VERY_WEAK":
        long_score -= 10
    elif recovery_strength == "WEAK":
        long_score -= 5
    elif recovery_strength == "STRONG":
        long_score += 10

    # 12) 경계선 / 실패 패턴
    if borderline:
        long_score -= 5
    if failed_div:
        long_score -= 20
    if bear_flag:
        long_score -= 15

    # 13) 스퀴즈 확장 — 즉시 무효화
    squeeze = r.get("squeeze_expansion")
    if squeeze:
        if squeeze["type"] == "BEARISH_EXPANSION":
            long_score -= 50  # 롱 완전 무효화
        elif squeeze["type"] == "BULLISH_EXPANSION":
            long_score += 30

    # 14) 거래량/OBV 종합 다이버전스 (NEW)
    synth_div = r.get("synth_div") or {}
    if synth_div.get("overall_bias") == "BULLISH":
        long_score += synth_div.get("bull_score", 0)
    elif synth_div.get("overall_bias") == "BEARISH":
        long_score -= synth_div.get("bear_score", 0) // 2

    # 15) 히든 다이버전스 (NEW)
    if div_v2 and div_v2.get("type") == "HIDDEN_BULL_DIV":
        long_score += 12  # 상승 추세 지속 신호
    elif div_v2 and div_v2.get("type") == "HIDDEN_BEAR_DIV":
        long_score -= 8  # 하락 지속 신호

    # ════════════════════════
    # 숏 점수 (0~100+)
    # ════════════════════════
    short_score = 0

    # 1) 레짐 하락
    if regime == "DOWN_TREND":
        short_score += 20
    elif regime == "DOWN_BIAS":
        short_score += 12

    # 2) 실패한 상승 다이버전스
    if failed_div:
        short_score += 25

    # 3) RSI 반등 약함
    rebound_high = rsi_recovery.get("rebound_high") or 50
    if rsi_recovery.get("strength") != "NEUTRAL":
        if rebound_high < 45:
            short_score += 15
        elif rebound_high < 55:
            short_score += 10

    # 4) VWAP 회복 실패
    if vwap and price < vwap:
        short_score += 15

    # 5) EMA20 회복 실패
    if price < ema20:
        short_score += 10

    # 6) 베어 플래그
    if bear_flag:
        short_score += 20

    # 7) 하락 지속형 거래량
    if vol_pattern.get("pattern") == "CONTINUATION":
        short_score += 10

    # 8) EMA 역배열
    if "하락" in ema_trend:
        short_score += 10
    elif "상승" in ema_trend:
        short_score -= 10

    # 9) MACD 음수
    if macd_hist < 0:
        short_score += 5

    # 10) RSI 과매수 접근
    if rsi >= 70:
        short_score += 15
    elif rsi >= 65:
        short_score += 8

    # 11) 하락 다이버전스
    if div_v2 and div_v2["type"] == "BEAR_DIV_CANDIDATE":
        short_score += 20

    # 12) ADX 추세 강도 (하락 추세)
    if adx >= 30 and regime in ("DOWN_TREND", "DOWN_BIAS"):
        short_score += 10
    elif adx >= 25 and regime in ("DOWN_TREND", "DOWN_BIAS"):
        short_score += 5

    # 13) 스퀴즈 확장 — 즉시 강화
    if squeeze:
        if squeeze["type"] == "BEARISH_EXPANSION":
            short_score += 40  # 숏 강력 가산
        elif squeeze["type"] == "BULLISH_EXPANSION":
            short_score -= 50  # 숏 무효화

    # 14) 거래량/OBV 종합 다이버전스 (NEW)
    if synth_div.get("overall_bias") == "BEARISH":
        short_score += synth_div.get("bear_score", 0)
    elif synth_div.get("overall_bias") == "BULLISH":
        short_score -= synth_div.get("bull_score", 0) // 2

    # 15) 히든 다이버전스 (NEW)
    if div_v2 and div_v2.get("type") == "HIDDEN_BEAR_DIV":
        short_score += 12  # 하락 추세 지속 신호
    elif div_v2 and div_v2.get("type") == "HIDDEN_BULL_DIV":
        short_score -= 8  # 상승 지속 → 숏 감점

    # ════════════════════════
    # 최종 판정
    # ════════════════════════
    long_score = max(0, long_score)
    short_score = max(0, short_score)

    if long_score >= 75 and long_score > short_score + 20:
        position = "롱"
        confidence = "확실"
        signal_type = "STRONG_LONG_REVERSAL"
    elif long_score >= 60 and long_score > short_score + 15:
        position = "롱"
        confidence = "강함"
        signal_type = "VALID_LONG"
    elif long_score >= 40 and long_score > short_score:
        position = "롱"
        confidence = "우세"
        if regime in ("DOWN_TREND", "DOWN_BIAS"):
            signal_type = "SCALP_LONG_ONLY"
        else:
            signal_type = "VALID_LONG"
    elif short_score >= 70 and short_score > long_score + 20:
        position = "숏"
        confidence = "확실"
        signal_type = "BEARISH_CONTINUATION"
    elif short_score >= 50 and short_score > long_score + 15:
        position = "숏"
        confidence = "강함"
        signal_type = "SHORT_BIAS"
    elif short_score >= 35 and short_score > long_score:
        position = "숏"
        confidence = "우세"
        signal_type = "SHORT_BIAS"
    elif abs(long_score - short_score) < 10:
        position = "중립"
        confidence = "관망"
        signal_type = "NEUTRAL"
    elif long_score > short_score:
        position = "롱"
        confidence = "약간"
        signal_type = "WEAK_LONG"
    else:
        position = "숏"
        confidence = "약간"
        signal_type = "WEAK_SHORT"

    # ════════════════════════
    # 스퀴즈 확장 오버라이드 — 최우선
    # ════════════════════════
    if squeeze:
        if squeeze["type"] == "BEARISH_EXPANSION":
            position = "숏"
            confidence = "확실"
            signal_type = "BEARISH_EXPANSION"
        elif squeeze["type"] == "BULLISH_EXPANSION":
            position = "롱"
            confidence = "확실"
            signal_type = "BULLISH_EXPANSION"

    return {
        "position": position,
        "confidence": confidence,
        "long_score": long_score,
        "short_score": short_score,
        "signal_type": signal_type,
    }


# ═══════════════════════════════════════════════
# 상위 프레임 필터
# ═══════════════════════════════════════════════

def apply_htf_filter(results):
    """상위 프레임 필터 적용 — results 딕셔너리를 in-place 수정

    상위 프레임(1D/4H/1H)이 강한 하락이면 하위 프레임(15m/5m/1m) 롱 점수 50% 감소
    """
    htf_order = ["1주", "1일", "4시간", "1시간"]
    ltf_order = ["15분", "5분", "1분"]

    # 상위 프레임 레짐 집계
    htf_bearish = 0
    htf_bullish = 0
    htf_total = 0

    for tf in htf_order:
        r = results.get(tf)
        if not r or r.get("error"):
            continue
        htf_total += 1
        regime = r.get("regime", "MIXED")
        if regime in ("DOWN_TREND", "DOWN_BIAS"):
            htf_bearish += 1
        elif regime in ("UP_TREND", "UP_BIAS"):
            htf_bullish += 1

    is_htf_bearish = htf_bearish > htf_bullish and htf_bearish >= 2
    is_htf_bullish = htf_bullish > htf_bearish and htf_bullish >= 2

    # 하위 프레임에 필터 적용
    for tf in ltf_order:
        r = results.get(tf)
        if not r or r.get("error"):
            continue

        r["htf_filter"] = "NEUTRAL"

        if is_htf_bearish:
            r["htf_filter"] = "HTF_BEARISH"
            if "long_score" in r:
                r["long_score_original"] = r["long_score"]
                r["long_score"] = int(r["long_score"] * 0.5)
                # signal_type 조정
                if r.get("signal_type") in ("STRONG_LONG_REVERSAL", "VALID_LONG"):
                    r["signal_type"] = "COUNTER_TREND_SCALP"
                # 포지션 재판정
                if r["long_score"] <= r.get("short_score", 0):
                    r["position"] = "숏"
                    r["confidence"] = "우세"
                else:
                    r["confidence"] = "약간"

        elif is_htf_bullish:
            r["htf_filter"] = "HTF_BULLISH"
            if "long_score" in r:
                r["long_score"] += 15
                # 롱 신뢰도 상승
                if r.get("signal_type") == "SCALP_LONG_ONLY":
                    r["signal_type"] = "VALID_LONG"


# ═══════════════════════════════════════════════
# 핵심 분석 함수
# ═══════════════════════════════════════════════

def analyze_rsi_wave(symbol="BTCUSDT"):
    """7개 타임프레임 데이터 수집 → RSI 사이클 분석 결과 반환 (v2)

    분석 파이프라인:
    1. 데이터 수집 + 지표 계산
    2. 시장 레짐 판정
    3. 레짐별 RSI 파라미터 설정
    4. 사이클 판정 (레짐 반영)
    5. 다이버전스 감지 (v2 후보/확정/실패)
    6. RSI 회복 강도
    7. 거래량 패턴 분석
    8. 베어 플래그 감지
    9. 실패한 다이버전스 감지
    10. 목표가 산출
    11. 포지션 점수화 (12요소)
    12. 상위 프레임 필터 (후처리)

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
            ema20_vals = calc_ema(closes, 20)
            e20 = ema20_vals[-1]
            e50 = calc_ema(closes, 50)[-1]
            e200 = calc_ema(closes, 200)[-1]

            # EMA20 기울기 (최근 5봉, % 변화)
            if len(ema20_vals) >= 6:
                ema20_slope = (ema20_vals[-1] - ema20_vals[-6]) / ema20_vals[-6] * 100 if ema20_vals[-6] != 0 else 0
            else:
                ema20_slope = 0

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

            # ── OBV (시계열 포함) ──
            obv, obv_ema, obv_series = calc_obv(candles, return_series=True)

            # ── VWAP ──
            vwap = calc_vwap(candles)

            # ── 거래량 ──
            avg_vol5 = np.mean(volumes[-6:-1]) if len(volumes) >= 6 else volumes[-1]
            vol_ratio = int(last["volume"] / avg_vol5 * 100) if avg_vol5 > 0 else 100

            # ════════════════════════════════════
            # NEW: 시장 레짐 판정
            # ════════════════════════════════════
            regime = determine_market_regime(
                cur, e20, e50, vwap, adx, plus_di, minus_di, bb_bw, ema20_slope
            )
            regime_params = get_regime_rsi_params(regime)

            # ── 사이클 판정 (레짐 반영) ──
            cycle_pos, cycle_desc = determine_cycle_position(
                cur_rsi, prev_rsi, adx, ema_trend, regime
            )
            arrow_dir, borderline = determine_arrow_direction(rsi_vals)

            # ── 다이버전스 (v2 + 기존 호환) ──
            div_type = detect_divergence(closes, rsi_vals)
            div_v2 = detect_divergence_v2(closes, rsi_vals)

            # ── RSI 회복 강도 ──
            rsi_recovery = calc_rsi_recovery_strength(rsi_vals)

            # ── 거래량 패턴 ──
            vol_pattern = analyze_volume_pattern(candles)

            # ── 거래량 다이버전스 (NEW) ──
            vol_div = detect_volume_divergence(closes, volumes)

            # ── OBV 다이버전스 (NEW) ──
            obv_div = detect_obv_divergence(closes, obv_series) if obv_series else None

            # ── 베어 플래그 ──
            bear_flag = detect_bear_flag(candles, rsi_vals, atr, e20, vwap)

            # ── 스퀴즈 확장 감지 (BB 이탈 + 거래량 폭발) ──
            squeeze_expansion = detect_squeeze_expansion(
                cur, cur_rsi, bb_upper, bb_lower, vwap,
                e20, e50, e200, vol_ratio, macd_hist, minus_di, plus_di
            )

            # 스퀴즈 확장 시 레짐 오버라이드
            if squeeze_expansion:
                if squeeze_expansion["type"] == "BEARISH_EXPANSION":
                    regime = "DOWN_TREND"
                    regime_params = get_regime_rsi_params(regime)
                elif squeeze_expansion["type"] == "BULLISH_EXPANSION":
                    regime = "UP_TREND"
                    regime_params = get_regime_rsi_params(regime)
                # 사이클 재판정 (레짐 변경 반영)
                cycle_pos, cycle_desc = determine_cycle_position(
                    cur_rsi, prev_rsi, adx, ema_trend, regime
                )

            # ── 다이버전스 확정/실패 평가 ──
            div_status = None
            if div_v2:
                div_status = evaluate_divergence_confirmation(
                    div_v2, cur, e20, vwap, cur_rsi, macd_hist, regime
                )

            # ── 실패한 상승 다이버전스 ──
            failed_div = None
            if (div_v2 and div_v2["type"] == "BULL_DIV_CANDIDATE"
                    and div_status == "UNCONFIRMED"):
                failed_div = detect_failed_bull_div(
                    div_v2, rsi_recovery, cur, e20, vwap, closes
                )

            # ── 추세장/횡보장 판별 (기존 호환 + 레짐 연동) ──
            if adx is not None and adx >= 25:
                market_type = "추세장"
                rsi_strategy_valid = "⚠️ RSI 사이클 신뢰도 낮음"
            elif adx is not None and adx < 20:
                market_type = "횡보장"
                rsi_strategy_valid = "✅ RSI 사이클 전략 유효"
            else:
                market_type = "약추세"
                rsi_strategy_valid = "🟡 RSI 사이클 보통"

            # ── 목표가 산출 ──
            targets = calc_regime_targets(
                regime, cur, e20, e50, vwap, bb_upper, bb_mid, bb_lower
            )

            results[tf_label] = {
                # 기존 필드 (호환 유지)
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
                # ── v2 새 필드 ──
                "regime": regime,
                "regime_params": regime_params,
                "div_v2": div_v2,
                "div_status": div_status,
                "failed_div": failed_div,
                "rsi_recovery": rsi_recovery,
                "vol_pattern": vol_pattern,
                "bear_flag": bear_flag,
                "squeeze_expansion": squeeze_expansion,
                "targets": targets,
                "ema20_slope": round(ema20_slope, 3),
                # ── v3 거래량/OBV 다이버전스 ──
                "vol_div": vol_div,
                "obv_div": obv_div,
                "synth_div": synthesize_divergence(div_v2, vol_div, obv_div),
            }

            # 포지션 판정 (v2 점수 모델)
            pos_result = determine_position(results[tf_label])
            results[tf_label]["position"] = pos_result["position"]
            results[tf_label]["confidence"] = pos_result["confidence"]
            results[tf_label]["long_score"] = pos_result["long_score"]
            results[tf_label]["short_score"] = pos_result["short_score"]
            results[tf_label]["signal_type"] = pos_result["signal_type"]

        except Exception as e:
            results[tf_label] = {"error": str(e)}

    # ════════════════════════════════════
    # POST-PROCESSING: 상위 프레임 필터
    # ════════════════════════════════════
    apply_htf_filter(results)

    return results


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
    """SVG 화살표 요소 생성 — 순수 위/아래 방향"""
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
        x1, y1 = cx, cy + half
        x2, y2 = cx, cy - half
        p1 = f"{cx},{cy - half}"
        p2 = f"{cx - head_w},{cy - half + 10}"
        p3 = f"{cx + head_w},{cy - half + 10}"
        line_y2 = cy - half + 10
    elif direction == "down":
        x1, y1 = cx, cy - half
        x2, y2 = cx, cy + half
        p1 = f"{cx},{cy + half}"
        p2 = f"{cx - head_w},{cy + half - 10}"
        p3 = f"{cx + head_w},{cy + half - 10}"
        line_y2 = cy + half - 10
    else:
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
    """분석 결과를 SVG 파동 위치 맵으로 변환 (v2 — 레짐 표시 추가)

    Returns:
        str: HTML 문자열 (div + inline SVG)
    """
    # ── 레이아웃 상수 (PAD_B 확장: 레짐 라벨 공간) ──
    W, H = 720, 490
    PAD_L, PAD_R, PAD_T, PAD_B = 60, 25, 45, 120
    PLOT_W = W - PAD_L - PAD_R
    PLOT_H = H - PAD_T - PAD_B

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

    # ── HTML 래퍼 시작 ──
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
        f'🌊 RSI 파동 위치 맵 v2</text>'
    )

    # ── 데이터 포인트 수집 ──
    points = []
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

    # ── 레짐별 RSI 목표 라인 (각 타임프레임별 작은 대시) ──
    for i, tf in enumerate(WAVE_TIMEFRAMES):
        r = results.get(tf)
        if not r or r.get("error"):
            continue
        regime = r.get("regime", "MIXED")
        params = r.get("regime_params") or get_regime_rsi_params(regime)
        target_rsi = params.get("expected_target", 50)
        bounce_cap = params.get("bounce_cap")

        x = x_positions[i]
        target_y = rsi_to_y(target_rsi)
        rc = REGIME_COLORS.get(regime, "#94A3B8")

        # 작은 수평 틱 (목표 RSI)
        tick_w = 12
        svg_parts.append(
            f'  <line x1="{x - tick_w:.1f}" y1="{target_y:.0f}" x2="{x + tick_w:.1f}" y2="{target_y:.0f}" '
            f'stroke="{rc}" stroke-width="1.5" stroke-dasharray="3,2" opacity="0.5"/>'
        )

        # 반등 한계 표시 (하락 추세)
        if bounce_cap:
            cap_y = rsi_to_y(bounce_cap[1])
            svg_parts.append(
                f'  <line x1="{x - tick_w:.1f}" y1="{cap_y:.0f}" x2="{x + tick_w:.1f}" y2="{cap_y:.0f}" '
                f'stroke="#FCA5A5" stroke-width="1" stroke-dasharray="2,2" opacity="0.4"/>'
            )

    # ── 각 타임프레임 화살표 + 라벨 ──
    for x, y, rsi, tf, arrow, color, adx in points:
        svg_parts.append(
            f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" opacity="0.2" '
            f'filter="url(#glow)"/>'
        )
        svg_parts.append(_svg_arrow(x, y, arrow, color, adx))
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

        # TF 라벨
        svg_parts.append(
            f'  <text x="{x:.1f}" y="{H - PAD_B + 22:.0f}" fill="#BBBBCC" '
            f'font-size="12" font-family="Inter,sans-serif" text-anchor="middle" '
            f'font-weight="500">{short}</text>'
        )

        # 포지션 라벨
        if r and not r.get("error"):
            pos = r.get("position", "")
            conf = r.get("confidence", "")
            if pos == "롱":
                pos_color = "#22C55E"
                pos_label = f"▲{pos}:{conf}"
            elif pos == "숏":
                pos_color = "#EF4444"
                pos_label = f"▼{pos}:{conf}"
            else:
                pos_color = "#94A3B8"
                pos_label = f"●{pos}"
            svg_parts.append(
                f'  <text x="{x:.1f}" y="{H - PAD_B + 40:.0f}" fill="{pos_color}" '
                f'font-size="10" font-family="Inter,sans-serif" text-anchor="middle" '
                f'font-weight="600">{pos_label}</text>'
            )

            # 레짐 라벨 (NEW)
            regime = r.get("regime", "")
            if regime:
                rc = REGIME_COLORS.get(regime, "#94A3B8")
                rs = REGIME_LABELS.get(regime, regime)
                svg_parts.append(
                    f'  <text x="{x:.1f}" y="{H - PAD_B + 56:.0f}" fill="{rc}" '
                    f'font-size="9" font-family="Inter,sans-serif" text-anchor="middle" '
                    f'font-weight="500">{rs}</text>'
                )

            # 신호 유형 라벨 (NEW — 핵심 신호만 표시)
            signal = r.get("signal_type", "")
            if signal in ("STRONG_LONG_REVERSAL", "BEARISH_CONTINUATION", "COUNTER_TREND_SCALP", "SCALP_LONG_ONLY",
                          "BEARISH_EXPANSION", "BULLISH_EXPANSION"):
                sig_labels = {
                    "STRONG_LONG_REVERSAL": "강롱전환",
                    "BEARISH_CONTINUATION": "하락지속",
                    "COUNTER_TREND_SCALP": "역추세",
                    "SCALP_LONG_ONLY": "스캘핑만",
                    "BEARISH_EXPANSION": "💥하방확장",
                    "BULLISH_EXPANSION": "💥상방확장",
                }
                sig_colors = {
                    "STRONG_LONG_REVERSAL": "#22C55E",
                    "BEARISH_CONTINUATION": "#EF4444",
                    "COUNTER_TREND_SCALP": "#F59E0B",
                    "SCALP_LONG_ONLY": "#F59E0B",
                    "BEARISH_EXPANSION": "#FF0000",
                    "BULLISH_EXPANSION": "#00FF00",
                }
                sig_text = sig_labels.get(signal, "")
                sig_color = sig_colors.get(signal, "#94A3B8")
                svg_parts.append(
                    f'  <text x="{x:.1f}" y="{H - PAD_B + 70:.0f}" fill="{sig_color}" '
                    f'font-size="8" font-family="Inter,sans-serif" text-anchor="middle" '
                    f'font-weight="600" opacity="0.8">{sig_text}</text>'
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
    """타임프레임별 상세 카드 마크다운 생성 (v2 — 레짐/다이버전스/점수 표시)"""
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
        regime = r.get("regime", "MIXED")
        regime_label = REGIME_LABELS.get(regime, regime)

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

        # 다이버전스 상태 (v2)
        div_str = ""
        div_v2 = r.get("div_v2")
        div_status = r.get("div_status")
        if div_v2:
            status_label = {"CONFIRMED": "✅확정", "UNCONFIRMED": "⏳미확정"}.get(div_status, "")
            div_str = f"\n> {div_v2['label']} — {status_label}"
        failed_div = r.get("failed_div")
        if failed_div:
            div_str += f"\n> ⚠️ **다이버전스 실패** — {failed_div['detail']}"

        # RSI 회복 강도
        rsi_recovery = r.get("rsi_recovery") or {}
        recovery_str = ""
        if rsi_recovery.get("strength") and rsi_recovery["strength"] != "NEUTRAL":
            strength_labels = {
                "VERY_WEAK": "🔴매우약함", "WEAK": "🟠약함",
                "NORMAL": "🟡정상", "STRONG": "🟢강함"
            }
            recovery_str = f" | RSI회복: {strength_labels.get(rsi_recovery['strength'], rsi_recovery['strength'])}"

        # 베어 플래그
        bear_flag_str = ""
        if r.get("bear_flag"):
            bear_flag_str = f"\n> ⚠️ **베어 플래그** — {r['bear_flag']['detail']}"

        # 스퀴즈 확장
        squeeze_str = ""
        if r.get("squeeze_expansion"):
            sq = r["squeeze_expansion"]
            sq_icon = "🔴💥" if sq["type"] == "BEARISH_EXPANSION" else "🟢💥"
            squeeze_str = f"\n> {sq_icon} **스퀴즈 확장** — {sq['detail']}"

        # 거래량 패턴
        vol_pat = r.get("vol_pattern") or {}
        vol_pat_str = ""
        if vol_pat.get("pattern") == "ABSORPTION":
            vol_pat_str = f" | 📊거래량흡수"
        elif vol_pat.get("pattern") == "CONTINUATION":
            vol_pat_str = f" | 📊하락지속형"

        # 거래량/OBV 종합 다이버전스 (NEW)
        synth_div_str = ""
        synth_div = r.get("synth_div")
        if synth_div and synth_div.get("overall_bias") != "NEUTRAL":
            synth_div_str = f"\n> 📊 **종합 다이버전스**: {synth_div['summary']}"
        vol_div = r.get("vol_div")
        if vol_div and vol_div.get("bias") != "NEUTRAL":
            synth_div_str += f"\n> {vol_div['label']} — {vol_div['detail']}"
        obv_div = r.get("obv_div")
        if obv_div:
            synth_div_str += f"\n> {obv_div['label']} — {obv_div['detail']}"

        # 롱/숏 점수
        long_s = r.get("long_score", 0)
        short_s = r.get("short_score", 0)
        signal = r.get("signal_type", "")
        signal_label = SIGNAL_LABELS.get(signal, signal)

        # 목표가
        targets = r.get("targets") or {}
        target_str = ""
        long_tgts = targets.get("long", [])
        short_tgts = targets.get("short", [])
        rsi_tgt = targets.get("rsi_target", "")
        if long_tgts:
            target_str += " | 롱목표: " + ", ".join(f"{t[0]}={t[1]}({t[2]})" for t in long_tgts[:3])
        if short_tgts:
            target_str += " | 숏목표: " + ", ".join(f"{t[0]}={t[1]}({t[2]})" for t in short_tgts[:2])

        # HTF 필터
        htf_str = ""
        htf = r.get("htf_filter", "")
        if htf == "HTF_BEARISH":
            htf_str = " | ⚠️상위프레임 하락 → 롱 감점"
        elif htf == "HTF_BULLISH":
            htf_str = " | ✅상위프레임 상승 → 롱 가점"

        card = f"""**⏱ {tf}** — {r['cycle_pos']} | 레짐: {regime_label}
| 항목 | 값 |
|------|-----|
| 현재가 | {price:,.1f} USDT |
| RSI(14) | **{rsi:.1f}** (이전: {r['prev_rsi']:.1f}){recovery_str} |
| EMA | {ema_str} → {r['ema_trend']} |
| ADX | {adx_str} ({r['market_type']}) |
| VWAP | {vwap_str} |
| 볼밴폭 | {bb_str} |
| MACD Hist | {r['macd_hist']:.2f} {macd_dir} |
| 거래량 | 5봉평균 대비 {r['vol_ratio']}%{vol_pat_str} |

📍 **{r['cycle_desc']}** — {r['rsi_strategy_valid']}
🎯 **{signal_label}** (롱:{long_s} / 숏:{short_s}){htf_str}
📐 {rsi_tgt}{target_str}{div_str}{bear_flag_str}{squeeze_str}{synth_div_str}

---
"""
        cards.append(card)

    return "\n".join(cards)


def generate_summary_text(results):
    """스캘핑/데이/스윙/장기 종합 판정 (v2 — 레짐/다이버전스/HTF 필터)"""
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
                pos = r.get('position', '')
                conf = r.get('confidence', '')
                regime = r.get('regime', '')
                regime_label = REGIME_LABELS.get(regime, '')
                pos_icon = '🟢' if pos == '롱' else '🔴' if pos == '숏' else '⚪'
                signal = r.get('signal_type', '')
                signal_label = SIGNAL_LABELS.get(signal, '')

                line = f"{tf}: {pos_icon} **{pos}:{conf}** ({r['cycle_desc']})"
                if regime_label:
                    line += f" [{regime_label}]"
                summaries.append(line)
        if summaries:
            sections.append(f"• **{group_name}**: {' / '.join(summaries)}")
        else:
            sections.append(f"• **{group_name}**: 데이터 없음")

    # ── 레짐 종합 ──
    regime_summary = []
    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if r and not r.get("error") and r.get("regime"):
            regime_label = REGIME_LABELS.get(r["regime"], r["regime"])
            regime_color = "🟢" if "UP" in r["regime"] else "🔴" if "DOWN" in r["regime"] else "🟡"
            regime_summary.append(f"{TF_LABELS_SHORT.get(tf, tf)}:{regime_color}{regime_label}")
    if regime_summary:
        sections.append(f"\n📊 **레짐 현황**: {' | '.join(regime_summary)}")

    # ── 상위-하위 프레임 충돌 ──
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

    # ── HTF 필터 경고 ──
    for tf in small_tfs:
        r = results.get(tf)
        if r and r.get("htf_filter") == "HTF_BEARISH":
            orig = r.get("long_score_original")
            cur = r.get("long_score", 0)
            if orig is not None:
                conflicts.append(
                    f"⚠️ **{tf} HTF 필터**: 상위 프레임 하락 → 롱 점수 {orig}→{cur} (50% 감점)"
                )
            break  # 하나만 표시

    # ── 다이버전스 상태 ──
    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if r and r.get("failed_div"):
            conflicts.append(
                f"⚠️ **{tf}: 실패한 상승 다이버전스** — {r['failed_div']['detail']} → 하락 지속 가능"
            )
        elif r and r.get("div_v2"):
            status = r.get("div_status", "")
            status_label = {"CONFIRMED": "✅확정", "UNCONFIRMED": "⏳미확정"}.get(status, "")
            conflicts.append(f"📊 **{tf}**: {r['div_v2']['label']} — {status_label}")
        if r and r.get("bear_flag"):
            conflicts.append(f"⚠️ **{tf}: 베어 플래그** — {r['bear_flag']['detail']}")
        if r and r.get("borderline"):
            conflicts.append(f"⚠️ **{tf}**: {r['borderline']['msg']} — AI 판단 필요")
        # 종합 다이버전스 (NEW)
        synth = r.get("synth_div") if r else None
        if synth and synth.get("overall_bias") != "NEUTRAL" and synth.get("confidence") in ("HIGH", "MEDIUM"):
            conflicts.append(f"📊 **{tf} 종합 다이버전스**: {synth['summary']}")

    summary = "### 📊 종합 판정 (v3 — RSI+거래량+OBV)\n\n" + "\n".join(sections)
    if conflicts:
        summary += "\n\n" + "\n".join(conflicts)

    return summary


def format_rsi_wave_for_ai(symbol, results):
    """분석 결과를 AI에게 보낼 텍스트로 포맷팅 (v2 — 레짐/다이버전스/점수/목표가 포함)"""
    lines = [
        f"[🌊 RSI 파동 분석 v2] {symbol} — 레짐 기반 멀티 타임프레임 RSI 사이클 분석\n",
        "아래 7개 타임프레임의 RSI 사이클 상태를 **시장 레짐별로** 분석해주세요.\n",
    ]

    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if not r or r.get("error"):
            continue

        regime = r.get("regime", "MIXED")
        regime_label = REGIME_LABELS.get(regime, regime)
        regime_params = r.get("regime_params") or {}

        lines.append(f"━━━ {tf} ━━━")
        lines.append(f"레짐: {regime} ({regime_label}) — {regime_params.get('desc', '')}")
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

        # 판정 (기존 호환)
        lines.append(f"📍 사이클: {r['cycle_pos']} — {r['cycle_desc']} | {r['rsi_strategy_valid']}")

        # v2 새 필드
        signal = r.get("signal_type", "")
        signal_label = SIGNAL_LABELS.get(signal, signal)
        long_s = r.get("long_score", 0)
        short_s = r.get("short_score", 0)
        lines.append(f"🎯 포지션: {r.get('position','')}:{r.get('confidence','')} — {signal_label} (L:{long_s}/S:{short_s})")

        # 다이버전스 v2
        div_v2 = r.get("div_v2")
        if div_v2:
            status = r.get("div_status", "?")
            lines.append(f"📊 다이버전스: {div_v2['label']} — 상태: {status}")
            if div_v2["type"] == "BULL_DIV_CANDIDATE":
                lines.append(f"   가격: {div_v2.get('price_low_1','?')} → {div_v2.get('price_low_2','?')}")
                lines.append(f"   RSI: {div_v2.get('rsi_low_1','?')} → {div_v2.get('rsi_low_2','?')}")
            elif div_v2["type"] == "BEAR_DIV_CANDIDATE":
                lines.append(f"   가격: {div_v2.get('price_high_1','?')} → {div_v2.get('price_high_2','?')}")
                lines.append(f"   RSI: {div_v2.get('rsi_high_1','?')} → {div_v2.get('rsi_high_2','?')}")
            elif div_v2["type"] == "HIDDEN_BEAR_DIV":
                lines.append(f"   가격 고점: {div_v2.get('price_high_1','?')} → {div_v2.get('price_high_2','?')} (LH)")
                lines.append(f"   RSI 고점: {div_v2.get('rsi_high_1','?')} → {div_v2.get('rsi_high_2','?')} (HH)")
            elif div_v2["type"] == "HIDDEN_BULL_DIV":
                lines.append(f"   가격 저점: {div_v2.get('price_low_1','?')} → {div_v2.get('price_low_2','?')} (HL)")
                lines.append(f"   RSI 저점: {div_v2.get('rsi_low_1','?')} → {div_v2.get('rsi_low_2','?')} (LL)")

        # 실패한 다이버전스
        failed = r.get("failed_div")
        if failed:
            lines.append(f"⚠️ 실패한 다이버전스: {failed['detail']}")

        # RSI 회복 강도
        recovery = r.get("rsi_recovery") or {}
        if recovery.get("strength") and recovery["strength"] != "NEUTRAL":
            lines.append(f"📈 RSI 회복 강도: {recovery['detail']}")

        # 거래량 패턴
        vol_pat = r.get("vol_pattern") or {}
        if vol_pat.get("pattern") != "NEUTRAL":
            lines.append(f"📊 거래량 패턴: {vol_pat['pattern']} — {vol_pat.get('detail','')}")

        # 베어 플래그
        bf = r.get("bear_flag")
        if bf:
            lines.append(f"⚠️ 베어 플래그: {bf['detail']}")

        # 스퀴즈 확장
        sq = r.get("squeeze_expansion")
        if sq:
            lines.append(f"💥 스퀴즈 확장: {sq['type']} — {sq['detail']}")
            lines.append(f"   조건 충족: {sq['core_met']}/5 핵심, {sq['total_met']}/8 전체")

        # 목표가
        targets = r.get("targets") or {}
        long_tgts = targets.get("long", [])
        short_tgts = targets.get("short", [])
        rsi_tgt = targets.get("rsi_target", "")
        if long_tgts:
            lines.append(f"🎯 롱 목표: {', '.join(f'{t[0]}={t[1]}({t[2]})' for t in long_tgts)}")
        if short_tgts:
            lines.append(f"🎯 숏 목표: {', '.join(f'{t[0]}={t[1]}({t[2]})' for t in short_tgts)}")
        if rsi_tgt:
            lines.append(f"🎯 RSI 목표: {rsi_tgt}")

        # HTF 필터
        htf = r.get("htf_filter", "")
        if htf == "HTF_BEARISH":
            orig = r.get("long_score_original", "?")
            lines.append(f"⚠️ HTF 필터: 상위프레임 하락 → 롱 점수 {orig}→{long_s} (감점)")
        elif htf == "HTF_BULLISH":
            lines.append(f"✅ HTF 필터: 상위프레임 상승 → 롱 점수 가점")

        # 거래량 다이버전스 (NEW)
        vol_div_data = r.get("vol_div")
        if vol_div_data:
            lines.append(f"📊 거래량 다이버전스: {vol_div_data['label']} — {vol_div_data['detail']}")

        # OBV 다이버전스 (NEW)
        obv_div_data = r.get("obv_div")
        if obv_div_data:
            lines.append(f"📊 OBV 다이버전스: {obv_div_data['label']} — {obv_div_data['detail']}")

        # 종합 다이버전스 (NEW)
        synth = r.get("synth_div")
        if synth and synth.get("overall_bias") != "NEUTRAL":
            lines.append(f"🎯 종합 다이버전스: {synth['summary']}")
            if synth.get("conflicts"):
                lines.append(f"   ⚠️ 신호 충돌: {' / '.join(synth['conflicts'])}")

        if r.get("borderline"):
            lines.append(f"⚠️ 경계선: {r['borderline']['msg']}")
        lines.append("")

    lines.append("위 데이터를 RSI 사이클 이론 v3에 따라 분석해주세요.")
    lines.append("핵심: 각 타임프레임의 **레짐**을 확인하고, 레짐별 RSI 파동 범위에 맞게 판단하세요.")
    lines.append("하락 추세에서는 RSI 과매수 목표 금지 — 반등 한계(45~55)를 기본값으로 판단하세요.")
    lines.append("다이버전스 후보가 있으면 확정/미확정/실패 여부를 반드시 판정하세요.")
    lines.append("히든 다이버전스는 반전 신호가 아닌 추세 지속 신호임을 명확히 구분하세요.")
    lines.append("거래량/OBV 다이버전스가 RSI 다이버전스와 일치/충돌하는지 반드시 언급하세요.")
    lines.append("각 관점(스캘핑/데이트레이딩/스윙/장기)별 현재 사이클 위치와 매매 방향을 구체적으로 판단하세요.")
    lines.append("상위-하위 프레임 간 충돌이나 다이버전스가 있으면 반드시 언급하세요.")
    lines.append("진입/청산 타점이 보이면 구체적 가격을 제시하세요.")

    return "\n".join(lines)
