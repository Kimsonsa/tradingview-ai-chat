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
from concurrent.futures import ThreadPoolExecutor
from core.market_data import (
    fetch_klines, calc_rsi, calc_ema, calc_macd, calc_bollinger,
    calc_stoch_rsi, calc_atr, calc_adx, calc_obv, calc_cvd, calc_vwap,
    fetch_open_interest_hist, fetch_funding_premium,
    INTERVAL_MAP, KLINE_WARMUP,
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

# OI 히스토리 지원 period (Binance) — 1분/1주는 미지원이라 제외
TF_OI_PERIOD = {
    "5분": "5m", "15분": "15m", "1시간": "1h", "4시간": "4h", "1일": "1d",
}

# 방향 집계용 TF 가중치 (상위 시간대 우선)
DIR_TF_WEIGHT = {
    "1분": 0.5, "5분": 0.8, "15분": 1.0, "1시간": 1.5,
    "4시간": 2.0, "1일": 2.5, "1주": 1.5,
}
CONF_WEIGHT = {"확실": 1.0, "강함": 0.8, "우세": 0.6, "약간": 0.3, "관망": 0.0}

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

━━━ CVD 다이버전스 (최우선 — 실제 시장가 체결) ━━━
CVD(Cumulative Volume Delta)는 시장가 매수 - 시장가 매도의 누적으로,
공격적 주문 주체(실제로 가격을 움직이는 체결)를 가장 직접적으로 보여줌.
OBV가 '종가 방향'으로 거래량을 가감하는 것보다 한 단계 더 정확함.
  가격 HH + CVD LH = 하락 다이버전스 (시장가 매수 약화) → 상승 추진력 의심
  가격 LL + CVD HL = 상승 다이버전스 (시장가 매도 약화) → 반등 가능

★ 핵심 활용 (다이버전스 진위 판별):
  • RSI 상승 다이버전스가 떠도 CVD가 저점 갱신(LL)이면 → 다이버전스 실패 확률 높음 (숏 익절성 반등일 뿐)
  • RSI 하락 다이버전스가 떠도 CVD가 고점 갱신(HH)이면 → 숏 다이버전스 실패 확률 높음
  • 가격 저점 갱신 시 CVD가 같이 깨지면(하락 확인) vs 버티면(반등 후보) — 이 차이가 핵심
  • CVD 현재 추세(매수우위↑/매도우위↓)가 가격 방향과 일치하는지 항상 확인

━━━ OBV 다이버전스 ━━━
OBV는 매수/매도 방향이 포함되어 raw volume보다 다이버전스 판단이 정확.
  가격 HH + OBV LH = 하락 다이버전스 (매수 주도 약화)
  가격 LL + OBV HL = 상승 다이버전스 (매도 주도 약화)

다이버전스 우선순위: CVD > OBV > RSI > raw volume
실제 시장가 체결(CVD)이 가장 신뢰도 높음.
동일 방향 신호가 겹치면 확신도 상승, 충돌하면 CVD → OBV 순으로 우선.

━━━ 시장 레짐 ━━━
• UP_TREND: 강한 상승 (ADX≥20, EMA 정배열, +DI 우위)
• UP_BIAS: 약한 상승
• RANGE: 횡보 (ADX<18, BB 수축)
• DOWN_TREND: 강한 하락 (ADX≥20, EMA 역배열, -DI 우위)
• DOWN_BIAS: 약한 하락
• MIXED: 혼조

━━━ 미결제약정(OI) — 신규 진입 vs 청산 구분 ━━━
OI 변화는 그 가격 움직임이 신규 포지션 유입인지 기존 포지션 청산인지를 알려줌.
RSI/CVD가 못 보는 차원이므로 방향 신뢰도 판정에 필수.
  가격↑ + OI↑ = 신규 롱 유입 → 상승 추세 건강, 롱 신뢰
  가격↑ + OI↓ = 숏커버 반등 → 숏 청산이 끌어올린 것, 상승 지속력 의심
  가격↓ + OI↑ = 신규 숏 유입 → 하락 추세 건강, 숏 신뢰
  가격↓ + OI↓ = 롱 청산/정리 → 투매 마무리, 반등 가능성 증가
RSI 결합 예:
  • RSI 과매도 + 상승 다이버전스 + OI 감소(롱 청산 마무리) → 반등 신뢰 ↑
  • RSI 과매도 + 상승 다이버전스 + OI 증가(신규 숏 유입) → 다이버전스 실패 가능 ↑

━━━ 펀딩비 / 프리미엄 — 포지션 쏠림 & 스퀴즈 위험 ━━━
펀딩 기준선 0.01%(중립). 마이너스 = 숏 과열, 강한 양수 = 롱 과열.
  강한 음수 펀딩 = 숏 쏠림 → 가격 조금만 올라도 숏커버 급등(숏스퀴즈) 위험
  강한 양수 펀딩 = 롱 쏠림 → 롱스퀴즈(급락) 위험
  중립 펀딩 = 포지션 쏠림 없음, 추세 지속 여지
활용:
  • 숏 과열일 땐 추격 숏 자제 (숏커버 반등 리스크), 과매도 반등 롱은 가점
  • 롱 과열일 땐 추격 롱 자제 (롱스퀴즈 리스크)
  • 마크-인덱스 프리미엄이 음전이면 선물이 현물보다 약세 — 하락 우위 보강

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

━━━ 리포트 출력 형식 (의사결정 리포트 — 반드시 이 구조 그대로) ━━━
이 리포트는 '분석'이 아니라 '매매 의사결정' 도구다. 방향이 맞아도 지금 들어가면
안 되는 경우(추격 위험)를 분명히 구분하라. 위에 제공된 '🧭 기계 판정' 수치
(방향성/진입적합도/손익비/진입금지)를 결론·판정·손익비에 반드시 반영하라.
각 섹션은 짧게 — 불릿·표 위주, 장문 서술 금지.

## 🎯 한 줄 결론
한 문장: 방향 + 즉시 진입 여부 + 핵심 단서.
예) "메인 방향은 숏이나 현재가는 저점 추격 구간 — 신규 진입은 1,747~1,761 반등 실패 확인 후만 유효."

## 📋 매매 판정
| 항목 | 값 |
|------|-----|
| 방향성 | 숏/롱/중립 (+점수) |
| 즉시 진입 | 가능 / 대기 / 금지 |
| 적합 전략 | 추격 / 반등실패 / 눌림 / 리테스트 / 관망 / 익절우선 |
| 권장 사이즈 | 정상 / 축소 / 최소 / 진입금지 |
| 리스크 | 낮음 / 보통 / 높음 |

## 💰 핵심 가격대 & 손익비
- **R(손익비) 설명을 첫 줄에 한 줄로**: "R = 목표까지 거리 ÷ 손절까지 거리. 1R=본전, 2R↑ 양호, 1R 미만 진입 부적합."
- 그 아래에 제공된 '📐 진입 시나리오 손익비' 표를 그대로 옮겨라(진입가·손절·목표·R). **R은 직접 계산하지 말고 제공된 값 사용.**
- 핵심 가격대(현재가/진입후보/손절/관점무효화/1~3차 목표)는 '📍 핵심 레벨 맵'의 값만 사용.

## 🅰️ 메인 / 🅱️ 반대 시나리오
- 🅰️ 메인(우세 방향): 조건 → 행동 → 목표
- 🅱️ 반대(관점 깨짐): 무효화 가격 → 그 후 행동

## ⚔️ 신호 충돌 해석
상위 레짐 vs 하위 다이버전스 등 충돌을 1~2줄로 판정 (롱 전환인지, 단순 '추격 경고'인지 명확히).

## 📊 타임프레임 요약
| TF | 방향 | 레짐 | 핵심(5단어 이내) |

## 💡 진입 전략 (관점별)
- 스캘핑(1~5분) / 데이(15분~1시간) / 스윙(4시간~일봉) 각 한 줄: 조건·진입·목표·손절.

## ✂️ 손절 & 익절
- 손절(포지션 종료가) ≠ 관점 무효화(분석이 틀린 조건) 를 구분.
- 익절은 부분익절 기준 포함 (1차 도달 시 일부, 리테스트 실패 시 보유, 반대 다이버전스 시 축소).

## 🚫 진입 금지 (지금 하면 안 되는 매매)
- 기계 판정의 진입금지 조건 + 추가 위험. 예: 저점 추격 숏, 상위 하락 레짐 무시한 롱, 1R 미만 진입.

## ✅ 최종 행동 지침 (3줄)
- 지금 무엇을 / 어떤 조건에서 진입 / 어떤 조건에서 관점 약화.

━━━ 작성 규칙 ━━━
• ⚠️ 가격·손익비는 반드시 제공된 '📍 핵심 레벨 맵'과 '📐 진입 시나리오 손익비'의 값만 사용.
  데이터에 없는 가격을 새로 만들거나 추정하지 마라. R은 직접 암산하지 마라.
• 가격 범위 표기는 물결(~)이 아니라 en-dash(–)나 화살표(→)로. 예: "1,739–1,743" (1,739~1,743 금지)
• 방향성 점수와 즉시 진입 점수를 분리 — 방향이 맞아도 추격 불리하면 "방향 O, 즉시진입 X" 명시
• 손절(포지션) ≠ 관점 무효화(분석) 항상 구분
• 손익비 1R 미만 진입은 '진입 금지'로 분류
• 표·불릿 위주, 장문 서술/반복 금지, 마크다운 취소선(~~) 금지
• 다이버전스는 상태(후보/확정/실패)까지 명시
• 레짐별 목표가 원칙(반등 목표 상한 엄수): 하락장 롱목표=EMA20/VWAP까지만(그 이상 과매수 목표 금지) · 횡보=BB상단/RSI70 · 상승=전고
• CVD가 가격과 어긋나거나, 펀딩 과열이거나, 경계선 RSI면 → '진입 금지' 또는 '리스크'에 반영

━━━ 대화 연속성 (앞 대화가 있을 때) ━━━
• 이 분석은 하나의 대화 방의 연장이다. 앞선 메시지(이전 RSI 분석·채팅·사용자가 밝힌 포지션/관점)가 있으면 반드시 인지하고 이어가라.
• 맨 첫 줄에 그 맥락을 한 줄로 연결하라. 예: "앞서 1,725 숏 보유·추가 숏 타이밍을 논의하셨죠 — 현재 RSI 데이터 기준:"
• 방향 판정은 데이터 기준으로 객관 유지한다. 사용자가 논의한 방향과 데이터 판정이 다르면, 동조하지 말고 그 차이를 분명히 짚어라. 예: "다만 데이터는 롱 우위라, 추가 숏은 신중해야 합니다."
• 앞서 제시한 레벨/시나리오가 있으면 그것과 일관되게 갱신·연결하라(바뀌었으면 무엇이 왜 바뀌었는지 한 줄).

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


def detect_cvd_divergence(closes, cvd_series, lookback=30):
    """CVD 다이버전스 감지 — 실제 시장가 체결 기반 (OBV/raw volume보다 직접적)

    CVD는 시장가 매수-매도의 누적이라 공격적 주문 주체를 가장 직접적으로 보여줌.
      가격 HH + CVD LH = 하락 다이버전스 (시장가 매수 약화)
      가격 LL + CVD HL = 상승 다이버전스 (시장가 매도 약화)

    Returns:
        dict | None: {type, label, detail, price/cvd 값들, bias}
    """
    if len(closes) < lookback or len(cvd_series) < lookback:
        return None

    price_peaks, price_troughs = _find_local_extremes(closes, lookback)
    recent_cvd = cvd_series[-lookback:] if len(cvd_series) >= lookback else list(cvd_series)

    def _cvd_at(idx):
        if idx < len(recent_cvd):
            return recent_cvd[idx]
        return recent_cvd[-1]

    # ═══ 하락 다이버전스: 가격 HH + CVD LH ═══
    if len(price_peaks) >= 2:
        p1_idx, p1_val = price_peaks[-2]
        p2_idx, p2_val = price_peaks[-1]
        c1 = _cvd_at(p1_idx)
        c2 = _cvd_at(p2_idx)

        if p2_val > p1_val and c2 < c1:
            return {
                "type": "CVD_BEAR_DIV",
                "label": "🔻 CVD 하락 다이버전스",
                "detail": f"가격 HH({p1_val:.1f}→{p2_val:.1f}) + CVD LH — 시장가 매수 약화",
                "bias": "BEARISH",
                "price_1": p1_val, "price_2": p2_val,
                "cvd_1": round(c1, 0), "cvd_2": round(c2, 0),
            }

    # ═══ 상승 다이버전스: 가격 LL + CVD HL ═══
    if len(price_troughs) >= 2:
        t1_idx, t1_val = price_troughs[-2]
        t2_idx, t2_val = price_troughs[-1]
        c1 = _cvd_at(t1_idx)
        c2 = _cvd_at(t2_idx)

        if t2_val < t1_val and c2 > c1:
            return {
                "type": "CVD_BULL_DIV",
                "label": "🔺 CVD 상승 다이버전스",
                "detail": f"가격 LL({t1_val:.1f}→{t2_val:.1f}) + CVD HL — 시장가 매도 약화",
                "bias": "BULLISH",
                "price_1": t1_val, "price_2": t2_val,
                "cvd_1": round(c1, 0), "cvd_2": round(c2, 0),
            }

    return None


def synthesize_divergence(rsi_div, vol_div, obv_div, cvd_div=None):
    """RSI + 거래량 + OBV + CVD 다이버전스 종합 판정

    우선순위: CVD > OBV > RSI > raw volume
    (CVD = 실제 시장가 체결, 가장 직접적 → 최우선)
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

    # CVD 다이버전스 분류 (우선순위 최고 — 실제 시장가 체결)
    if cvd_div:
        cvd_bias = cvd_div.get("bias", "NEUTRAL")
        if cvd_bias == "BULLISH":
            bullish_signals.append(("CVD", cvd_div["label"], 13))
        elif cvd_bias == "BEARISH":
            bearish_signals.append(("CVD", cvd_div["label"], 13))
        all_signals.append(f"CVD: {cvd_div['label']}")

    # OBV 다이버전스 분류 (우선순위 높음)
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
        "cvd_div": cvd_div,
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
# OI / 펀딩 분석
# ═══════════════════════════════════════════════

def analyze_oi_change(closes, oi_series, lookback=14):
    """가격-OI 사분면 분류 — 그 움직임이 신규 진입인지 청산인지 구분

    가격↑ + OI↑ = 신규 롱 유입 (상승 신뢰)
    가격↑ + OI↓ = 숏커버 반등 (지속력 의심)
    가격↓ + OI↑ = 신규 숏 유입 (하락 신뢰)
    가격↓ + OI↓ = 롱 청산/정리 (투매 후 반등 가능)

    Args:
        closes: 종가 시계열 (klines)
        oi_series: fetch_open_interest_hist 결과 (같은 period)
        lookback: 변화 측정 봉 수

    Returns:
        dict | None: {quadrant, label, detail, bias, oi_change_pct, price_change_pct}
    """
    if not oi_series or len(oi_series) < lookback + 1 or len(closes) < lookback + 1:
        return None

    oi_vals = [p["oi"] for p in oi_series]
    oi_now = oi_vals[-1]
    oi_then = oi_vals[-1 - lookback]
    price_now = closes[-1]
    price_then = closes[-1 - lookback]

    if oi_then == 0 or price_then == 0:
        return None

    oi_chg = (oi_now - oi_then) / oi_then * 100
    price_chg = (price_now - price_then) / price_then * 100

    price_up = price_chg > 0
    oi_up = oi_chg > 0

    if price_up and oi_up:
        quadrant, label, bias = "NEW_LONG", "신규 롱 유입", "BULLISH"
        detail = "가격↑ + OI↑ — 신규 매수 진입, 상승 추세 건강"
    elif price_up and not oi_up:
        quadrant, label, bias = "SHORT_COVER", "숏커버 반등", "WEAK_BULLISH"
        detail = "가격↑ + OI↓ — 숏 청산 반등, 상승 지속력 의심"
    elif not price_up and oi_up:
        quadrant, label, bias = "NEW_SHORT", "신규 숏 유입", "BEARISH"
        detail = "가격↓ + OI↑ — 신규 매도 진입, 하락 추세 건강"
    else:
        quadrant, label, bias = "LONG_LIQ", "롱 청산/정리", "WEAK_BULLISH"
        detail = "가격↓ + OI↓ — 롱 청산 마무리, 투매 후 반등 가능"

    return {
        "quadrant": quadrant,
        "label": label,
        "detail": detail,
        "bias": bias,
        "oi_change_pct": round(oi_chg, 2),
        "price_change_pct": round(price_chg, 2),
    }


def analyze_funding(funding_info):
    """펀딩비/프리미엄 해석 — 포지션 쏠림 & 스퀴즈 위험

    펀딩 기준선 0.01%(중립). 강한 양수 = 롱 과열(롱스퀴즈 위험),
    마이너스 = 숏 과열(숏스퀴즈 위험).

    Returns:
        dict | None: {bias, squeeze_risk, label, detail, funding_pct, premium_pct}
    """
    if not funding_info:
        return None

    fr = funding_info.get("funding_pct", 0)
    premium = funding_info.get("premium_pct", 0)

    if fr >= 0.05:
        bias, squeeze_risk, label = "LONG_CROWDED", "LONG_SQUEEZE", "🔴 롱 과열"
        detail = f"펀딩 {fr:+.4f}% — 롱 쏠림, 롱스퀴즈(급락) 위험"
    elif fr >= 0.02:
        bias, squeeze_risk, label = "LONG_BIAS", None, "🟠 약한 롱 편향"
        detail = f"펀딩 {fr:+.4f}% — 롱 약우위"
    elif fr <= -0.05:
        bias, squeeze_risk, label = "SHORT_CROWDED", "SHORT_SQUEEZE", "🟢 숏 과열"
        detail = f"펀딩 {fr:+.4f}% — 숏 쏠림, 숏스퀴즈(급등) 위험"
    elif fr <= -0.01:
        bias, squeeze_risk, label = "SHORT_BIAS", None, "🟡 약한 숏 편향"
        detail = f"펀딩 {fr:+.4f}% — 숏 약우위"
    else:
        bias, squeeze_risk, label = "NEUTRAL", None, "⚪ 중립"
        detail = f"펀딩 {fr:+.4f}% — 포지션 쏠림 없음"

    if premium:
        detail += f" | 프리미엄 {premium:+.4f}%"

    return {
        "bias": bias,
        "squeeze_risk": squeeze_risk,
        "label": label,
        "detail": detail,
        "funding_pct": fr,
        "premium_pct": premium,
    }


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
    oi_an = r.get("oi_analysis")
    fund = r.get("funding_analysis")

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

    # 16) OI 변화 — 신규 진입 vs 청산 (NEW)
    if oi_an:
        q = oi_an["quadrant"]
        if q == "NEW_SHORT":          # 가격↓+OI↑ 신규 숏 → 하락 신뢰
            short_score += 12
        elif q == "NEW_LONG":         # 가격↑+OI↑ 신규 롱 → 상승 신뢰
            long_score += 12
        elif q == "LONG_LIQ":         # 가격↓+OI↓ 롱 청산 마무리 → 반등 가능
            long_score += 6
            short_score -= 5          # 추격 숏 신뢰 약화
        elif q == "SHORT_COVER":      # 가격↑+OI↓ 숏커버 → 지속력 약함
            long_score += 2

    # 17) 펀딩 스퀴즈 리스크 (NEW)
    if fund:
        if fund["squeeze_risk"] == "SHORT_SQUEEZE":   # 숏 과열 → 숏커버 급등 위험
            short_score -= 10
            long_score += 6
        elif fund["squeeze_risk"] == "LONG_SQUEEZE":  # 롱 과열 → 롱스퀴즈 급락 위험
            long_score -= 10
            short_score += 6
        elif fund["bias"] == "SHORT_BIAS":
            short_score += 2
        elif fund["bias"] == "LONG_BIAS":
            long_score += 2

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

    # ── 데이터 프리페치 (병렬) — 펀딩 1회 + TF별 klines/OI 최대 13개 요청을
    # 동시에 받아 총 수집 시간을 '가장 느린 요청 1개' 수준으로 단축 ──
    with ThreadPoolExecutor(max_workers=13) as ex:
        funding_fut = ex.submit(fetch_funding_premium, symbol)
        kline_futs = {tf: ex.submit(fetch_klines, symbol, INTERVAL_MAP[tf], KLINE_WARMUP)
                      for tf in WAVE_TIMEFRAMES if INTERVAL_MAP.get(tf)}
        # fetch_open_interest_hist 는 실패 시 내부에서 None 반환
        oi_futs = {tf: ex.submit(fetch_open_interest_hist, symbol, period)
                   for tf, period in TF_OI_PERIOD.items()}

        funding_info = funding_fut.result()
        kline_data, kline_err = {}, {}
        for tf, fut in kline_futs.items():
            try:
                kline_data[tf] = fut.result()
            except Exception as e:
                kline_err[tf] = e
        oi_data = {tf: fut.result() for tf, fut in oi_futs.items()}

    funding_analysis = analyze_funding(funding_info)

    for tf_label in WAVE_TIMEFRAMES:
        bi = INTERVAL_MAP.get(tf_label)
        if not bi:
            continue

        try:
            if tf_label in kline_err:
                raise kline_err[tf_label]
            candles = kline_data[tf_label]
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

            # ── CVD (시계열 포함) ──
            cvd, cvd_ema, cvd_series = calc_cvd(candles, return_series=True)

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

            # ── CVD 다이버전스 (NEW) ──
            cvd_div = detect_cvd_divergence(closes, cvd_series) if cvd_series else None

            # ── OI 변화 (NEW — 프리페치된 매칭 period 데이터 사용) ──
            oi_series = oi_data.get(tf_label)
            oi_analysis = analyze_oi_change(closes, oi_series) if oi_series else None

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

            # ── 최근 스윙 고저 (실제 S/R 레벨) ──
            recent_high = max(c["high"] for c in candles[-30:])
            recent_low = min(c["low"] for c in candles[-30:])

            results[tf_label] = {
                # 기존 필드 (호환 유지)
                "price": cur,
                "recent_high": round(recent_high, 2),
                "recent_low": round(recent_low, 2),
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
                "cvd": cvd,
                "cvd_ema": cvd_ema,
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
                # ── v3 거래량/OBV/CVD 다이버전스 ──
                "vol_div": vol_div,
                "obv_div": obv_div,
                "cvd_div": cvd_div,
                "synth_div": synthesize_divergence(div_v2, vol_div, obv_div, cvd_div),
                # ── v4 OI / 펀딩 ──
                "oi_analysis": oi_analysis,
                "funding_analysis": funding_analysis,
                "funding_info": funding_info,
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
# 결정 레이어 — 진입 평가 / 레벨 맵 / 시나리오
# (렌더링 함수는 core/rsi_render.py 로 이동)
# ═══════════════════════════════════════════════


def assess_entry(results, ref_tf_order=("1시간", "15분", "4시간", "5분", "1일")):
    """방향성과 '즉시 진입 적합도'를 분리 평가 (결정 레이어).

    방향이 숏이어도 현재가에서 추격이 불리할 수 있음 → 별도 점수로 분리.
    진입 금지 조건, 권장 포지션 사이즈, 리스크 등급, 현재가 추격 손익비(R)를 계산.

    Returns:
        dict | None: {
            direction, direction_score, align,           # 방향성
            entry_score, chase_ok, position_size, risk_level,  # 진입 적합도
            blocks: [str],                               # 진입 금지/주의 조건
            ref_tf, levels: {price, target, invalidation, support, resistance},
            rr,                                          # 현재가 추격 손익비
        }
    """
    valid = {tf: r for tf, r in results.items() if r and not r.get("error")}
    if not valid:
        return None

    # ── 1) 방향 집계 (TF·확신 가중) ──
    long_w = short_w = 0.0
    for tf, r in valid.items():
        w = DIR_TF_WEIGHT.get(tf, 1.0) * CONF_WEIGHT.get(r.get("confidence", ""), 0)
        if r.get("position") == "숏":
            short_w += w
        elif r.get("position") == "롱":
            long_w += w

    if short_w > long_w and short_w - long_w > 0.5:
        direction = "숏"
    elif long_w > short_w and long_w - short_w > 0.5:
        direction = "롱"
    else:
        direction = "중립"

    total_w = sum(DIR_TF_WEIGHT.get(tf, 1.0) for tf in valid)
    dom_w = short_w if direction == "숏" else long_w if direction == "롱" else 0.0
    direction_score = round(min(100, dom_w / total_w * 100)) if total_w else 0
    n_align = sum(1 for r in valid.values() if r.get("position") == direction)
    align = f"{n_align}/{len(valid)}"

    # ── 2) 기준 TF ──
    ref_tf = next((tf for tf in ref_tf_order if tf in valid), None)
    if direction == "중립" or ref_tf is None:
        return {
            "direction": direction, "direction_score": direction_score, "align": align,
            "entry_score": 0, "chase_ok": False, "position_size": "관망",
            "risk_level": "보통", "blocks": ["방향성 불명확 — 관망"],
            "ref_tf": ref_tf, "levels": {}, "rr": None,
        }

    r = valid[ref_tf]
    price = r.get("price", 0)
    blocks = []
    entry_score = direction_score

    # ── 3) 즉시 진입 감점 + 진입 금지 조건 ──
    if direction == "숏":
        if r.get("bb_lower") and price <= r["bb_lower"] * 1.003:
            entry_score -= 20
            blocks.append(f"{ref_tf} BB하단 근접 — 추격 숏 위험")
        for htf in ("1일", "4시간"):
            rr = valid.get(htf)
            if rr and rr.get("rsi", 50) <= 20:
                entry_score -= 15
                blocks.append(f"{htf} RSI 극과매도({rr['rsi']:.0f}) — 반등 리스크")
                break
        for tf2, rr in valid.items():
            sd = rr.get("synth_div") or {}
            if sd.get("overall_bias") == "BULLISH" and sd.get("confidence") in ("HIGH", "MEDIUM"):
                entry_score -= 15
                blocks.append(f"{tf2} 상승 다이버전스({sd['confidence']}) 생존 — 반등 리스크")
                break
        h4 = valid.get("4시간")
        if h4 and (h4.get("vol_pattern") or {}).get("pattern") == "ABSORPTION":
            entry_score -= 10
            blocks.append("4시간 거래량 흡수 — 저점 추격 숏 감점")
    else:  # 롱
        if r.get("bb_upper") and price >= r["bb_upper"] * 0.997:
            entry_score -= 20
            blocks.append(f"{ref_tf} BB상단 근접 — 추격 롱 위험")
        for htf in ("1일", "4시간"):
            rr = valid.get(htf)
            if rr and rr.get("rsi", 50) >= 80:
                entry_score -= 15
                blocks.append(f"{htf} RSI 극과매수({rr['rsi']:.0f}) — 조정 리스크")
                break
        for tf2, rr in valid.items():
            sd = rr.get("synth_div") or {}
            if sd.get("overall_bias") == "BEARISH" and sd.get("confidence") in ("HIGH", "MEDIUM"):
                entry_score -= 15
                blocks.append(f"{tf2} 하락 다이버전스({sd['confidence']}) 생존 — 조정 리스크")
                break
        h4 = valid.get("4시간")
        if h4 and (h4.get("vol_pattern") or {}).get("pattern") == "CONTINUATION":
            entry_score -= 10
            blocks.append("4시간 하락지속형 거래량 — 고점 추격 롱 감점")

    # ── 4) 현재가 추격 손익비(R) — 레벨 맵 기반 (진입 시나리오 표와 동일 기준가·공식) ──
    lm = build_level_map(results)
    atr = r.get("atr") or (price * 0.005)
    rp = lm["ref_price"] if lm else price  # 시나리오 표와 동일 기준가 사용
    levels = {"price": round(rp, 2)}
    rr_val = None
    if lm:
        stop, target, rr_val = _scenario_rr(direction, rp, lm, atr, rp)
        levels["resistance" if direction == "숏" else "support"] = stop
        levels["target"] = target
    if rr_val is not None and rr_val < 1.0:
        entry_score -= 25
        blocks.append(f"현재가 추격 손익비 {rr_val}R < 1 — 진입 부적합")

    # ── 5) 포지션 사이즈 + 리스크 등급 ──
    entry_score = max(0, min(100, entry_score))
    n_blocks = len(blocks)
    if entry_score >= 65 and n_blocks == 0:
        position_size, chase_ok = "정상", True
    elif entry_score >= 45:
        position_size, chase_ok = "축소(50~70%)", False
    elif entry_score >= 25:
        position_size, chase_ok = "최소(30%↓)", False
    else:
        position_size, chase_ok = "진입 금지(대기)", False

    atr_pct = r.get("atr_pct") or 0
    if atr_pct >= 1.5 or n_blocks >= 2:
        risk_level = "높음"
    elif atr_pct >= 0.8 or n_blocks == 1:
        risk_level = "보통"
    else:
        risk_level = "낮음"

    return {
        "direction": direction,
        "direction_score": direction_score,
        "align": align,
        "entry_score": entry_score,
        "chase_ok": chase_ok,
        "position_size": position_size,
        "risk_level": risk_level,
        "blocks": blocks,
        "ref_tf": ref_tf,
        "levels": levels,
        "rr": rr_val,
    }


def build_level_map(results, level_tfs=("5분", "15분", "1시간", "4시간", "1일", "1주"), tol_pct=0.15):
    """여러 TF의 실제 레벨(EMA/VWAP/BB/스윙고저/목표)을 모아 클러스터링.

    AI가 가격을 지어내지 않도록, 인용 가능한 '실제 레벨 메뉴'를 제공한다.
    가까운 레벨끼리 묶어 컨플루언스(n≥3)를 표시.

    Returns:
        dict | None: {ref_price, above: [clusters], below: [clusters]}
                     cluster = {price, labels: [str], n}
    """
    valid = {tf: r for tf, r in results.items() if r and not r.get("error")}
    if not valid:
        return None
    ref = next((valid[tf] for tf in ("15분", "1시간", "5분", "4시간", "1일") if tf in valid), None)
    if ref is None:
        return None
    price = ref["price"]

    raw = []
    for tf in level_tfs:
        r = valid.get(tf)
        if not r:
            continue
        s = TF_LABELS_SHORT.get(tf, tf)
        for key, name in (("ema20", "EMA20"), ("ema50", "EMA50"), ("vwap", "VWAP"),
                          ("bb_upper", "BB상"), ("bb_lower", "BB하"),
                          ("recent_high", "최근고"), ("recent_low", "최근저")):
            v = r.get(key)
            if v and v > 0:
                raw.append((round(float(v), 1), f"{s}{name}"))
        for side in ("long", "short"):
            for t in (r.get("targets") or {}).get(side, []):
                raw.append((round(float(t[1]), 1), f"{s}{t[2]}"))

    if not raw:
        return None

    raw.sort()
    tol = price * tol_pct / 100
    clusters = []
    for p, lbl in raw:
        if clusters and abs(p - clusters[-1]["price"]) <= tol:
            c = clusters[-1]
            c["price"] = (c["price"] * c["n"] + p) / (c["n"] + 1)
            c["n"] += 1
            if lbl not in c["labels"]:
                c["labels"].append(lbl)
        else:
            clusters.append({"price": p, "labels": [lbl], "n": 1})
    for c in clusters:
        c["price"] = round(c["price"], 1)

    above = sorted([c for c in clusters if c["price"] > price], key=lambda c: c["price"])
    below = sorted([c for c in clusters if c["price"] < price], key=lambda c: -c["price"])
    return {"ref_price": round(price, 1), "above": above, "below": below}


def _rr_grade(R):
    if R is None:
        return "산출불가"
    return "양호" if R >= 2 else "보통" if R >= 1 else "부적합"


def _scenario_rr(direction, entry, lm, atr, price):
    """진입가 1개에 대한 (손절, 목표, 손익비R) 계산 — 단일 공식(헤드라인·표 공용).

    손절: 진입에서 0.5×ATR 이상 떨어진 가장 가까운 반대 레벨 (없으면 1.2×ATR)
    목표: 현재가 기준 0.3% 이상 떨어진 가장 가까운 1차 지지/저항
    """
    above, below = lm["above"], lm["below"]
    min_gap = atr * 0.5
    tgt_gap = price * 0.003

    if direction == "숏":
        below_m = [c for c in below if (price - c["price"]) >= tgt_gap] or below
        target = below_m[0]["price"] if below_m else None
        stop = next((c["price"] for c in above if c["price"] >= entry + min_gap),
                    round(entry + 1.2 * atr, 1))
    else:
        above_m = [c for c in above if (c["price"] - price) >= tgt_gap] or above
        target = above_m[0]["price"] if above_m else None
        stop = next((c["price"] for c in below if c["price"] <= entry - min_gap),
                    round(entry - 1.2 * atr, 1))

    if not target or not stop:
        return (round(stop, 1) if stop else None, round(target, 1) if target else None, None)
    risk = abs(stop - entry)
    reward = abs(entry - target)
    R = round(reward / risk, 2) if risk > 0 else None
    return (round(stop, 1), round(target, 1), R)


def build_entry_scenarios(results):
    """진입가별 손익비(R)를 파이썬이 직접 계산 (AI 암산 금지).

    손절은 'ATR의 0.5배 이상 떨어진 가장 가까운 반대 레벨', 목표는 현재가
    기준 1차 지지/저항으로 통일. 높은 데서 진입할수록 R이 좋아지는 구조를 보여줌.

    Returns:
        dict | None: {direction, price, scenarios: [{label, entry, stop, target, R, grade}]}
    """
    ea = assess_entry(results)
    if not ea or ea["direction"] == "중립":
        return None
    lm = build_level_map(results)
    if not lm:
        return None

    price = lm["ref_price"]
    direction = ea["direction"]
    rref = results.get(ea.get("ref_tf")) or {}
    atr = rref.get("atr") or (price * 0.005)
    scenarios = []

    def _mk(label, entry):
        stop, target, R = _scenario_rr(direction, entry, lm, atr, price)
        if not target:
            return None
        return {"label": label, "entry": round(entry, 1), "stop": stop,
                "target": target, "R": R, "grade": _rr_grade(R)}

    # 현재가 추격
    s = _mk("현재가 추격", price)
    if s:
        scenarios.append(s)

    # 반등/눌림 진입 후보 — 현재가에서 0.3% 이상 떨어진 레벨만 (붙은 레벨 제외)
    if direction == "숏":
        cands = [c for c in lm["above"] if (c["price"] - price) >= price * 0.003][:3]
        verb = "반등실패"
    else:
        cands = [c for c in lm["below"] if (price - c["price"]) >= price * 0.003][:3]
        verb = "눌림"
    for c in cands:
        s = _mk(f"{'·'.join(c['labels'][:2])} {verb}", c["price"])
        if s:
            scenarios.append(s)

    seen, uniq = set(), []
    for s in scenarios:
        if s["entry"] in seen:
            continue
        seen.add(s["entry"])
        uniq.append(s)
    return {"direction": direction, "price": price, "scenarios": uniq[:4]}


def _format_level_map(lm, max_each=5):
    """레벨 맵 → AI/요약용 텍스트"""
    price = lm["ref_price"]

    def fmt(clusters):
        rows = []
        for c in clusters[:max_each]:
            pct = (c["price"] - price) / price * 100 if price else 0
            conf = " ⭐컨플루언스" if c["n"] >= 3 else ""
            rows.append(f"  {c['price']} ({'+' if pct >= 0 else ''}{pct:.1f}%) [{', '.join(c['labels'][:3])}]{conf}")
        return "\n".join(rows) if rows else "  (없음)"

    return f"현재가 {price}\n저항(위):\n{fmt(lm['above'])}\n지지·목표(아래):\n{fmt(lm['below'])}"


def _format_scenarios_md(es):
    """진입 시나리오 → 마크다운 표"""
    rows = ["| 진입 시나리오 | 진입가 | 손절 | 목표 | 손익비 |",
            "|---|---|---|---|---|"]
    for s in es["scenarios"]:
        rr = f"{s['R']}R" if s["R"] is not None else "-"
        rows.append(f"| {s['label']} | {s['entry']} | {s['stop']} | {s['target']} | {rr}({s['grade']}) |")
    return "\n".join(rows)


def generate_summary_text(results):
    """RSI 파동 종합 판정 — 간결 스캔형 (관점별 포지션 + 레짐 + 핵심 경고)"""
    lines = ["### 📊 종합 판정"]

    # ── 🧭 결정 판정 (방향 ≠ 즉시진입) ──
    ea = assess_entry(results)
    es = build_entry_scenarios(results)
    if ea:
        dir_icon = "🔴" if ea["direction"] == "숏" else "🟢" if ea["direction"] == "롱" else "⚪"
        if ea["chase_ok"]:
            entry_icon, entry_txt = "✅", "즉시진입 가능"
        elif ea["position_size"].startswith("진입 금지"):
            entry_icon, entry_txt = "🚫", "추격 금지·반등 실패 대기"
        else:
            entry_icon, entry_txt = "⚠️", "추격 자제·대기"

        # 현재가 추격 손익비 — 시나리오(파이썬 계산)에서 가져옴
        chase_R = None
        if es:
            chase = next((s for s in es["scenarios"] if s["label"] == "현재가 추격"), None)
            chase_R = chase["R"] if chase else None
        rr_txt = f" · 추격 손익비 {chase_R}R" if chase_R is not None else ""

        def _bar(v, n=10):
            fill = max(0, min(n, round(v / 100 * n)))
            return "█" * fill + "░" * (n - fill)

        lines.append(
            f"> 🧭 **{dir_icon} {ea['direction']} 우위** (정렬 {ea['align']}) "
            f"· {entry_icon} **{entry_txt}**{rr_txt}"
        )
        lines.append(
            f"> 방향성 `{_bar(ea['direction_score'])}` {ea['direction_score']} · "
            f"즉시진입 `{_bar(ea['entry_score'])}` {ea['entry_score']}"
        )
        lines.append(
            f"> 권장 포지션 **{ea['position_size']}** · 리스크 **{ea['risk_level']}**"
        )
        if ea["blocks"]:
            lines.append(f"> ⛔ 진입 주의: {' / '.join(ea['blocks'][:3])}")
        lines.append("")

        # ── 💰 진입가별 손익비 표 (파이썬 계산) ──
        if es and es["scenarios"]:
            lines.append("**💰 진입가별 손익비** (R = 목표거리÷손절거리, 1R 미만은 진입 부적합)")
            lines.append(_format_scenarios_md(es))
            lines.append("")

    # ── 관점별 포지션 (한 줄씩) ──
    groups = {
        "스캘핑": ["1분", "5분"],
        "데이": ["15분", "1시간"],
        "스윙": ["4시간", "1일"],
        "장기": ["1주"],
    }
    for group_name, tfs in groups.items():
        parts = []
        for tf in tfs:
            r = results.get(tf)
            if r and not r.get("error"):
                pos = r.get("position", "")
                conf = r.get("confidence", "")
                icon = "🟢" if pos == "롱" else "🔴" if pos == "숏" else "⚪"
                parts.append(f"{tf} {icon}{pos}:{conf}")
        lines.append(f"- **{group_name}**: {' · '.join(parts) if parts else '데이터 없음'}")

    # ── 레짐 현황 (한 줄) ──
    regime_summary = []
    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if r and not r.get("error") and r.get("regime"):
            rl = REGIME_LABELS.get(r["regime"], r["regime"])
            rc = "🟢" if "UP" in r["regime"] else "🔴" if "DOWN" in r["regime"] else "🟡"
            regime_summary.append(f"{TF_LABELS_SHORT.get(tf, tf)} {rc}{rl}")
    if regime_summary:
        lines.append(f"- **레짐**: {' · '.join(regime_summary)}")

    # ── 펀딩 (심볼 단위, 1회) ──
    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if r and not r.get("error") and r.get("funding_analysis"):
            f = r["funding_analysis"]
            lines.append(f"- **펀딩**: {f['label']} ({f['detail']})")
            break

    # ── 핵심 경고 (같은 유형은 TF 묶고, 우선순위 정렬 후 최대 5개) ──
    squeeze_alerts = []     # (tf, dir, core_met)
    bear_flag_tfs = []
    failed_div_tfs = []
    synth_alerts = []       # (tf, summary)

    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if not r or r.get("error"):
            continue
        sq = r.get("squeeze_expansion")
        if sq:
            sq_dir = "하방" if sq["type"] == "BEARISH_EXPANSION" else "상방"
            squeeze_alerts.append((tf, sq_dir, sq.get("core_met", "?")))
        if r.get("failed_div"):
            failed_div_tfs.append(tf)
        if r.get("bear_flag"):
            bear_flag_tfs.append(tf)
        synth = r.get("synth_div")
        if synth and synth.get("overall_bias") != "NEUTRAL" and synth.get("confidence") in ("HIGH", "MEDIUM"):
            synth_alerts.append((tf, synth["summary"]))

    alerts = []  # (priority, text) — priority 작을수록 먼저
    for tf, sq_dir, core in squeeze_alerts:
        icon = "🔴💥" if sq_dir == "하방" else "🟢💥"
        alerts.append((0, f"{icon} {tf} {sq_dir} 스퀴즈 확장 ({core}/5 조건)"))
    if failed_div_tfs:
        alerts.append((1, f"⚠️ 다이버전스 실패 ({'·'.join(failed_div_tfs)}) — 하락 지속 가능"))
    if bear_flag_tfs:
        alerts.append((2, f"⚠️ 베어 플래그 ({'·'.join(bear_flag_tfs)}) — 반등 후 재하락 주의"))
    for tf, summ in synth_alerts[:2]:
        alerts.append((3, f"📊 {tf} {summ}"))

    # 프레임 간 충돌 (상위 vs 하위 RSI 방향)
    def _first_dir(tfs):
        for tf in tfs:
            r = results.get(tf)
            if r and not r.get("error"):
                return "상승" if r["rsi"] >= 50 else "하락"
        return None
    big_dir = _first_dir(["1일", "4시간", "1시간"])
    small_dir = _first_dir(["15분", "5분", "1분"])
    if big_dir and small_dir and big_dir != small_dir:
        alerts.append((4, f"⚠️ 프레임 충돌 — 상위({big_dir}) ↔ 하위({small_dir})"))

    # HTF 필터 (롱 감점) — 하나만
    for tf in ["15분", "5분", "1분"]:
        r = results.get(tf)
        if r and r.get("htf_filter") == "HTF_BEARISH":
            alerts.append((5, f"⚠️ {tf} 상위프레임 하락 → 롱 감점"))
            break

    if alerts:
        alerts.sort(key=lambda x: x[0])
        lines.append("\n**⚠️ 주의**")
        lines.extend(f"- {text}" for _, text in alerts[:5])

    return "\n".join(lines)


def format_rsi_wave_for_ai(symbol, results):
    """분석 결과를 AI에게 보낼 텍스트로 포맷팅 (v2 — 레짐/다이버전스/점수/목표가 포함)"""
    lines = [
        f"[🌊 RSI 파동 분석 v2] {symbol} — 레짐 기반 멀티 타임프레임 RSI 사이클 분석\n",
        "아래 7개 타임프레임의 RSI 사이클 상태를 **시장 레짐별로** 분석해주세요.\n",
    ]

    # ── 펀딩/프리미엄 (심볼 단위 — 헤더에 1회) ──
    for tf in WAVE_TIMEFRAMES:
        r = results.get(tf)
        if r and not r.get("error") and r.get("funding_analysis"):
            f = r["funding_analysis"]
            lines.append(f"💸 펀딩/프리미엄(심볼 공통): {f['label']} — {f['detail']}\n")
            break

    # ── 🧭 기계 판정 (방향 ≠ 즉시진입 — 리포트에 반드시 반영) ──
    ea = assess_entry(results)
    if ea:
        lv = ea.get("levels") or {}
        lines.append("🧭 기계 판정 (아래 수치를 결론·진입판정·손익비에 반드시 반영하세요):")
        lines.append(f"- 방향: {ea['direction']} (정렬 {ea['align']}, 방향성 점수 {ea['direction_score']})")
        lines.append(f"- 즉시 진입 적합도: {ea['entry_score']}/100 → 추격 {'가능' if ea['chase_ok'] else '부적합(대기)'}")
        lines.append(f"- 권장 포지션 크기: {ea['position_size']} | 리스크 등급: {ea['risk_level']}")
        if ea.get("rr") is not None:
            lines.append(f"- 현재가 추격 손익비: {ea['rr']}R ({'양호' if ea['rr'] >= 1.5 else '보통' if ea['rr'] >= 1 else '불리'})")
        if lv:
            level_bits = [f"현재가 {lv.get('price')}"]
            if lv.get("resistance"): level_bits.append(f"저항/무효화 {lv['resistance']}")
            if lv.get("support"): level_bits.append(f"지지/무효화 {lv['support']}")
            if lv.get("target"): level_bits.append(f"1차 목표 {lv['target']}")
            lines.append(f"- 기준 TF {ea['ref_tf']}: {' · '.join(level_bits)}")
        if ea.get("blocks"):
            lines.append(f"- 진입 금지/주의: {' / '.join(ea['blocks'])}")
        lines.append("")

    # ── 📍 핵심 레벨 맵 (AI는 이 가격들만 인용) ──
    lm = build_level_map(results)
    if lm:
        lines.append("📍 핵심 레벨 맵 — ⚠️ 리포트의 모든 가격은 아래 값만 사용하고, 없는 가격은 만들지 마세요:")
        lines.append(_format_level_map(lm))
        lines.append("")

    # ── 📐 진입 시나리오 손익비 (파이썬 계산 — AI는 이 R값 그대로 사용) ──
    es = build_entry_scenarios(results)
    if es and es["scenarios"]:
        lines.append("📐 진입 시나리오 손익비 (파이썬 계산 — 손익비는 직접 암산하지 말고 이 값을 사용):")
        for s in es["scenarios"]:
            rr = f"{s['R']}R" if s["R"] is not None else "-"
            lines.append(f"- {s['label']}: 진입 {s['entry']} / 손절 {s['stop']} / 목표 {s['target']} → {rr} ({s['grade']})")
        lines.append("")

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

        # CVD 추세 + 다이버전스 (NEW — 최우선)
        if r.get("cvd") is not None and r.get("cvd_ema") is not None:
            cvd_dir = "매수우위(↑)" if r["cvd"] > r["cvd_ema"] else "매도우위(↓)"
            lines.append(f"📈 CVD: {cvd_dir} (CVD={r['cvd']:,.0f} EMA={r['cvd_ema']:,.0f})")
        cvd_div_data = r.get("cvd_div")
        if cvd_div_data:
            lines.append(f"📊 CVD 다이버전스(최우선): {cvd_div_data['label']} — {cvd_div_data['detail']}")

        # OI 변화 (NEW)
        oi_an = r.get("oi_analysis")
        if oi_an:
            lines.append(
                f"🔗 OI: {oi_an['label']} "
                f"(OI {oi_an['oi_change_pct']:+.1f}%, 가격 {oi_an['price_change_pct']:+.1f}%) — {oi_an['detail']}"
            )

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
    lines.append("CVD/OBV/거래량 다이버전스가 RSI 다이버전스와 일치/충돌하는지 반드시 언급하세요.")
    lines.append("특히 CVD(실제 시장가 체결)가 가격을 따라가는지 확인 — RSI 상승 다이버전스가 있어도 CVD가 저점 갱신이면 다이버전스 실패 가능성이 높습니다.")
    lines.append("OI 변화로 그 움직임이 신규 진입(추세 신뢰)인지 청산/숏커버(지속력 의심)인지 구분하세요.")
    lines.append("펀딩이 극단(숏 과열/롱 과열)이면 스퀴즈 반전 위험을 반드시 경고하세요 — 추격 진입 감점 요인입니다.")
    lines.append("각 관점(스캘핑/데이트레이딩/스윙/장기)별 현재 사이클 위치와 매매 방향을 구체적으로 판단하세요.")
    lines.append("상위-하위 프레임 간 충돌이나 다이버전스가 있으면 반드시 언급하세요.")
    lines.append("진입/청산 타점이 보이면 구체적 가격을 제시하세요.")

    return "\n".join(lines)


# ── 호환 재 export: 렌더링 함수는 core/rsi_render.py 로 이동 ──
# (지연 임포트 — 순환 임포트 없이 기존 임포트 경로 유지)
_RENDER_FUNCS = ("generate_wave_svg", "generate_price_ladder_svg", "generate_tf_cards")


def __getattr__(name):
    if name in _RENDER_FUNCS:
        from core import rsi_render
        return getattr(rsi_render, name)
    raise AttributeError(name)
