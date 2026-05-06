"""
OpenAI Vision API 클라이언트 — 차트 이미지 + 데이터 결합 분석
"""
import json
from openai import OpenAI


SYSTEM_PROMPT_BASE = """당신은 암호화폐/주식 기술적 분석 전문가입니다. 사용자는 차트를 볼 줄 아는 숙련 트레이더입니다.

당신에게 제공될 수 있는 것:
1. 사용자의 TradingView 차트 스크린샷 (있을 수도 없을 수도 있음)
2. 실시간 시장 데이터 (Binance API에서 자동 계산된 지표 수치)

━━━ 핵심 원칙 ━━━
- 지표 수치 → 실시간 데이터(텍스트)에서 가져올 것
- 차트 패턴/드로잉 → 스크린샷에서 시각적으로 확인할 것
- "미제공", "확인 불가" 등으로 데이터를 무시하지 말 것

━━━ 분석 규칙 ━━━
- 원론적/교과서적 설명 금지 (지표 설명 불필요)
- 실시간 수치 기반 구체적 판단만 제공
- 간결하게 핵심만 답변
- 질문에 해당하는 내용만 답변
- 마크다운 취소선(~~텍스트~~) 절대 사용 금지. 삭제/무효 표현이 필요하면 일반 텍스트로 서술

⚠️ 투자 조언이 아닌 기술적 분석 의견입니다."""

SYSTEM_PROMPT_BRIEFING = """
━━━ 지표 브리핑 규칙 (차트 이미지 첨부됨) ━━━

차트 스크린샷이 첨부되었으므로 응답 첫머리에 "📊 지표 브리핑"을 반드시 포함하세요.

⚠️⚠️⚠️ 최우선 규칙: 표의 수치는 반드시 아래 텍스트 데이터에서 가져오세요! ⚠️⚠️⚠️

이 메시지 하단에 "📊 [○○] 실시간 데이터" 블록이 있습니다. 이 블록에서:
- "📈 EMA: 20=○ | 50=○ | 200=○" → EMA 행에 기입
- "📉 RSI(14): ○○ | StochRSI: K=○ D=○" → RSI, 스토캐스틱 RSI 행에 기입
- "📊 MACD: ○ | Sig: ○ | Hist: ○" → MACD 행에 기입
- "📏 볼린저(20,2): 상=○ | 중=○ | 하=○" → 볼린저밴드 행에 기입
- "📐 ATR(14): ○ | ADX(14): ○" → ATR, ADX 행에 기입
- "💰 VWAP: ○ | OBV: ○" → VWAP, OBV 행에 기입
- "📊 거래량: ○" → 거래량 행에 기입

"차트상 확인 없음", "판단 제외", "미제공" 등은 절대 금지!
차트 이미지는 패턴/드로잉 확인 용도이지, 지표 수치를 읽는 용도가 아닙니다.

【타임프레임 확인】
- 각 이미지의 왼쪽 상단 차트 헤더에서 타임프레임 확인 (예: "· 15 ·" = 15분봉)
- 첨부된 이미지의 타임프레임만 브리핑

【표 형식 — 반드시 아래 10행 고정, 행 추가/삭제/변경 금지】

## 📊 지표 브리핑

### ⏱ ○○봉
| 지표 | 현재 상황 | 해석 |
|------|-----------|------|
| EMA (20/50/200) | 20=○ 50=○ 200=○ | (배열/가격 위치) |
| RSI (14) | ○○ | (과매수/과매도/중립) |
| 스토캐스틱 RSI | K=○ D=○ | (크로스/과매수/과매도) |
| MACD | MACD=○ Sig=○ Hist=○ | (골든/데드크로스) |
| 볼린저밴드 | 상=○ 중=○ 하=○ 밴드폭=○% | (스퀴즈/확장) |
| ATR (14) | ○ (○%) | (변동성) |
| ADX (14) | ADX=○ +DI=○ -DI=○ | (추세 강도) |
| OBV | ○ | (매집/분산) |
| VWAP | ○ | (가격 위치) |
| 거래량 | ○ (평균 대비 ○%) | (급증/감소) |

위 10행을 정확히 유지하세요. "채널", "가격 위치", "단기 구조" 같은 임의 행을 추가하거나 기존 행을 대체하지 마세요.

### 🔗 종합 해석
> 모든 타임프레임의 지표들을 조합하여 현재 시장 상태를 **한 문단(3~5문장)**으로 요약.
> 타임프레임 간 상충하는 신호가 있으면 반드시 언급.

여기까지가 지표 브리핑입니다. 이후에 사용자 질문에 대한 본격적인 분석/답변을 작성하세요."""


def analyze_chart(api_key, model, messages, image_base64=None, market_data="", extra_images=None):
    """
    차트 이미지 + 시장 데이터로 AI 분석 스트리밍.
    extra_images: 추가 이미지 base64 문자열 리스트
    Yields: content chunks (str)
    """
    client = OpenAI(api_key=api_key)

    # 이미지가 있을 때만 브리핑 지시 포함
    has_images = image_base64 is not None or (extra_images and len(extra_images) > 0)
    system_msg = SYSTEM_PROMPT_BASE
    if has_images:
        system_msg += SYSTEM_PROMPT_BRIEFING
    if market_data:
        system_msg += f"\n\n{market_data}"

    api_messages = [{"role": "system", "content": system_msg}]

    # 이전 대화 내역 추가
    for msg in messages[:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # 마지막 메시지 (이미지 포함 가능)
    last_msg = messages[-1]
    if image_base64:
        content_parts = [
            {"type": "text", "text": last_msg["content"]},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}",
                    "detail": "high"
                }
            }
        ]
        # 추가 이미지들
        if extra_images:
            for extra_b64 in extra_images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{extra_b64}",
                        "detail": "high"
                    }
                })
        api_messages.append({"role": "user", "content": content_parts})
    else:
        api_messages.append({"role": "user", "content": last_msg["content"]})

    # 모델별 파라미터 조정
    is_new = model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4")
    kwargs = {
        "model": model,
        "messages": api_messages,
        "stream": True,
    }
    if is_new:
        kwargs["max_completion_tokens"] = 16000
    else:
        kwargs["max_tokens"] = 16000
        kwargs["temperature"] = 0.7

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# ─── 거래 종료 시 대화 분석 ───

TRADE_SUMMARY_PROMPT = """아래는 트레이더와 AI 어시스턴트의 대화 기록입니다.
이 대화를 분석하여 거래 요약을 JSON 형식으로 작성하세요.

반드시 아래 JSON 형식만 출력하세요 (다른 텍스트 없이):

{
  "symbol": "거래 종목 (예: BTCUSDT)",
  "direction": "LONG 또는 SHORT 또는 UNKNOWN",
  "result": "익절 또는 손절 또는 미확정",
  "entry_price": 진입가(숫자) 또는 null,
  "exit_price": 종료가(숫자) 또는 null,
  "pnl_percent": 손익률(숫자, 예: 2.3 또는 -1.1) 또는 null,
  "actions": ["물타기", "불타기", "부분익절", "부분손절"] 중 해당하는 것만 배열로,
  "title": "대화 내용을 한줄로 요약 (15자 이내)",
  "note": "매매 진행 과정 요약 (2-3문장)"
}

규칙:
- 대화에서 명시적으로 언급된 정보만 사용
- 가격이 언급되지 않으면 null
- 방향이 불분명하면 "UNKNOWN"
- 거래가 없는 단순 분석 대화면 result를 "미확정"으로
- title은 핵심만 (예: "BTC 롱 익절 +2.3%", "ETH 숏 손절")
"""


def analyze_trade_summary(api_key, model, messages):
    """
    거래 종료 시 대화 전체를 분석하여 매매 요약 반환.
    반환: dict (파싱 실패 시 기본값 반환)
    """
    client = OpenAI(api_key=api_key)

    # 대화 내용을 텍스트로 변환
    conversation_text = ""
    for msg in messages:
        role = "트레이더" if msg["role"] == "user" else "AI"
        conversation_text += f"[{role}]: {msg['content']}\n\n"

    api_messages = [
        {"role": "system", "content": TRADE_SUMMARY_PROMPT},
        {"role": "user", "content": conversation_text},
    ]

    # 비스트리밍으로 한번에 받기
    is_new = model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4")
    kwargs = {
        "model": model,
        "messages": api_messages,
    }
    if is_new:
        kwargs["max_completion_tokens"] = 1000
    else:
        kwargs["max_tokens"] = 1000
        kwargs["temperature"] = 0.3

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content.strip()

        # JSON 파싱 (코드블록 감싸져 있을 수 있음)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        summary = json.loads(content)

        # 필수 필드 보장
        defaults = {
            "symbol": "UNKNOWN", "direction": "UNKNOWN", "result": "미확정",
            "entry_price": None, "exit_price": None, "pnl_percent": None,
            "actions": [], "title": "거래 분석", "note": "",
        }
        for k, v in defaults.items():
            if k not in summary:
                summary[k] = v

        return summary

    except Exception as e:
        return {
            "symbol": "UNKNOWN", "direction": "UNKNOWN", "result": "미확정",
            "entry_price": None, "exit_price": None, "pnl_percent": None,
            "actions": [], "title": "분석 실패", "note": f"AI 분석 오류: {str(e)}",
        }
