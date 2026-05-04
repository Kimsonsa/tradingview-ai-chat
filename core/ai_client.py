"""
OpenAI Vision API 클라이언트 — 차트 이미지 + 데이터 결합 분석
"""
import json
from openai import OpenAI


SYSTEM_PROMPT = """당신은 암호화폐/주식 기술적 분석 전문가입니다. 사용자는 차트를 볼 줄 아는 숙련 트레이더입니다.

당신에게 두 가지가 제공됩니다:
1. 사용자의 TradingView 차트 스크린샷 (지지선, 저항선, 트렌드라인 등 드로잉 포함)
2. 실시간 시장 데이터 (가격, EMA, RSI, MACD, 볼린저밴드 등)

중요: 스크린샷에서 종목명과 타임프레임을 직접 확인하고, 해당 종목에 맞는 분석을 제공하세요.

규칙:
- 차트 이미지에 그려진 선/패턴을 반드시 참조하여 분석
- 원론적/교과서적 설명 금지 (지표 설명 불필요)
- 실시간 수치 기반 구체적 판단만 제공
- 간결하게 핵심만 답변
- 질문에 해당하는 내용만 답변

⚠️ 투자 조언이 아닌 기술적 분석 의견입니다."""


def analyze_chart(api_key, model, messages, image_base64=None, market_data=""):
    """
    차트 이미지 + 시장 데이터로 AI 분석 스트리밍.
    Yields: content chunks (str)
    """
    client = OpenAI(api_key=api_key)

    system_msg = SYSTEM_PROMPT
    if market_data:
        system_msg += f"\n\n{market_data}"

    api_messages = [{"role": "system", "content": system_msg}]

    # 이전 대화 내역 추가
    for msg in messages[:-1]:
        api_messages.append({"role": msg["role"], "content": msg["content"]})

    # 마지막 메시지 (이미지 포함 가능)
    last_msg = messages[-1]
    if image_base64:
        api_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": last_msg["content"]},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "high"
                    }
                }
            ]
        })
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
