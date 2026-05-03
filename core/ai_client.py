"""
OpenAI Vision API 클라이언트 — 차트 이미지 + 데이터 결합 분석
"""
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
                        "url": f"data:image/png;base64,{image_base64}",
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
        kwargs["max_completion_tokens"] = 4000
    else:
        kwargs["max_tokens"] = 4000
        kwargs["temperature"] = 0.7

    stream = client.chat.completions.create(**kwargs)

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content
