"""
AI 클라이언트 — 차트 이미지 + 데이터 결합 분석
OpenAI(gpt-*)와 Anthropic Claude(claude-*) 듀얼 프로바이더.
모델명으로 프로바이더를 판별하므로 호출부는 model 문자열만 바꾸면 된다.
"""
import json
from openai import OpenAI


def is_claude_model(model):
    """모델명으로 프로바이더 판별 — claude-* 는 Anthropic API 사용"""
    return (model or "").startswith("claude")


# 채팅 UI에서 선택 가능한 Claude 모델 (기본: 최상위 Fable 5)
CLAUDE_MODELS = [
    "claude-fable-5",      # 최신·최강 (Opus 위 티어, $10/$50 per 1M tokens)
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]

# 적응형 사고(adaptive thinking)를 지원하는 모델 프리픽스.
# 해당 모델엔 thinking을 켜서 복잡한 차트 분석 품질을 높인다.
# (haiku-4-5 등 미지원 모델에 보내면 400이므로 화이트리스트 방식)
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable", "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
    "claude-sonnet-4-6",
)


def _claude_kwargs(model):
    """모델별 Claude 추가 파라미터"""
    extra = {}
    if model.startswith(_ADAPTIVE_THINKING_PREFIXES):
        extra["thinking"] = {"type": "adaptive"}
    return extra


SYSTEM_PROMPT_BASE = """당신은 암호화폐/주식 기술적 분석 전문가입니다. 사용자는 차트를 볼 줄 아는 숙련 트레이더입니다.

당신에게 제공될 수 있는 것:
1. 사용자의 TradingView 차트 스크린샷 (있을 수도 없을 수도 있음)
2. 실시간 시장 데이터 (Binance API에서 자동 계산된 지표 수치)

━━━ 핵심 원칙 ━━━
- 지표 수치 → 실시간 데이터(텍스트)에서 가져올 것
- 차트 패턴/드로잉 → 스크린샷에서 시각적으로 확인할 것
- "미제공", "확인 불가" 등으로 데이터를 무시하지 말 것

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

분석 시 적용 규칙:
1. 스크린샷에서 H .00 / L .00 라벨이 보이면, 해당 가격을 핵심 지지/저항으로 우선 반영
2. 동일 가격에 여러 번 나타난 H .00 → 매우 강한 저항 (돌파 실패 시 숏 유리)
3. 동일 가격에 여러 번 나타난 L .00 → 매우 강한 지지 (이탈 시 하락 가속)
4. 현재가에서 가장 가까운 H .00 / L .00 가격을 반드시 언급
5. .00 변곡점과 EMA/VWAP/피보나치 레벨이 겹치는 구간은 컨플루언스 존으로 강조

━━━ 대화 연속성 (중요) ━━━
- 이 대화는 하나의 분석 방이다. 앞선 RSI 파동 분석·차트 논의·사용자가 밝힌 포지션/관점을 모두 인지하고 일관되게 이어가라.
- 앞 분석에서 제시된 레벨·방향·시나리오가 있으면 그것을 기준으로 답하라. 새 판단이 다르면 "앞선 분석에선 X였는데 지금은 Y(이유)"로 차이를 명시하라(앞 내용을 무시하고 새로 시작하지 말 것).
- 사용자가 특정 방향(예: 숏)을 논의 중이면 그 맥락에서 답하되, 데이터가 반대를 가리키면 동조하지 말고 분명히 알려라.

━━━ 분석 규칙 ━━━
- 원론적/교과서적 설명 금지 (지표 설명 불필요)
- 실시간 수치 기반 구체적 판단만 제공
- 간결하게 핵심만 답변 (단, 위 '대화 연속성'은 항상 우선)
- 질문에 답하되, 방의 앞 맥락(직전 RSI 분석 등)과 연결해서 답할 것
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


def analyze_chart(api_key, model, messages, image_base64=None, market_data="", extra_images=None, system_prompt_override=None):
    """
    차트 이미지 + 시장 데이터로 AI 분석 스트리밍 (OpenAI / Claude 공용).
    extra_images: 추가 이미지 base64 문자열 리스트
    system_prompt_override: 시스템 프롬프트 완전 교체 (RSI 파동 분석 등)
    Yields: content chunks (str)
    """
    # ── 시스템 프롬프트 결정 (프로바이더 공통) ──
    has_images = image_base64 is not None or (extra_images and len(extra_images) > 0)
    if system_prompt_override:
        system_msg = system_prompt_override
    else:
        # 이미지가 있을 때만 브리핑 지시 포함
        system_msg = SYSTEM_PROMPT_BASE
        if has_images:
            system_msg += SYSTEM_PROMPT_BRIEFING
    # 실시간 데이터는 한 곳에만 넣는다 (중복 전송 = 토큰 2배):
    # override(RSI 파동 등) → 시스템 프롬프트, 일반 채팅 → 마지막 user 메시지
    if market_data and system_prompt_override:
        system_msg += f"\n\n{market_data}"

    # ── 이전 대화 내역 (최근 20개만 — 토큰 초과 방지) ──
    recent_messages = messages[:-1]
    if len(recent_messages) > 20:
        recent_messages = recent_messages[-20:]
    history = [{"role": m["role"], "content": m["content"]} for m in recent_messages]

    # ── 마지막 메시지 텍스트 ──
    last_text = messages[-1]["content"]
    # 일반 채팅: 실시간 데이터를 마지막 user 메시지에 포함
    # (시스템 프롬프트보다 마지막 user 메시지의 데이터를 모델이 더 잘 따름)
    if market_data and not system_prompt_override:
        last_text = last_text + f"\n\n{market_data}"

    # 첨부 이미지 (첫 장 + 추가 장, base64 JPEG)
    images = ([image_base64] if image_base64 else []) + list(extra_images or [])

    if is_claude_model(model):
        yield from _stream_claude(api_key, model, system_msg, history, last_text, images)
    else:
        yield from _stream_openai(api_key, model, system_msg, history, last_text, images)


def _stream_openai(api_key, model, system_msg, history, last_text, images):
    """OpenAI chat.completions 스트리밍"""
    client = OpenAI(api_key=api_key)
    api_messages = [{"role": "system", "content": system_msg}] + history

    if images:
        content_parts = [{"type": "text", "text": last_text}]
        for b64 in images:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }
            })
        api_messages.append({"role": "user", "content": content_parts})
    else:
        api_messages.append({"role": "user", "content": last_text})

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


def _stream_claude(api_key, model, system_msg, history, last_text, images):
    """Anthropic Messages API 스트리밍.

    OpenAI와의 포맷 차이:
    - 시스템 프롬프트는 messages 배열이 아닌 별도 system 파라미터
    - 이미지는 image_url 대신 {"type": "image", "source": {"type": "base64", ...}}
    - opus-4.7+ 는 temperature 미지원 → 전송하지 않음 (adaptive thinking 사용)
    """
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    api_messages = list(history)
    if images:
        content_blocks = [{"type": "text", "text": last_text}]
        for b64 in images:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })
        api_messages.append({"role": "user", "content": content_blocks})
    else:
        api_messages.append({"role": "user", "content": last_text})

    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system_msg,
        messages=api_messages,
        **_claude_kwargs(model),
    ) as stream:
        for text in stream.text_stream:
            yield text


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
    거래 종료 시 대화 전체를 분석하여 매매 요약 반환 (OpenAI / Claude 공용).
    반환: dict (파싱 실패 시 기본값 반환)
    """
    # 대화 내용을 텍스트로 변환
    conversation_text = ""
    for msg in messages:
        role = "트레이더" if msg["role"] == "user" else "AI"
        conversation_text += f"[{role}]: {msg['content']}\n\n"

    try:
        if is_claude_model(model):
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=TRADE_SUMMARY_PROMPT,
                messages=[{"role": "user", "content": conversation_text}],
            )
            content = next(
                (b.text for b in response.content if b.type == "text"), ""
            ).strip()
        else:
            client = OpenAI(api_key=api_key)
            api_messages = [
                {"role": "system", "content": TRADE_SUMMARY_PROMPT},
                {"role": "user", "content": conversation_text},
            ]
            is_new = model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4")
            kwargs = {
                "model": model,
                "messages": api_messages,
                # JSON 모드 — 모델이 유효한 JSON만 출력하도록 강제 (파싱 실패 방지)
                "response_format": {"type": "json_object"},
            }
            if is_new:
                kwargs["max_completion_tokens"] = 1000
            else:
                kwargs["max_tokens"] = 1000
                kwargs["temperature"] = 0.3

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception:
                # 일부 모델이 JSON 모드를 지원하지 않으면 일반 모드로 재시도
                kwargs.pop("response_format", None)
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
