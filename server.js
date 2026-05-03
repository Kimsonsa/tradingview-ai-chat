import express from 'express';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = 3000;

app.use(express.json());
app.use(express.static(join(__dirname, 'public')));

// In-memory state
let apiKey = null;
let openai = null;
let modelName = 'gpt-5.5';

// Set API key
app.post('/api/set-key', (req, res) => {
  const { key, model } = req.body;
  if (!key) return res.status(400).json({ error: 'API key is required' });
  apiKey = key;
  if (model) modelName = model;
  openai = new OpenAI({ apiKey });
  res.json({ success: true, model: modelName });
});

// Check API key
app.get('/api/check-key', (req, res) => {
  res.json({ hasKey: !!apiKey, model: modelName });
});

// Chat endpoint with SSE streaming
app.post('/api/chat', async (req, res) => {
  if (!openai) return res.status(401).json({ error: 'API key not set' });

  const { messages, chartContext } = req.body;

  const systemMessage = {
    role: 'system',
    content: `당신은 전문 암호화폐 트레이딩 분석가입니다. 사용자가 보고 있는 차트를 기반으로 기술적 분석을 제공합니다.

현재 차트 정보:
- 종목: ${chartContext?.symbol || 'BTCUSDT.P'}
- 타임프레임: ${chartContext?.interval || '1시간'}
- 거래소: Binance (선물)

적용된 보조지표:
- EMA 20, 50, 200
- RSI (14)
- 거래량 (Volume)

분석 시 다음을 포함해주세요:
1. 현재 추세 분석 (EMA 배열 기반)
2. RSI 과매수/과매도 상태
3. 거래량 분석
4. 주요 지지/저항 레벨
5. 매매 전략 제안

답변은 한국어로 해주세요. 구체적인 수치와 함께 분석해주세요.
마크다운 포맷을 사용하여 가독성 좋게 답변해주세요.

⚠️ 중요: 이것은 투자 조언이 아닌 기술적 분석 의견임을 항상 명시해주세요.`
  };

  try {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');

    const stream = await openai.chat.completions.create({
      model: modelName,
      messages: [systemMessage, ...messages],
      stream: true,
      temperature: 0.7,
      max_tokens: 2000,
    });

    for await (const chunk of stream) {
      const content = chunk.choices[0]?.delta?.content || '';
      if (content) {
        res.write(`data: ${JSON.stringify({ content })}\n\n`);
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (error) {
    console.error('Chat error:', error);
    const errMsg = error?.message || 'Unknown error';
    res.write(`data: ${JSON.stringify({ error: errMsg })}\n\n`);
    res.end();
  }
});

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
