/* ============================================
   Chat Module — Direct OpenAI API + Binance Market Data
   ============================================ */

/* ---- Market Data Fetcher (Binance Futures API) ---- */
class MarketData {
  static INTERVAL_MAP = {
    '1':'1m','5':'5m','15':'15m','60':'1h','240':'4h','D':'1d'
  };

  static async fetch(symbol, interval) {
    const pair = symbol.replace('.P','').replace('BINANCE:','');
    const bi = this.INTERVAL_MAP[interval] || '1h';
    const url = `https://fapi.binance.com/fapi/v1/klines?symbol=${pair}&interval=${bi}&limit=210`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Binance API 오류');
    const raw = await res.json();
    // [openTime, open, high, low, close, volume, ...]
    return raw.map(c => ({
      time: c[0], open: +c[1], high: +c[2], low: +c[3], close: +c[4], volume: +c[5]
    }));
  }

  static calcEMA(closes, period) {
    const k = 2 / (period + 1);
    let ema = closes[0];
    const result = [ema];
    for (let i = 1; i < closes.length; i++) {
      ema = closes[i] * k + ema * (1 - k);
      result.push(ema);
    }
    return result;
  }

  static calcRSI(closes, period = 14) {
    const gains = [], losses = [];
    for (let i = 1; i < closes.length; i++) {
      const diff = closes[i] - closes[i - 1];
      gains.push(diff > 0 ? diff : 0);
      losses.push(diff < 0 ? -diff : 0);
    }
    let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
    const rsi = [];
    for (let i = period; i < gains.length; i++) {
      avgGain = (avgGain * (period - 1) + gains[i]) / period;
      avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(+(100 - 100 / (1 + rs)).toFixed(2));
    }
    return rsi;
  }

  static async getAnalysisContext(symbol, interval) {
    try {
      const candles = await this.fetch(symbol, interval);
      const closes = candles.map(c => c.close);
      const volumes = candles.map(c => c.volume);

      const ema20 = this.calcEMA(closes, 20);
      const ema50 = this.calcEMA(closes, 50);
      const ema200 = this.calcEMA(closes, 200);
      const rsi = this.calcRSI(closes);

      const last = candles[candles.length - 1];
      const cur = last.close;
      const e20 = ema20[ema20.length - 1];
      const e50 = ema50[ema50.length - 1];
      const e200 = ema200[ema200.length - 1];
      const curRSI = rsi[rsi.length - 1];

      // 최근 20봉 고가/저가
      const recent20 = candles.slice(-20);
      const high20 = Math.max(...recent20.map(c => c.high));
      const low20 = Math.min(...recent20.map(c => c.low));

      // 최근 5봉 평균 거래량 vs 현재
      const avgVol5 = volumes.slice(-6, -1).reduce((a, b) => a + b, 0) / 5;
      const curVol = last.volume;
      const volRatio = (curVol / avgVol5 * 100).toFixed(0);

      // 최근 10봉 데이터
      const recentCandles = candles.slice(-10).map(c =>
        `  ${new Date(c.time).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',month:'numeric',day:'numeric',hour:'2-digit',minute:'2-digit'})} | O:${c.open.toFixed(1)} H:${c.high.toFixed(1)} L:${c.low.toFixed(1)} C:${c.close.toFixed(1)} V:${c.volume.toFixed(0)}`
      ).join('\n');

      // EMA 배열 판단
      let trend = '';
      if (cur > e20 && e20 > e50 && e50 > e200) trend = '강한 상승 정배열 ↑';
      else if (cur > e20 && e20 > e50) trend = '상승 추세 ↑';
      else if (cur < e20 && e20 < e50 && e50 < e200) trend = '강한 하락 역배열 ↓';
      else if (cur < e20 && e20 < e50) trend = '하락 추세 ↓';
      else trend = '횡보/혼조세 ↔';

      return `
📊 실시간 시장 데이터 (Binance Futures):
━━━━━━━━━━━━━━━━━━━━━━━
현재가: ${cur.toFixed(1)} USDT
24봉 고가: ${high20.toFixed(1)} | 24봉 저가: ${low20.toFixed(1)}

📈 이동평균선 (EMA):
- EMA 20: ${e20.toFixed(1)} (${cur > e20 ? '현재가 위 ▲' : '현재가 아래 ▼'})
- EMA 50: ${e50.toFixed(1)} (${cur > e50 ? '현재가 위 ▲' : '현재가 아래 ▼'})
- EMA 200: ${e200.toFixed(1)} (${cur > e200 ? '현재가 위 ▲' : '현재가 아래 ▼'})
- EMA 배열: ${trend}

📉 RSI (14): ${curRSI} ${curRSI > 70 ? '⚠️ 과매수 구간' : curRSI < 30 ? '⚠️ 과매도 구간' : '중립 구간'}

📊 거래량:
- 현재봉 거래량: ${curVol.toFixed(0)}
- 최근5봉 평균 대비: ${volRatio}% ${+volRatio > 150 ? '🔥 거래량 급증' : +volRatio < 50 ? '📉 거래량 감소' : ''}

📋 최근 10봉 OHLCV:
${recentCandles}
━━━━━━━━━━━━━━━━━━━━━━━`;
    } catch (e) {
      console.error('Market data error:', e);
      return '\n⚠️ 실시간 데이터를 가져오지 못했습니다. 일반 분석으로 진행합니다.\n';
    }
  }
}


/* ---- Chat Manager ---- */
class ChatManager {
  constructor() {
    this.messages = [];
    this.isStreaming = false;
    this.chartContext = { symbol: 'BTCUSDT.P', interval: '60', intervalLabel: '1시간' };
  }

  setContext(symbol, interval, intervalLabel) {
    this.chartContext = { symbol, interval, intervalLabel: intervalLabel || interval };
  }

  getSystemPrompt(marketData) {
    return `당신은 암호화폐 기술적 분석 전문가입니다. 사용자는 차트를 볼 줄 아는 트레이더입니다.

규칙:
- 원론적/교과서적 설명 금지 (EMA란 무엇인가, RSI란 등 설명 불필요)
- 오직 아래 실시간 데이터 기반으로 구체적 수치와 판단만 제공
- 간결하게 핵심만 답변
- 질문에 해당하는 내용만 답변 (묻지 않은 것은 생략)

종목: ${this.chartContext.symbol} | ${this.chartContext.intervalLabel} | Binance 선물
${marketData}

⚠️ 투자 조언이 아닌 기술적 분석 의견입니다.`;
  }

  async sendMessage(userText) {
    if (!userText.trim() || this.isStreaming) return;

    const apiKey = localStorage.getItem('tradeai_apikey');
    if (!apiKey) {
      this.showError('🔑 설정에서 OpenAI API 키를 먼저 입력하세요.');
      return;
    }

    this.messages.push({ role: 'user', content: userText });
    this.renderUserMessage(userText);
    this.hideWelcome();

    this.isStreaming = true;
    this.updateStatus('데이터 수집 중...', 'typing');
    this.updateSendButton();
    const typingEl = this.showTypingIndicator();

    const model = localStorage.getItem('tradeai_model') || 'gpt-5.5';

    try {
      // 실시간 시장 데이터 수집
      this.updateStatus('차트 데이터 분석 중...', 'typing');
      const symbol = this.chartContext.symbol;
      const interval = this.chartContext.interval;
      const marketData = await MarketData.getAnalysisContext(symbol, interval);

      this.updateStatus('AI 분석 중...', 'typing');

      const useNewParam = model.startsWith('gpt-5') || model.startsWith('o3') || model.startsWith('o4');
      const body = {
        model: model,
        messages: [
          { role: 'system', content: this.getSystemPrompt(marketData) },
          ...this.messages
        ],
        stream: true,
      };
      if (useNewParam) {
        body.max_completion_tokens = 4000;
      } else {
        body.max_tokens = 4000;
        body.temperature = 0.7;
      }

      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error?.message || `API 오류 (${response.status})`);
      }

      typingEl.remove();
      const { contentEl } = this.createAIMessageBubble();
      let fullContent = '';

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6).trim();
          if (data === '[DONE]') break;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content || '';
            if (content) {
              fullContent += content;
              this.renderMarkdown(contentEl, fullContent);
              this.scrollToBottom();
            }
          } catch (e) { /* ignore partial JSON */ }
        }
      }

      this.messages.push({ role: 'assistant', content: fullContent });
      this.saveChat();

    } catch (error) {
      typingEl?.remove();
      this.showError(error.message);
    } finally {
      this.isStreaming = false;
      this.updateStatus('대기 중', 'online');
      this.updateSendButton();
    }
  }

  renderUserMessage(text) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
      <div class="message-avatar">You</div>
      <div class="message-content">${this.escapeHtml(text)}</div>
    `;
    container.appendChild(div);
    this.scrollToBottom();
  }

  createAIMessageBubble() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message ai';
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = 'AI';
    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';
    div.appendChild(avatar);
    div.appendChild(contentEl);
    container.appendChild(div);
    return { messageEl: div, contentEl };
  }

  renderMarkdown(el, text) {
    if (typeof marked !== 'undefined') {
      marked.setOptions({ breaks: true, gfm: true });
      el.innerHTML = marked.parse(text);
    } else {
      el.innerHTML = this.escapeHtml(text).replace(/\n/g, '<br>');
    }
  }

  showTypingIndicator() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message ai';
    div.id = 'typingMsg';
    div.innerHTML = `<div class="message-avatar">AI</div><div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
    container.appendChild(div);
    this.scrollToBottom();
    return div;
  }

  showError(msg) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message-error';
    div.textContent = `⚠️ ${msg}`;
    container.appendChild(div);
    this.scrollToBottom();
  }

  hideWelcome() {
    const w = document.querySelector('.welcome-message');
    if (w) w.style.display = 'none';
  }

  updateStatus(text, type) {
    const el = document.getElementById('chatStatus');
    if (el) el.innerHTML = `<span class="status-dot ${type}"></span> ${text}`;
  }

  updateSendButton() {
    const btn = document.getElementById('sendBtn');
    if (btn) btn.disabled = this.isStreaming;
  }

  scrollToBottom() {
    const c = document.getElementById('chatMessages');
    requestAnimationFrame(() => { c.scrollTop = c.scrollHeight; });
  }

  clearChat() {
    this.messages = [];
    localStorage.removeItem('tradeai_chat');
    const container = document.getElementById('chatMessages');
    container.innerHTML = `
      <div class="welcome-message">
        <div class="ai-avatar"><span>◈</span></div>
        <h3>AI 트레이딩 어시스턴트</h3>
        <p>차트를 보면서 궁금한 점을 물어보세요.<br>기술적 분석, 매매 전략, 지지/저항 분석을 도와드립니다.</p>
        <div class="quick-actions" id="quickActions">
          <button class="quick-btn" data-msg="현재 차트 기술적 분석을 해주세요">📊 차트 분석</button>
          <button class="quick-btn" data-msg="현재 매매 전략을 제안해주세요">💡 매매 전략</button>
          <button class="quick-btn" data-msg="주요 지지선과 저항선을 알려주세요">📐 지지/저항</button>
          <button class="quick-btn" data-msg="현재 RSI와 EMA 상태를 분석해주세요">📈 지표 분석</button>
        </div>
      </div>`;
    this.bindQuickActions();
  }

  bindQuickActions() {
    document.querySelectorAll('.quick-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const msg = btn.getAttribute('data-msg');
        if (msg) this.sendMessage(msg);
      });
    });
  }

  saveChat() {
    try {
      localStorage.setItem('tradeai_chat', JSON.stringify(this.messages));
    } catch (e) { /* ignore quota errors */ }
  }

  loadChat() {
    try {
      const saved = localStorage.getItem('tradeai_chat');
      if (!saved) return;
      this.messages = JSON.parse(saved);
      if (this.messages.length > 0) {
        this.hideWelcome();
        for (const msg of this.messages) {
          if (msg.role === 'user') this.renderUserMessage(msg.content);
          else if (msg.role === 'assistant') {
            const { contentEl } = this.createAIMessageBubble();
            this.renderMarkdown(contentEl, msg.content);
          }
        }
        this.scrollToBottom();
      }
    } catch (e) { /* ignore */ }
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

window.chatManager = new ChatManager();
