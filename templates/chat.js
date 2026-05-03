/* ============================================
   Chat Module — Direct OpenAI API calls
   ============================================ */
class ChatManager {
  constructor() {
    this.messages = [];
    this.isStreaming = false;
    this.chartContext = { symbol: 'BTCUSDT.P', interval: '1시간' };
  }

  setContext(symbol, interval) {
    this.chartContext = { symbol, interval };
  }

  getSystemPrompt() {
    return `당신은 전문 암호화폐 트레이딩 분석가입니다.

현재 차트 정보:
- 종목: ${this.chartContext.symbol}
- 타임프레임: ${this.chartContext.interval}
- 거래소: Binance (선물)

적용된 보조지표:
- EMA 20 (cyan), EMA 50 (orange), EMA 200 (purple)
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

⚠️ 중요: 이것은 투자 조언이 아닌 기술적 분석 의견임을 항상 명시해주세요.`;
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
    this.updateStatus('분석 중...', 'typing');
    this.updateSendButton();
    const typingEl = this.showTypingIndicator();

    const model = localStorage.getItem('tradeai_model') || 'gpt-5.5';

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: model,
          messages: [
            { role: 'system', content: this.getSystemPrompt() },
            ...this.messages
          ],
          stream: true,
          temperature: 0.7,
          max_tokens: 2000
        })
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
