/* ============================================
   Chat Module — Handles AI chat communication
   ============================================ */

class ChatManager {
  constructor() {
    this.messages = [];
    this.isStreaming = false;
    this.chartContext = {
      symbol: 'BTCUSDT.P',
      interval: '1시간'
    };
  }

  /** Update chart context info */
  setContext(symbol, interval) {
    this.chartContext = { symbol, interval };
  }

  /** Send a message and stream the AI response */
  async sendMessage(userText) {
    if (!userText.trim() || this.isStreaming) return;

    // Add user message
    this.messages.push({ role: 'user', content: userText });
    this.renderUserMessage(userText);
    this.hideWelcome();

    // Show typing indicator
    this.isStreaming = true;
    this.updateStatus('분석 중...', 'typing');
    const typingEl = this.showTypingIndicator();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: this.messages,
          chartContext: this.chartContext
        })
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || 'API 요청 실패');
      }

      // Remove typing indicator
      typingEl.remove();

      // Create AI message bubble for streaming
      const { contentEl } = this.createAIMessageBubble();
      let fullContent = '';

      // Read SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);

          if (data === '[DONE]') break;

          try {
            const parsed = JSON.parse(data);
            if (parsed.error) throw new Error(parsed.error);
            if (parsed.content) {
              fullContent += parsed.content;
              this.renderMarkdown(contentEl, fullContent);
              this.scrollToBottom();
            }
          } catch (e) {
            if (e.message !== 'Unexpected end of JSON input') {
              console.warn('Parse error:', e);
            }
          }
        }
      }

      // Save AI response
      this.messages.push({ role: 'assistant', content: fullContent });

    } catch (error) {
      typingEl?.remove();
      this.showError(error.message);
    } finally {
      this.isStreaming = false;
      this.updateStatus('대기 중', 'online');
      this.updateSendButton();
    }
  }

  /** Render user message bubble */
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

  /** Create an AI message bubble (returns the content element for streaming) */
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

  /** Render markdown content */
  renderMarkdown(el, text) {
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
      });
      el.innerHTML = marked.parse(text);
    } else {
      el.innerHTML = this.escapeHtml(text).replace(/\n/g, '<br>');
    }
  }

  /** Show typing indicator */
  showTypingIndicator() {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message ai';
    div.id = 'typingMsg';
    div.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-content">
        <div class="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    `;
    container.appendChild(div);
    this.scrollToBottom();
    return div;
  }

  /** Show error message */
  showError(msg) {
    const container = document.getElementById('chatMessages');
    const div = document.createElement('div');
    div.className = 'message-error';
    div.textContent = `⚠️ ${msg}`;
    container.appendChild(div);
    this.scrollToBottom();
  }

  /** Hide welcome message */
  hideWelcome() {
    const welcome = document.querySelector('.welcome-message');
    if (welcome) welcome.style.display = 'none';
  }

  /** Update status indicator */
  updateStatus(text, type) {
    const statusEl = document.getElementById('chatStatus');
    if (!statusEl) return;
    statusEl.innerHTML = `<span class="status-dot ${type}"></span> ${text}`;
  }

  /** Update send button state */
  updateSendButton() {
    const btn = document.getElementById('sendBtn');
    if (btn) btn.disabled = this.isStreaming;
  }

  /** Scroll chat to bottom */
  scrollToBottom() {
    const container = document.getElementById('chatMessages');
    requestAnimationFrame(() => {
      container.scrollTop = container.scrollHeight;
    });
  }

  /** Clear all messages */
  clearChat() {
    this.messages = [];
    const container = document.getElementById('chatMessages');
    container.innerHTML = '';

    // Show welcome message again
    container.innerHTML = `
      <div class="welcome-message">
        <div class="ai-avatar"><span>◈</span></div>
        <h3>안녕하세요! AI 트레이딩 어시스턴트입니다</h3>
        <p>차트를 보면서 궁금한 점이나 분석이 필요한 부분을 물어보세요.<br>
        현재 <strong>BINANCE:${this.chartContext.symbol}</strong> 차트를 보고 있습니다.</p>
        <div class="quick-actions" id="quickActions">
          <button class="quick-btn" data-msg="현재 차트 기술적 분석을 해주세요">📊 차트 분석</button>
          <button class="quick-btn" data-msg="현재 매매 전략을 제안해주세요">💡 매매 전략</button>
          <button class="quick-btn" data-msg="주요 지지선과 저항선을 알려주세요">📐 지지/저항</button>
          <button class="quick-btn" data-msg="현재 RSI와 EMA 상태를 분석해주세요">📈 지표 분석</button>
        </div>
      </div>
    `;

    // Re-bind quick action events
    this.bindQuickActions();
  }

  /** Bind quick action button events */
  bindQuickActions() {
    document.querySelectorAll('.quick-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const msg = btn.getAttribute('data-msg');
        if (msg) this.sendMessage(msg);
      });
    });
  }

  /** Escape HTML to prevent XSS */
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Export as global
window.chatManager = new ChatManager();
