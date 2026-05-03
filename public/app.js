/* ============================================
   App Module — Main application logic
   ============================================ */

(function () {
  // ---- State ----
  const state = {
    symbol: 'BINANCE:BTCUSDT.P',
    interval: '60',
    hasApiKey: false
  };

  const INTERVAL_LABELS = {
    '1': '1분', '5': '5분', '15': '15분',
    '60': '1시간', '240': '4시간', 'D': '1일'
  };

  // ---- TradingView Widget ----
  function createWidget() {
    const container = document.getElementById('tvWidgetContainer');
    container.innerHTML = '';

    const wrapper = document.createElement('div');
    wrapper.className = 'tradingview-widget-container';
    wrapper.style.cssText = 'height:100%;width:100%';

    const inner = document.createElement('div');
    inner.className = 'tradingview-widget-container__widget';
    inner.style.cssText = 'height:100%;width:100%';

    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    script.textContent = JSON.stringify({
      autosize: true,
      symbol: state.symbol,
      interval: state.interval,
      timezone: 'Asia/Seoul',
      theme: 'dark',
      style: '1',
      locale: 'ko',
      backgroundColor: 'rgba(8, 11, 18, 1)',
      gridColor: 'rgba(30, 40, 60, 0.25)',
      allow_symbol_change: false,
      calendar: false,
      hide_top_toolbar: false,
      hide_legend: false,
      save_image: false,
      studies: [
        'MAExp@tv-basicstudies',
        'MAExp@tv-basicstudies',
        'MAExp@tv-basicstudies',
        'RSI@tv-basicstudies',
        'Volume@tv-basicstudies'
      ],
      support_host: 'https://www.tradingview.com'
    });

    wrapper.appendChild(inner);
    wrapper.appendChild(script);
    container.appendChild(wrapper);
  }

  function changeInterval(interval) {
    state.interval = interval;
    createWidget();
    updateContextUI();

    // Update active button
    document.querySelectorAll('.tf-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.interval === interval);
    });

    // Update chat context
    window.chatManager.setContext(
      state.symbol.replace('BINANCE:', ''),
      INTERVAL_LABELS[interval] || interval
    );
  }

  function updateContextUI() {
    const sym = state.symbol.replace('BINANCE:', '');
    const tf = INTERVAL_LABELS[state.interval] || state.interval;

    const contextSymbol = document.getElementById('contextSymbol');
    const contextTf = document.getElementById('contextTf');
    if (contextSymbol) contextSymbol.textContent = sym;
    if (contextTf) contextTf.textContent = tf;
  }

  // ---- API Key Management ----
  async function checkApiKey() {
    try {
      const res = await fetch('/api/check-key');
      const data = await res.json();
      state.hasApiKey = data.hasKey;
      if (!data.hasKey) showNoKeyBanner();
    } catch (e) {
      console.error('Key check failed:', e);
    }
  }

  function showNoKeyBanner() {
    const existing = document.querySelector('.no-key-banner');
    if (existing) return;

    const banner = document.createElement('div');
    banner.className = 'no-key-banner';
    banner.textContent = '🔑 AI 채팅을 사용하려면 OpenAI API 키를 설정하세요. 클릭하여 설정';
    banner.addEventListener('click', () => openSettings());

    const chatMessages = document.getElementById('chatMessages');
    chatMessages.parentNode.insertBefore(banner, chatMessages);
  }

  function removeNoKeyBanner() {
    const banner = document.querySelector('.no-key-banner');
    if (banner) banner.remove();
  }

  async function saveApiKey() {
    const keyInput = document.getElementById('apiKeyInput');
    const modelInput = document.getElementById('modelInput');
    const statusEl = document.getElementById('keyStatus');
    const key = keyInput.value.trim();

    if (!key) {
      statusEl.className = 'key-status error';
      statusEl.textContent = 'API 키를 입력하세요.';
      statusEl.style.display = 'block';
      return;
    }

    try {
      const res = await fetch('/api/set-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          key,
          model: modelInput.value
        })
      });

      const data = await res.json();
      if (data.success) {
        state.hasApiKey = true;
        statusEl.className = 'key-status success';
        statusEl.textContent = `✅ API 키가 설정되었습니다. (모델: ${data.model})`;
        statusEl.style.display = 'block';
        removeNoKeyBanner();
        setTimeout(() => closeSettings(), 1500);
      }
    } catch (e) {
      statusEl.className = 'key-status error';
      statusEl.textContent = `❌ 설정 실패: ${e.message}`;
      statusEl.style.display = 'block';
    }
  }

  // ---- Modal ----
  function openSettings() {
    document.getElementById('settingsModal').classList.add('open');
  }

  function closeSettings() {
    document.getElementById('settingsModal').classList.remove('open');
  }

  // ---- Resize Handle ----
  function initResize() {
    const handle = document.getElementById('resizeHandle');
    const chatPanel = document.getElementById('chatPanel');
    let isResizing = false;

    handle.addEventListener('mousedown', (e) => {
      isResizing = true;
      handle.classList.add('dragging');
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
      if (!isResizing) return;
      const newWidth = window.innerWidth - e.clientX;
      const clamped = Math.max(300, Math.min(600, newWidth));
      chatPanel.style.width = clamped + 'px';
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        handle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
    });
  }

  // ---- Chat Input ----
  function initChatInput() {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    // Auto-resize textarea
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    });

    // Send on Enter (Shift+Enter for newline)
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendCurrentMessage();
      }
    });

    sendBtn.addEventListener('click', () => sendCurrentMessage());
  }

  function sendCurrentMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;

    if (!state.hasApiKey) {
      openSettings();
      return;
    }

    input.value = '';
    input.style.height = 'auto';
    window.chatManager.sendMessage(text);
  }

  // ---- Event Bindings ----
  function bindEvents() {
    // Timeframe buttons
    document.querySelectorAll('.tf-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        changeInterval(btn.dataset.interval);
      });
    });

    // Settings
    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('modalClose').addEventListener('click', closeSettings);
    document.getElementById('saveKeyBtn').addEventListener('click', saveApiKey);

    // Close modal on overlay click
    document.getElementById('settingsModal').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeSettings();
    });

    // Clear chat
    document.getElementById('clearChatBtn').addEventListener('click', () => {
      window.chatManager.clearChat();
    });

    // Quick actions
    window.chatManager.bindQuickActions();

    // Escape key closes modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeSettings();
    });
  }

  // ---- Init ----
  function init() {
    createWidget();
    updateContextUI();
    initResize();
    initChatInput();
    bindEvents();
    checkApiKey();

    // Set initial chat context
    window.chatManager.setContext('BTCUSDT.P', '1시간');
  }

  // Run when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
