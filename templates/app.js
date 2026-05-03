/* ============================================
   App Module — Chart, Resize, Settings, Persistence
   ============================================ */
(function () {
  const INTERVAL_LABELS = {
    '1':'1분','5':'5분','15':'15분','60':'1시간','240':'4시간','D':'1일'
  };

  const state = {
    symbol: localStorage.getItem('tradeai_symbol') || 'BINANCE:BTCUSDT.P',
    interval: localStorage.getItem('tradeai_interval') || '60',
    chatWidth: parseInt(localStorage.getItem('tradeai_chatwidth')) || 380
  };

  /* ---- TradingView Widget ---- */
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

  /* ---- Timeframe (AI 분석용 컨텍스트만 변경, 차트 재생성 안함) ---- */
  function changeInterval(interval) {
    state.interval = interval;
    localStorage.setItem('tradeai_interval', interval);
    // 차트는 재생성하지 않음 — TradingView 내장 툴바로 변경
    updateContextUI();
    document.querySelectorAll('.tf-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.interval === interval);
    });
    window.chatManager.setContext(
      state.symbol.replace('BINANCE:', ''),
      interval,
      INTERVAL_LABELS[interval] || interval
    );
  }

  function updateContextUI() {
    const sym = state.symbol.replace('BINANCE:', '');
    const tf = INTERVAL_LABELS[state.interval] || state.interval;
    const cs = document.getElementById('contextSymbol');
    const ct = document.getElementById('contextTf');
    const st = document.getElementById('symbolText');
    if (cs) cs.textContent = sym;
    if (ct) ct.textContent = tf;
    if (st) st.textContent = sym;
  }

  /* ---- Resize ---- */
  function initResize() {
    const handle = document.getElementById('resizeHandle');
    const chatPanel = document.getElementById('chatPanel');
    let isResizing = false;

    chatPanel.style.width = state.chatWidth + 'px';

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
      state.chatWidth = clamped;
    });

    document.addEventListener('mouseup', () => {
      if (isResizing) {
        isResizing = false;
        handle.classList.remove('dragging');
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        localStorage.setItem('tradeai_chatwidth', state.chatWidth);
      }
    });
  }

  /* ---- Settings Modal ---- */
  function openSettings() {
    const modal = document.getElementById('settingsModal');
    const keyInput = document.getElementById('apiKeyInput');
    const modelInput = document.getElementById('modelInput');
    const saved = localStorage.getItem('tradeai_apikey');
    if (saved) keyInput.value = saved;
    const savedModel = localStorage.getItem('tradeai_model') || 'gpt-5.5';
    modelInput.value = savedModel;
    modal.classList.add('open');
  }

  function closeSettings() {
    document.getElementById('settingsModal').classList.remove('open');
    document.getElementById('keyStatus').style.display = 'none';
  }

  function saveSettings() {
    const key = document.getElementById('apiKeyInput').value.trim();
    const model = document.getElementById('modelInput').value;
    const statusEl = document.getElementById('keyStatus');

    if (!key) {
      statusEl.className = 'key-status error';
      statusEl.textContent = 'API 키를 입력하세요.';
      statusEl.style.display = 'block';
      return;
    }

    localStorage.setItem('tradeai_apikey', key);
    localStorage.setItem('tradeai_model', model);
    statusEl.className = 'key-status success';
    statusEl.textContent = '✅ 저장 완료 (모델: ' + model + ')';
    statusEl.style.display = 'block';

    const banner = document.querySelector('.no-key-banner');
    if (banner) banner.remove();

    setTimeout(() => closeSettings(), 1200);
  }

  /* ---- Chat Input ---- */
  function initChatInput() {
    const input = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');

    input.addEventListener('input', () => {
      input.style.height = 'auto';
      input.style.height = Math.min(input.scrollHeight, 100) + 'px';
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendCurrent();
      }
    });

    sendBtn.addEventListener('click', () => sendCurrent());
  }

  function sendCurrent() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    if (!localStorage.getItem('tradeai_apikey')) {
      openSettings();
      return;
    }
    input.value = '';
    input.style.height = 'auto';
    window.chatManager.sendMessage(text);
  }

  /* ---- No-key banner ---- */
  function checkApiKey() {
    if (!localStorage.getItem('tradeai_apikey')) {
      const banner = document.createElement('div');
      banner.className = 'no-key-banner';
      banner.textContent = '🔑 AI 채팅을 사용하려면 OpenAI API 키를 설정하세요.';
      banner.addEventListener('click', () => openSettings());
      const chatMessages = document.getElementById('chatMessages');
      chatMessages.parentNode.insertBefore(banner, chatMessages);
    }
  }

  /* ---- Event Bindings ---- */
  function bindEvents() {
    document.querySelectorAll('.tf-btn').forEach(btn => {
      btn.addEventListener('click', () => changeInterval(btn.dataset.interval));
    });

    document.getElementById('settingsBtn').addEventListener('click', openSettings);
    document.getElementById('modalClose').addEventListener('click', closeSettings);
    document.getElementById('saveKeyBtn').addEventListener('click', saveSettings);

    document.getElementById('settingsModal').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeSettings();
    });

    document.getElementById('clearChatBtn').addEventListener('click', () => {
      window.chatManager.clearChat();
    });

    window.chatManager.bindQuickActions();

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeSettings();
    });
  }

  /* ---- Init ---- */
  function init() {
    // Restore active timeframe button
    document.querySelectorAll('.tf-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.interval === state.interval);
    });

    createWidget();
    updateContextUI();
    initResize();
    initChatInput();
    bindEvents();
    checkApiKey();

    window.chatManager.setContext(
      state.symbol.replace('BINANCE:', ''),
      state.interval,
      INTERVAL_LABELS[state.interval] || state.interval
    );

    // Restore chat history
    window.chatManager.loadChat();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
