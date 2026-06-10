@echo off
title TradeAI Assistant
cd /d "%~dp0"

:: Start server in its own minimized window so it survives this launcher closing.
:: (If already running, the new instance dies on port conflict and we reuse it.)
start "TradeAI Server" /min cmd /c "streamlit run assistant.py --server.port 8502"

:: Wait until the server actually responds (max 60s), then open the browser.
for /L %%i in (1,1,60) do (
    curl -s -o nul --max-time 1 http://localhost:8502/_stcore/health && goto open
    timeout /t 1 /nobreak >nul
)
:open
start http://localhost:8502
