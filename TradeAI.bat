@echo off
title TradeAI Assistant
cd /d "%~dp0"

:: 서버를 별도 창(최소화)으로 시작 — 이 런처가 닫혀도 서버는 유지
:: (이미 떠 있으면 포트 충돌로 죽고 기존 서버를 그대로 사용)
start "TradeAI Server" /min cmd /c "streamlit run assistant.py --server.port 8502"

:: 서버가 실제로 응답할 때까지 대기 (최대 60초) 후 브라우저 열기
for /L %%i in (1,1,60) do (
    curl -s -o nul --max-time 1 http://localhost:8502/_stcore/health && goto open
    timeout /t 1 /nobreak >nul
)
:open
start http://localhost:8502
