@echo off
title TradeAI Assistant
cd /d "%~dp0"

:: 서버 시작 (이미 떠 있으면 포트 충돌로 조용히 죽고 기존 서버 사용)
start /b cmd /c "streamlit run assistant.py --server.port 8502"

:: 서버가 실제로 응답할 때까지 대기 (최대 60초) 후 브라우저 열기
set /a tries=0
:wait
set /a tries+=1
if %tries% gtr 60 goto open
curl -s -o nul --max-time 1 http://localhost:8502/_stcore/health 2>nul
if %errorlevel% neq 0 (
    timeout /t 1 /nobreak >nul
    goto wait
)
:open
start http://localhost:8502
