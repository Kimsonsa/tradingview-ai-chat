@echo off
title TradeAI PWA
cd /d "%~dp0"

:: Streamlit 앱이 이미 실행 중인지 확인
netstat -ano | findstr :8502 >nul 2>&1
if %errorlevel% neq 0 (
    echo Starting Streamlit...
    start /b cmd /c "streamlit run assistant.py --server.port 8502 --server.headless true"
    timeout /t 3 /nobreak >nul
)

:: PWA 셸 서버 시작
echo Starting PWA shell...
start /b cmd /c "python serve.py"
timeout /t 1 /nobreak >nul

:: 브라우저 열기
start http://localhost:3000

echo.
echo  TradeAI is running!
echo  PWA Shell: http://localhost:3000
echo  Streamlit: http://localhost:8502
echo.
echo  Press any key to stop...
pause >nul

:: 종료 시 프로세스 정리
taskkill /f /im python.exe /fi "WINDOWTITLE eq TradeAI*" >nul 2>&1
