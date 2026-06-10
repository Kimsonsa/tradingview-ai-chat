' TradeAI Assistant Launcher (명령창 숨김)
' 1) Streamlit 서버를 숨김 창으로 시작 (이미 떠 있으면 포트 충돌로 죽고 기존 서버 사용)
' 2) curl 헬스체크가 성공할 때까지 대기 (최대 60초) — 고정 sleep이면 콜드스타트가
'    길 때 "연결을 거부했습니다" 빈 창이 뜸
' 3) 브라우저 열기
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

WshShell.Run "cmd /c streamlit run assistant.py --server.port 8502", 0, False

' 서버 응답 대기 (동기 실행 — 성공 시 즉시 종료, 실패 시 1초 간격 재시도)
WshShell.Run "cmd /c for /L %i in (1,1,60) do curl -s -o nul --max-time 1 http://localhost:8502/_stcore/health && exit 0 || timeout /t 1 /nobreak >nul", 0, True

WshShell.Run "http://localhost:8502"
