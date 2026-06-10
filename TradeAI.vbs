' TradeAI Assistant Launcher (hidden console)
' 1) Start the Streamlit server hidden. If one is already running, the new
'    instance dies on port conflict and the existing server is reused.
' 2) Poll the health endpoint via curl until it responds (max 60s) -- a fixed
'    sleep showed a "connection refused" blank window on slow cold starts.
' 3) Open the browser.
' NOTE: comments are ASCII on purpose -- cmd/wscript read these files in the
'       ANSI codepage, and UTF-8 Korean comments corrupted the .bat variant.
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

WshShell.Run "cmd /c streamlit run assistant.py --server.port 8502", 0, False

' Synchronous wait: exits as soon as the health check succeeds.
WshShell.Run "cmd /c for /L %i in (1,1,60) do curl -s -o nul --max-time 1 http://localhost:8502/_stcore/health && exit 0 || timeout /t 1 /nobreak >nul", 0, True

WshShell.Run "http://localhost:8502"
