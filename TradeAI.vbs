' TradeAI Assistant Launcher (명령창 숨김)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c streamlit run assistant.py --server.port 8502", 0, False
WScript.Sleep 2000
WshShell.Run "http://localhost:8502"
