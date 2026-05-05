' TradeAI PWA Launcher (명령창 숨김)
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
WshShell.Run "cmd /c python serve.py", 0, False
WScript.Sleep 1500
WshShell.Run "http://localhost:3000"
