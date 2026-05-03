$ws = New-Object -ComObject WScript.Shell
$desktop = [Environment]::GetFolderPath('Desktop')
$shortcut = $ws.CreateShortcut("$desktop\TradeAI Assistant.lnk")
$shortcut.TargetPath = "c:\projects\tradingview-ai-chat\TradeAI.vbs"
$shortcut.WorkingDirectory = "c:\projects\tradingview-ai-chat"
$shortcut.Description = "TradeAI Assistant"
$shortcut.Save()
Write-Host "Desktop shortcut created!"
