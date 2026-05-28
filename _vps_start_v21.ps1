Set-Location C:\Users\Administrator\tradingbote

# Kill V19/V20/V21 si tournent
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2

"" | Out-File live_v21_stdout.log -Encoding utf8
"" | Out-File live_v21_stderr.log -Encoding utf8

$wrapper = @'
@echo off
cd /d C:\Users\Administrator\tradingbote
set DASHBOARD_URL=https://tradingbote-production.up.railway.app/api/ingest
set RECENT_CUTOFF_MIN=60
set BOT_THRESHOLD=0.80
set RISK_PCT=0.05
python -u -m bot_v2.live_runner_v21 1>>C:\Users\Administrator\tradingbote\live_v21_stdout.log 2>>C:\Users\Administrator\tradingbote\live_v21_stderr.log
'@
$wrapper | Out-File C:\Users\Administrator\tradingbote\_run_v21.bat -Encoding ascii -Force

# Supprime ancienne taches
schtasks /Delete /TN "TradingBotV19" /F 2>$null
schtasks /Delete /TN "TradingBotV20" /F 2>$null
schtasks /Delete /TN "TradingBotV21" /F 2>$null

$startTime = (Get-Date).AddSeconds(30).ToString("HH:mm")
schtasks /Create /TN "TradingBotV21" /TR "C:\Users\Administrator\tradingbote\_run_v21.bat" /SC ONCE /ST $startTime /RU "Administrator" /F /IT
schtasks /Run /TN "TradingBotV21"

Start-Sleep -Seconds 3
Write-Output ""
Write-Output "=== Statut Python ==="
Get-Process python -ErrorAction SilentlyContinue | Format-Table Id, StartTime -AutoSize
Write-Output "=== Statut Task ==="
schtasks /Query /TN "TradingBotV21" /FO LIST
