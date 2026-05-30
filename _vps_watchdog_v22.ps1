# =============================================================
# V22 Watchdog - Monitore le bot, restart auto, notif Telegram
# =============================================================
# Logique :
#   Toutes les 30s :
#     1. Verifie qu'un python.exe v22_live_runner tourne
#     2. Verifie que v22_live.log a ete ecrit dans les 5 dernieres minutes
#        (= bot vivant, pas hang)
#     3. Si KO -> kill l'ancien + relance le bot + envoie notif Telegram
#
# Usage : Start-Process powershell -ArgumentList "-File","_vps_watchdog_v22.ps1" -WindowStyle Hidden
# Stop  : Get-Process powershell | Where-Object { $_.MainWindowTitle -like "*watchdog*" } | Stop-Process

$ErrorActionPreference = "Continue"
$BotDir         = "C:\Users\Administrator\tradingbote"
$BotModule      = "bot_v2.v22_live_runner"
$BotLogFile     = "$BotDir\v22_live.log"
$WatchdogLog    = "$BotDir\v22_watchdog.log"
$DashboardUrl   = "https://tradingbote-production.up.railway.app/api/notify/watchdog"

# Env vars pour le bot
$env:VANTAGE_LOGIN    = "25451501"
$env:VANTAGE_SERVER   = "VantageMarkets-Demo"
$env:INITIAL_BALANCE  = "1000"

# Seuil log inactif (5 min = bot probablement hang)
$LogInactiveSeconds   = 300
# Periodicite check
$CheckIntervalSeconds = 30
# Notif heartbeat toutes les N heures (pour savoir que watchdog tourne)
$HeartbeatIntervalH   = 12

function Write-WLog($msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  $line = "$ts $msg"
  Add-Content -Path $WatchdogLog -Value $line -ErrorAction SilentlyContinue
  Write-Host $line
}

function Send-Notify($text, $level = "info") {
  try {
    $body = @{ text = $text; level = $level } | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri $DashboardUrl -Method Post -Body $body `
      -ContentType "application/json" -TimeoutSec 10 -ErrorAction Stop | Out-Null
    Write-WLog "  notif envoyee ($level)"
  } catch {
    Write-WLog "  notif fail : $($_.Exception.Message)"
  }
}

function Get-BotProcess {
  # Cherche python.exe avec la commande line contenant v22_live_runner
  $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue
  foreach ($p in $procs) {
    if ($p.CommandLine -and $p.CommandLine -like "*v22_live_runner*") {
      return @{ Pid = $p.ProcessId; Start = $p.CreationDate; CmdLine = $p.CommandLine }
    }
  }
  return $null
}

function Get-LogAgeSeconds {
  if (-not (Test-Path $BotLogFile)) { return [int]::MaxValue }
  $last = (Get-Item $BotLogFile).LastWriteTime
  return [int]((Get-Date) - $last).TotalSeconds
}

function Start-Bot {
  Write-WLog "  Demarrage du bot..."
  Set-Location $BotDir
  $stdoutF = "$BotDir\v22_live_stdout.log"
  $stderrF = "$BotDir\v22_live_stderr.log"
  # Truncate logs
  "" | Out-File -FilePath $stdoutF -Encoding utf8 -Force
  "" | Out-File -FilePath $stderrF -Encoding utf8 -Force
  $p = Start-Process python `
    -ArgumentList "-u","-m",$BotModule `
    -RedirectStandardOutput $stdoutF `
    -RedirectStandardError $stderrF `
    -WindowStyle Hidden `
    -PassThru
  Write-WLog "  Bot lance, PID=$($p.Id)"
  return $p.Id
}

function Stop-Bot {
  $proc = Get-BotProcess
  if ($proc) {
    Write-WLog "  Kill process PID=$($proc.Pid)"
    try { Stop-Process -Id $proc.Pid -Force -ErrorAction Stop } catch {}
    Start-Sleep -Seconds 3
  }
}

# =============================================================
# MAIN LOOP
# =============================================================

Write-WLog "============================================================"
Write-WLog "V22 Watchdog demarre (PID watchdog = $PID)"
Write-WLog "  Bot dir       : $BotDir"
Write-WLog "  Bot log       : $BotLogFile"
Write-WLog "  Notify URL    : $DashboardUrl"
Write-WLog "  Check every   : $CheckIntervalSeconds s"
Write-WLog "  Log inactive  : $LogInactiveSeconds s (= bot mort/hang)"
Write-WLog "============================================================"

Send-Notify "Watchdog V22 demarre sur VPS Contabo (PID $PID)" "info"

$lastHeartbeat   = Get-Date
$lastRestart     = $null
$consecutiveRestarts = 0

while ($true) {
  try {
    $proc    = Get-BotProcess
    $logAge  = Get-LogAgeSeconds
    $now     = Get-Date

    # Cas 1 : pas de process
    if (-not $proc) {
      Write-WLog "[KO] Aucun process bot python detecte"
      $msg = "Le bot V22 ne tourne plus sur le VPS. Redemarrage auto..."
      Send-Notify $msg "error"
      $newPid = Start-Bot
      Start-Sleep -Seconds 20  # laisse le temps de demarrer + init MT5
      $lastRestart = $now
      $consecutiveRestarts++
      if ($consecutiveRestarts -ge 3) {
        Send-Notify "Le bot V22 a deja crash $consecutiveRestarts fois en peu de temps. Probleme persistant a investiguer manuellement." "error"
      }
    }
    # Cas 2 : process tourne mais log fige depuis > 5 min => hang
    elseif ($logAge -gt $LogInactiveSeconds) {
      Write-WLog "[HANG] Log fige depuis $logAge s alors que process PID=$($proc.Pid) tourne -> kill+restart"
      Send-Notify "Le bot V22 est freeze (log inactif depuis ${logAge}s, process PID $($proc.Pid)). Kill + restart..." "warn"
      Stop-Bot
      Start-Sleep -Seconds 5
      $newPid = Start-Bot
      Start-Sleep -Seconds 20
      $lastRestart = $now
      $consecutiveRestarts++
    }
    # Cas OK
    else {
      # Reset compteur si on a tenu 30 min sans restart
      if ($lastRestart -and (($now - $lastRestart).TotalMinutes -ge 30)) {
        if ($consecutiveRestarts -gt 0) {
          Write-WLog "  OK 30min sans crash, reset compteur (etait $consecutiveRestarts)"
          $consecutiveRestarts = 0
        }
      }
      # Heartbeat toutes les N heures
      if (($now - $lastHeartbeat).TotalHours -ge $HeartbeatIntervalH) {
        $hours = [math]::Round((($now - $proc.Start).TotalHours), 1)
        Send-Notify "Bot V22 OK (PID $($proc.Pid), uptime ${hours}h, log age ${logAge}s)" "info"
        $lastHeartbeat = $now
      }
      # Log discret seulement toutes les 10 min pour pas spammer
      if ((Get-Random -Maximum 20) -eq 0) {
        Write-WLog "[OK] Bot PID=$($proc.Pid), log age $logAge s"
      }
    }
  } catch {
    Write-WLog "[ERR watchdog loop] $($_.Exception.Message)"
  }
  Start-Sleep -Seconds $CheckIntervalSeconds
}
