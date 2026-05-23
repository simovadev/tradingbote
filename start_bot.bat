@echo off
REM Lance le bot live en dry-run, totalement detache (survit a la fermeture SSH).
REM Usage : start start_bot.bat   (ou Start-Process via PowerShell)
REM
REM BOT_WORKERS controle le ProcessPool. Sur VPS 4 coeurs, 4 = optimal
REM (8 workers degradent a 18s vs 14s en 4 workers, mesure 2026-05-23).
REM Override : set BOT_WORKERS=N avant d'appeler ce .bat.
if "%BOT_WORKERS%"=="" set BOT_WORKERS=4
cd /d "%~dp0"
python -u -m bot_v2.live_runner_v2 --dry-run > bot_stdout.log 2> bot_stderr.log
