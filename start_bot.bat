@echo off
REM Lance le bot live en dry-run, totalement detache (survit a la fermeture SSH).
REM Usage : start start_bot.bat   (ou Start-Process via PowerShell)
cd /d "%~dp0"
python -u -m bot_v2.live_runner_v2 --dry-run > bot_stdout.log 2> bot_stderr.log
