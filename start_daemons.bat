@echo off
REM Lance les 14 cache_daemons en parallele + auto-restart, detache.
cd /d "%~dp0"
python -u launch_daemons.py > launcher.log 2> launcher.err
