@echo off
cd /d C:\Users\prool\trading-bot
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe run_daily.py >> logs\scheduler.log 2>&1
