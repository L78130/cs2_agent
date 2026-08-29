@echo off
rem demo_coach one-click launcher:
rem starts the server if it isn't running, then opens the web UI.
cd /d "%~dp0"

netstat -an | findstr /C:"127.0.0.1:8000" | findstr LISTENING >nul 2>&1
if errorlevel 1 (
  rem server not running -> start it minimized; it must not open its own tab
  set DEMO_COACH_NO_BROWSER=1
  start "demo_coach server" /min .venv\Scripts\python.exe -m uvicorn demo_coach.web.server:app --host 127.0.0.1 --port 8000
  rem wait until the port is listening (max ~15s)
  for /l %%i in (1,1,15) do (
    ping -n 2 127.0.0.1 >nul
    netstat -an | findstr /C:"127.0.0.1:8000" | findstr LISTENING >nul 2>&1 && goto :open
  )
)

:open
start "" http://127.0.0.1:8000
