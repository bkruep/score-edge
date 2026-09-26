@echo off
setlocal
title ScoreEdge server restart
cd /d "%~dp0"
echo [go] ScoreEdge - restarting uvicorn on 8512 ...
set LOGS=server.log
set ELOGS=server_err.log
for /f "tokens=2 delims==" %%V in ('wmic process where "CommandLine like '%%uvicorn%%app:app%%8512%%'" get processid /value 2^>nul') do (
  taskkill /f /pid %%V >nul 2>&1
)
for /f "tokens=2 delims==" %%V in ('wmic process where "CommandLine like '%%8512%%'" get processid /value 2^>nul') do (
  taskkill /f /pid %%V >nul 2>&1
)
timeout /t 2 /nobreak >nul
start "" /b "%~dp0.venv\Scripts\pythonw.exe" -m uvicorn app:app --host 127.0.0.1 --port 8512 >>"%LOGS%" 2>>"%ELOGS%"
echo   waiting for 127.0.0.1:8512 ...
set UP=
for /L %%i in (1,1,30) do (
  powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8512 -State Listen -EA SilentlyContinue) { exit 0 }" >nul 2>&1
  if not errorlevel 1 set UP=1
  if defined UP goto :ready
  timeout /t 1 /nobreak >nul
)
:ready
if not defined UP (
  echo   NOT UP - last log lines:
  type "%ELOGS%" 2>nul | findstr /n "^" | findstr /b "[1-9][0-9]*:" | findstr /n "." | findstr /b "[1-9][0-9]*:" | more +0
  goto :end
)
echo   listening - probing /api/slate ...
powershell -NoProfile -Command "$r=Invoke-WebRequest -Uri 'http://127.0.0.1:8512/api/slate' -UseBasicParsing -TimeoutSec 6; $j=$r.Content|ConvertFrom-Json; Write-Host ('ok='+$j.ok+' rows='+$j.rows+' map.rows='+$j.map.rows+' slots='+$j.map.slots+' radar.axes='+$j.radar.axes.Count)"
goto :end
:end
choice /c YN /n /m "Open the dashboard now? [Y/N] " >nul
if not errorlevel 2 start "" http://127.0.0.1:8512/
endlocal
