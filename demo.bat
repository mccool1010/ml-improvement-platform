@echo off
rem Double-click or run `demo` from C:\mlops. Arguments pass through, e.g.
rem   demo -Retrain -Failure -Traffic 200
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo.ps1" %*
if errorlevel 1 pause
