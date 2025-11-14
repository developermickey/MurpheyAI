@echo off
REM MurpheyAI - One-Click Startup Script (Batch File)
REM This script sets up and starts the entire project

echo.
echo ========================================
echo   MurpheyAI - One-Click Startup
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: PowerShell is required but not found!
    pause
    exit /b 1
)

REM Run the PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0start.ps1"

pause

