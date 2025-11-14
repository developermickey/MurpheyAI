@echo off
REM MurpheyAI - Stop Script

echo.
echo ========================================
echo   Stopping MurpheyAI Services
echo ========================================
echo.

echo Stopping Docker containers...
cd deployment
docker-compose down
cd ..

echo.
echo ✅ All services stopped!
echo.
pause

