# MurpheyAI - Stop Script (PowerShell)

Write-Host ""
Write-Host "🛑 Stopping MurpheyAI services..." -ForegroundColor Yellow
Write-Host ""

# Stop Docker containers
Set-Location deployment

# Check which docker compose command works
$dockerComposeCmd = "docker-compose"
try {
    docker-compose --version | Out-Null 2>&1
} catch {
    $dockerComposeCmd = "docker compose"
}

& $dockerComposeCmd down
Set-Location ..

Write-Host ""
Write-Host "✅ All services stopped!" -ForegroundColor Green
Write-Host ""
Write-Host "💡 Note: If backend/frontend are running in separate windows, close them manually." -ForegroundColor Yellow
Write-Host ""

