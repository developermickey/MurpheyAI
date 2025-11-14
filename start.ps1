# MurpheyAI - One-Click Startup Script (PowerShell)
# This script sets up and starts the entire project

Write-Host "🚀 Starting MurpheyAI..." -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "📋 Checking prerequisites..." -ForegroundColor Yellow

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found! Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check Node.js
try {
    $nodeVersion = node --version 2>&1
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found! Please install Node.js 20+" -ForegroundColor Red
    exit 1
}

# Check Docker
try {
    $dockerVersion = docker --version 2>&1
    Write-Host "✅ Docker found: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker not found! Please install Docker Desktop" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 1: Start Databases
Write-Host "🗄️  Starting databases (PostgreSQL, MongoDB, Redis)..." -ForegroundColor Yellow
Set-Location deployment

# Check which docker compose command works
$dockerComposeCmd = "docker-compose"
try {
    docker-compose --version | Out-Null
} catch {
    $dockerComposeCmd = "docker compose"
}

& $dockerComposeCmd up -d postgres mongodb redis
Write-Host "⏳ Waiting for databases to initialize (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30
Set-Location ..

# Step 2: Setup Backend
Write-Host ""
Write-Host "🔧 Setting up backend..." -ForegroundColor Yellow
Set-Location backend

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Install dependencies
if (-not (Test-Path "venv\Lib\site-packages\fastapi")) {
    Write-Host "Installing Python dependencies (this may take a few minutes)..." -ForegroundColor Cyan
    pip install -r requirements.txt
} else {
    Write-Host "✅ Python dependencies already installed" -ForegroundColor Green
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Cyan
    $random1 = Get-Random
    $random2 = Get-Random
    $envLines = @(
        "# Database",
        "POSTGRES_URL=postgresql://murpheyai:password@localhost:5432/murpheyai",
        "MONGODB_URL=mongodb://localhost:27017/murpheyai",
        "REDIS_URL=redis://localhost:6379/0",
        "",
        "# Security",
        "SECRET_KEY=dev-secret-key-change-in-production-$random1",
        "JWT_SECRET_KEY=dev-jwt-secret-key-change-in-production-$random2",
        "",
        "# API",
        "API_V1_STR=/api/v1",
        "DEBUG=True",
        "",
        "# Model",
        "MODEL_NAME=small",
        "MODEL_PATH=./models",
        "MAX_TOKENS=2048",
        "TEMPERATURE=0.7",
        "",
        "# CORS",
        "CORS_ORIGINS=http://localhost:3000,http://localhost:3001"
    )
    $envLines | Out-File -FilePath .env -Encoding utf8
    Write-Host "Created .env file" -ForegroundColor Green
}

# Initialize database
Write-Host "Initializing database..." -ForegroundColor Cyan
alembic upgrade head

# Create admin user if it doesn't exist
Write-Host "Creating admin user..." -ForegroundColor Cyan
python create_admin.py

Set-Location ..

# Step 3: Setup Frontend
Write-Host ""
Write-Host "🎨 Setting up frontend..." -ForegroundColor Yellow
Set-Location frontend

# Install dependencies if node_modules doesn't exist
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing Node.js dependencies (this may take a few minutes)..." -ForegroundColor Cyan
    npm install
} else {
    Write-Host "✅ Node.js dependencies already installed" -ForegroundColor Green
}

# Create .env.local if it doesn't exist
if (-not (Test-Path ".env.local")) {
    Write-Host "Creating .env.local file..." -ForegroundColor Cyan
    "NEXT_PUBLIC_API_URL=http://localhost:8000" | Out-File -FilePath .env.local -Encoding utf8
    Write-Host "✅ Created .env.local file" -ForegroundColor Green
}

Set-Location ..

# Step 4: Start Services
Write-Host ""
Write-Host "🚀 Starting services..." -ForegroundColor Yellow
Write-Host ""
Write-Host "📝 Starting backend on http://localhost:8000" -ForegroundColor Cyan
Write-Host "🎨 Starting frontend on http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  Two terminal windows will open. Keep them open!" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press any key to start services..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Start backend in new window
Start-Process powershell -ArgumentList @"
-cd '$PWD\backend'; .\venv\Scripts\Activate.ps1; Write-Host '🚀 Backend starting on http://localhost:8000' -ForegroundColor Green; uvicorn app.main:app --reload
"@

# Wait a bit for backend to start
Start-Sleep -Seconds 5

# Start frontend in new window
Start-Process powershell -ArgumentList @"
-cd '$PWD\frontend'; Write-Host '🎨 Frontend starting on http://localhost:3000' -ForegroundColor Green; npm run dev
"@

Write-Host ""
Write-Host "✅ Services starting!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access your application:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "   Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "👤 Login credentials:" -ForegroundColor Cyan
Write-Host "   Email: admin@test.com" -ForegroundColor White
Write-Host "   Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "⏹️  To stop services:" -ForegroundColor Yellow
Write-Host "   1. Close the two PowerShell windows" -ForegroundColor White
Write-Host "   2. Run: cd deployment && docker-compose down" -ForegroundColor White
Write-Host ""

