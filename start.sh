#!/bin/bash
# MurpheyAI - One-Click Startup Script (macOS)

echo ""
echo "========================================"
echo "   MurpheyAI - One-Click Startup (mac)"
echo "========================================"
echo ""

# ----- Detect Python -----
echo "📋 Checking Python..."
if command -v python3 &>/dev/null; then
    PYTHON="python3"
elif command -v python &>/dev/null; then
    PYTHON="python"
else
    echo "❌ Python 3.11+ not found! Install using: brew install python"
    exit 1
fi
echo "✅ Python found: $($PYTHON --version)"

# ----- Detect Node -----
echo "📋 Checking Node.js..."
if ! command -v node &>/dev/null; then
    echo "❌ Node.js not found! Install using: brew install node"
    exit 1
fi
echo "✅ Node found: $(node --version)"

# ----- Detect Docker -----
echo "📋 Checking Docker..."
if ! command -v docker &>/dev/null; then
    echo "❌ Docker not found! Install Docker Desktop for Mac"
    exit 1
fi
echo "✅ Docker found"

echo ""
echo "🗄️  Starting databases (PostgreSQL, MongoDB, Redis)..."
cd deployment || exit

if command -v docker compose &>/dev/null; then
    DC="docker compose"
else
    DC="docker-compose"
fi

$DC up -d postgres mongodb redis

echo "⏳ Waiting 20 seconds for DB services..."
sleep 20

cd ..

# ----- Backend Setup -----
echo ""
echo "🔧 Setting up backend..."
cd backend || exit

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv venv
fi

echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install backend dependencies
if [ ! -d "venv/lib" ]; then
    echo "📦 Installing backend dependencies..."
    pip install -r requirements.txt
else
    echo "✅ Backend dependencies already installed"
fi

# Create .env if missing
if [ ! -f ".env" ]; then
    echo "📦 Creating backend .env..."
    cat <<EOT > .env
POSTGRES_URL=postgresql://murpheyai:password@localhost:5432/murpheyai
MONGODB_URL=mongodb://localhost:27017/murpheyai
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=dev-secret-key-$(date +%s)
JWT_SECRET_KEY=dev-jwt-secret-key-$(date +%s)
API_V1_STR=/api/v1
DEBUG=True
MODEL_NAME=small
MODEL_PATH=./models
MAX_TOKENS=2048
TEMPERATURE=0.7
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
EOT
fi

echo "📦 Running Alembic migrations..."
alembic upgrade head

echo "👤 Creating admin user..."
$PYTHON create_admin.py

BACKEND_PATH=$(pwd)
cd ..

# ----- Frontend Setup -----
echo ""
echo "🎨 Setting up frontend..."
cd frontend || exit

# Install dependencies
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Node dependencies..."
    npm install
else
    echo "✅ Node dependencies already installed"
fi

# Create .env.local
if [ ! -f ".env.local" ]; then
    echo "📦 Creating frontend .env.local..."
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
fi

FRONTEND_PATH=$(pwd)
cd ..

# ----- Start Services -----

echo ""
echo "🚀 Starting backend & frontend (macOS tabs)..."

# Open backend in new terminal tab
osascript <<EOT
tell application "Terminal"
    activate
    do script "cd '$BACKEND_PATH'; source venv/bin/activate; uvicorn app.main:app --reload"
end tell
EOT

# Wait a moment
sleep 3

# Open frontend
osascript <<EOT
tell application "Terminal"
    activate
    do script "cd '$FRONTEND_PATH'; npm run dev"
end tell
EOT

echo ""
echo "======================================"
echo " MurpheyAI is now running! 🎉"
echo "======================================"
echo "Frontend:  http://localhost:3000"
echo "Backend:   http://localhost:8000"
echo "Docs:      http://localhost:8000/docs"
echo ""
echo "Admin Login:"
echo "Email:    admin@test.com"
echo "Password: admin123"
echo ""
