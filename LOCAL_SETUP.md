# 🚀 One-Click Local Setup Guide

## Quick Start (Windows)

### Option 1: Double-Click Start (Easiest)

1. **Double-click `start.bat`** in the project root
2. Wait for setup to complete (~5-10 minutes first time)
3. Two windows will open automatically:
   - Backend server (http://localhost:8000)
   - Frontend server (http://localhost:3000)
4. Open your browser to **http://localhost:3000**

**Login:**
- Email: `admin@test.com`
- Password: `admin123`

### Option 2: PowerShell Script

1. Right-click `start.ps1` → "Run with PowerShell"
2. Or open PowerShell and run:
   ```powershell
   .\start.ps1
   ```

### Option 3: Manual Setup

If the scripts don't work, follow the manual steps in `QUICKSTART.md`

---

## What the Script Does

The `start.ps1` script automatically:

1. ✅ Checks prerequisites (Python, Node.js, Docker)
2. ✅ Starts databases (PostgreSQL, MongoDB, Redis) via Docker
3. ✅ Creates Python virtual environment
4. ✅ Installs backend dependencies
5. ✅ Creates `.env` file with default settings
6. ✅ Initializes database with migrations
7. ✅ Creates admin user
8. ✅ Installs frontend dependencies
9. ✅ Creates `.env.local` file
10. ✅ Starts backend server (new window)
11. ✅ Starts frontend server (new window)

**Total time:** ~5-10 minutes (first time), ~1 minute (subsequent runs)

---

## Stopping Services

### Option 1: Double-Click Stop
- Double-click `stop.bat`

### Option 2: PowerShell
```powershell
.\stop.ps1
```

### Option 3: Manual
1. Close the two PowerShell windows (backend & frontend)
2. Run: `cd deployment && docker-compose down`

---

## Prerequisites

Before running, make sure you have:

- ✅ **Python 3.11+** - [Download](https://www.python.org/downloads/)
- ✅ **Node.js 20+** - [Download](https://nodejs.org/)
- ✅ **Docker Desktop** - [Download](https://www.docker.com/products/docker-desktop/)
- ✅ **Git** (optional, for cloning)

### Verify Installation

```powershell
python --version    # Should show 3.11+
node --version      # Should show v20+
docker --version    # Should show Docker version
```

---

## Troubleshooting

### Script Won't Run

**Issue:** "Execution policy" error
**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Issue:** Script opens and closes immediately
**Solution:** Run from PowerShell:
```powershell
.\start.ps1
```

### Docker Issues

**Issue:** "Docker is not running"
**Solution:** 
1. Open Docker Desktop
2. Wait for it to start (whale icon in system tray)
3. Run the script again

**Issue:** Port already in use
**Solution:**
- Check if services are already running: `docker ps`
- Stop existing containers: `cd deployment && docker-compose down`

### Backend Won't Start

**Issue:** "Module not found"
**Solution:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Issue:** "Database connection error"
**Solution:**
- Wait 30-60 seconds after starting Docker
- Check databases are running: `docker ps`
- Verify `.env` file has correct connection strings

### Frontend Won't Start

**Issue:** "Port 3000 already in use"
**Solution:**
- Close other applications using port 3000
- Or change port in `frontend/package.json`: `"dev": "next dev -p 3001"`

**Issue:** "Module not found"
**Solution:**
```powershell
cd frontend
npm install
```

### Can't Login

**Issue:** "Invalid credentials"
**Solution:**
- Default credentials: `admin@test.com` / `admin123`
- Or create new user via API: `POST http://localhost:8000/api/v1/auth/register`

---

## Access Points

Once running, access:

- 🌐 **Frontend:** http://localhost:3000
- 🔌 **Backend API:** http://localhost:8000
- 📚 **API Documentation:** http://localhost:8000/docs
- 🔍 **Admin Dashboard:** http://localhost:3000/admin

---

## Development Workflow

### Making Changes

1. **Backend changes:** Auto-reloads (thanks to `--reload` flag)
2. **Frontend changes:** Auto-reloads (Next.js hot reload)
3. **Database changes:** Run migrations:
   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   alembic revision --autogenerate -m "description"
   alembic upgrade head
   ```

### Viewing Logs

- **Backend logs:** Check the backend PowerShell window
- **Frontend logs:** Check the frontend PowerShell window
- **Database logs:** `docker-compose logs -f postgres mongodb redis`

---

## Next Steps

1. ✅ **Test the chat interface** - Send a message (uses mock responses)
2. ✅ **Load a real model** - See `MODEL_INTEGRATION_GUIDE.md`
3. ✅ **Customize settings** - Edit `.env` files
4. ✅ **Add features** - See `ROADMAP.md`

---

## Quick Commands Reference

```powershell
# Start everything
.\start.bat

# Stop everything
.\stop.bat

# Start only databases
cd deployment
docker-compose up -d

# Stop only databases
cd deployment
docker-compose down

# View database logs
cd deployment
docker-compose logs -f

# Backend only
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

# Frontend only
cd frontend
npm run dev

# Reset everything (fresh start)
cd deployment
docker-compose down -v  # Removes volumes too
cd ..
Remove-Item -Recurse -Force backend\venv
Remove-Item -Recurse -Force frontend\node_modules
.\start.ps1
```

---

## Need Help?

- 📖 **Full Documentation:** See `README.md` and `ROADMAP.md`
- 🔧 **Implementation Guide:** See `IMPLEMENTATION_GUIDE.md`
- 🚀 **Next Steps:** See `NEXT_STEPS.md`
- 🔒 **Security:** See `SECURITY.md`

---

**Happy coding! 🎉**

