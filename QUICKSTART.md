# 🚀 Quick Start Guide

Get MurpheyAI up and running in 10 minutes!

## Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- 8GB+ RAM

## Step 1: Clone & Setup (2 minutes)

```bash
# Navigate to project
cd MurpheyAI

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

## Step 2: Start Databases (1 minute)

```bash
cd ../deployment
docker-compose up -d postgres mongodb redis
```

Wait 30 seconds for databases to initialize.

## Step 3: Configure Environment (2 minutes)

### Backend
```bash
cd ../backend
cp .env.example .env
```

Edit `.env`:
```env
POSTGRES_URL=postgresql://murpheyai:password@localhost:5432/murpheyai
MONGODB_URL=mongodb://localhost:27017/murpheyai
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key-here
```

### Frontend
```bash
cd ../frontend
```

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Step 4: Initialize Database (1 minute)

```bash
cd ../backend
alembic upgrade head
```

## Step 5: Create Admin User (1 minute)

```bash
python
```

```python
from app.core.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash

db = SessionLocal()
admin = User(
    email="admin@test.com",
    username="admin",
    hashed_password=get_password_hash("admin123"),
    is_admin=True
)
db.add(admin)
db.commit()
print("Admin user created!")
exit()
```

## Step 6: Start Servers (2 minutes)

### Terminal 1 - Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
```

### Terminal 2 - Frontend
```bash
cd frontend
npm run dev
```

## Step 7: Access Application (1 minute)

1. Open http://localhost:3000
2. Login with:
   - Username: `admin`
   - Password: `admin123`

## ✅ You're Done!

You should now see:
- ✅ Chat interface
- ✅ Sidebar with conversations
- ✅ Settings panel
- ✅ Admin dashboard (at /admin)

## 🎯 Next Steps

1. **Test Chat**: Send a message (will use mock responses until model is loaded)
2. **Load Model**: See `IMPLEMENTATION_GUIDE.md` for model integration
3. **Customize**: Modify settings, add features
4. **Deploy**: See `DEPLOYMENT.md` for production setup

## 🐛 Troubleshooting

### Backend won't start
- Check database is running: `docker ps`
- Verify `.env` configuration
- Check port 8000 is available

### Frontend won't start
- Check Node.js version: `node --version` (should be 20+)
- Delete `node_modules` and run `npm install` again
- Check port 3000 is available

### Database connection error
- Wait for databases to fully start (30-60 seconds)
- Check Docker containers: `docker-compose ps`
- Verify connection strings in `.env`

### Can't login
- Verify admin user was created
- Check database has users: `SELECT * FROM users;`
- Try creating user via API: `POST /api/v1/auth/register`

## 📚 More Help

- **Full Documentation**: See `ROADMAP.md`
- **Implementation Guide**: See `IMPLEMENTATION_GUIDE.md`
- **Deployment**: See `DEPLOYMENT.md`
- **Security**: See `SECURITY.md`

## 🎉 Happy Building!

You now have a complete AI platform foundation. Start customizing and building your AI!

