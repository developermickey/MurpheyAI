# Implementation Guide

This guide provides step-by-step instructions for implementing and deploying the MurpheyAI platform.

## Phase 1: Setup Development Environment

### 1.1 Prerequisites

```bash
# Install Python 3.11+
python --version

# Install Node.js 20+
node --version

# Install Docker & Docker Compose
docker --version
docker-compose --version

# Install PostgreSQL, MongoDB, Redis (or use Docker)
```

### 1.2 Clone and Setup

```bash
# Clone repository
git clone <repository-url>
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

### 1.3 Configure Environment

```bash
# Backend
cd backend
cp .env.example .env
# Edit .env with your database URLs and secrets

# Frontend
cd ../frontend
cp .env.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Phase 2: Database Setup

### 2.1 Start Databases (Docker)

```bash
cd deployment
docker-compose up -d postgres mongodb redis
```

### 2.2 Initialize Database Schema

```bash
cd backend
alembic upgrade head
```

### 2.3 Create Admin User

```python
# Run in Python shell
from app.core.database import SessionLocal
from app.models.user import User
from app.core.security import get_password_hash

db = SessionLocal()
admin = User(
    email="admin@murpheyai.com",
    username="admin",
    hashed_password=get_password_hash("admin123"),
    is_admin=True
)
db.add(admin)
db.commit()
```

## Phase 3: Model Training (Optional - Start with Pre-trained)

### 3.1 Collect Data

```bash
cd training
python data/data_collector.py
```

### 3.2 Process Data

```bash
python data/data_processor.py
```

### 3.3 Train Tokenizer

```bash
python tokenizer/train_tokenizer.py
```

### 3.4 Train Model

```bash
# Small model (2B-7B) - requires 8x A100 40GB
python scripts/train.py --config configs/train_config.yaml
```

**Note**: For initial testing, you can use a pre-trained model from Hugging Face and fine-tune it.

## Phase 4: Backend Development

### 4.1 Start Backend Server

```bash
cd backend
uvicorn app.main:app --reload
```

### 4.2 Test API

```bash
# Health check
curl http://localhost:8000/health

# Register user
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@test.com","password":"test123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'
```

## Phase 5: Frontend Development

### 5.1 Start Frontend

```bash
cd frontend
npm run dev
```

### 5.2 Access Application

Open http://localhost:3000 in your browser.

## Phase 6: Model Integration

### 6.1 Load Model

Update `backend/app/services/model_service.py` to load your trained model:

```python
def _load_model(self, model_name: str):
    model_path = f"{settings.MODEL_PATH}/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    self.models[model_name] = model
    self.tokenizers[model_name] = tokenizer
```

### 6.2 Test Inference

```python
# Test in Python
from app.services.model_service import model_service
response = model_service.generate("Hello, how are you?")
```

## Phase 7: Production Deployment

### 7.1 Build Docker Images

```bash
cd deployment
docker-compose build
```

### 7.2 Deploy

```bash
docker-compose up -d
```

### 7.3 Configure Reverse Proxy (NGINX)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 7.4 SSL Certificate

```bash
# Using Let's Encrypt
certbot --nginx -d your-domain.com
```

## Phase 8: Monitoring & Maintenance

### 8.1 Set Up Monitoring

- Configure Prometheus
- Set up Grafana dashboards
- Configure alerting

### 8.2 Backup Strategy

- Daily database backups
- Model versioning
- Configuration backups

### 8.3 Scaling

- Add more API servers
- Scale model inference servers
- Database read replicas

## Troubleshooting

### Backend Issues

1. **Database Connection Error**
   - Check database is running
   - Verify connection string in .env
   - Check network connectivity

2. **Model Loading Error**
   - Verify model files exist
   - Check GPU availability
   - Verify CUDA installation

### Frontend Issues

1. **API Connection Error**
   - Check NEXT_PUBLIC_API_URL
   - Verify backend is running
   - Check CORS settings

2. **Build Errors**
   - Clear .next directory
   - Reinstall node_modules
   - Check Node.js version

## Next Steps

1. **Customize Model**: Fine-tune on your domain data
2. **Add Features**: Voice, images, plugins
3. **Scale Infrastructure**: Add more GPUs, servers
4. **Optimize Costs**: Use quantization, caching
5. **Improve Safety**: Add more safety filters, RLHF

## Resources

- [ROADMAP.md](./ROADMAP.md) - Complete architecture
- [SECURITY.md](./SECURITY.md) - Security practices
- [DEPLOYMENT.md](./DEPLOYMENT.md) - Deployment guide

