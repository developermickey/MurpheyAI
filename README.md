# MurpheyAI - Complete AI Model & Platform

A full-stack AI platform with custom LLM training and deployment infrastructure.

## 🏗️ Project Structure

```
MurpheyAI/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes
│   │   ├── core/           # Core config, security
│   │   ├── models/         # Database models
│   │   ├── services/       # Business logic
│   │   └── main.py         # FastAPI app
│   ├── training/           # Model training scripts
│   └── inference/          # Model inference server
├── frontend/                # Next.js frontend
│   ├── app/                # Next.js app directory
│   ├── components/         # React components
│   └── lib/                # Utilities
├── training/                # Training infrastructure
│   ├── data/               # Data processing
│   ├── models/             # Model definitions
│   └── scripts/            # Training scripts
└── deployment/             # Docker, K8s configs
```

## 🚀 Quick Start

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## 📚 Documentation

See [ROADMAP.md](./ROADMAP.md) for complete architecture and development guide.

## 🛠️ Tech Stack

- **Backend**: FastAPI, PostgreSQL, MongoDB, Redis, Celery
- **Frontend**: Next.js 14, Tailwind CSS, shadcn/ui
- **ML**: PyTorch, Transformers, vLLM
- **Vector DB**: Pinecone/Weaviate
- **Deployment**: Docker, Kubernetes

## 📝 License

MIT

