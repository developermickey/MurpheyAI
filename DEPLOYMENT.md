# Deployment Guide

## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA (for model inference)
- 16GB+ RAM
- 100GB+ storage

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd MurpheyAI
```

### 2. Configure Environment

```bash
# Backend
cd backend
cp .env.example .env
# Edit .env with your configuration

# Frontend
cd ../frontend
cp .env.example .env.local
# Edit .env.local with your API URL
```

### 3. Start Services

```bash
cd deployment
docker-compose up -d
```

### 4. Initialize Database

```bash
# Run migrations
cd backend
alembic upgrade head
```

## Production Deployment

### Option 1: Docker Compose (Single Server)

1. **Update docker-compose.yml** with production settings
2. **Set environment variables**
3. **Configure reverse proxy** (NGINX)
4. **Set up SSL certificates**
5. **Configure monitoring**

### Option 2: Kubernetes

1. **Create Kubernetes cluster**
2. **Apply configurations**:
   ```bash
   kubectl apply -f deployment/k8s/
   ```
3. **Set up ingress**
4. **Configure persistent volumes**
5. **Set up monitoring**

### Option 3: Cloud Services

#### AWS
- **ECS/EKS** for containers
- **RDS** for PostgreSQL
- **DocumentDB** for MongoDB
- **ElastiCache** for Redis
- **S3** for model storage
- **CloudFront** for CDN

#### Google Cloud
- **GKE** for Kubernetes
- **Cloud SQL** for PostgreSQL
- **Cloud Firestore** for MongoDB
- **Memorystore** for Redis
- **Cloud Storage** for models

#### Azure
- **AKS** for Kubernetes
- **Azure Database** for PostgreSQL
- **Cosmos DB** for MongoDB
- **Azure Cache** for Redis
- **Blob Storage** for models

## Model Deployment

### 1. Train or Download Model

```bash
cd training
python scripts/train.py --config configs/train_config.yaml
```

### 2. Convert to Inference Format

```bash
# Convert to ONNX or TensorRT
python scripts/convert_model.py --input checkpoints/model.pt --output models/model.onnx
```

### 3. Deploy Model Server

```bash
# Using vLLM
python -m vllm.entrypoints.api_server \
    --model ./models/murpheyai-7b \
    --port 8001
```

## Scaling

### Horizontal Scaling

1. **Load Balancer**: NGINX, HAProxy
2. **API Servers**: Multiple backend instances
3. **Model Servers**: Multiple GPU instances with load balancing
4. **Database**: Read replicas, sharding

### Vertical Scaling

1. **GPU Upgrades**: A100 → H100
2. **Memory**: Increase RAM/VRAM
3. **Storage**: Faster SSDs

## Monitoring

### Metrics
- Prometheus for metrics collection
- Grafana for visualization
- Custom dashboards

### Logging
- ELK Stack (Elasticsearch, Logstash, Kibana)
- CloudWatch (AWS)
- Stackdriver (GCP)

### APM
- Sentry for error tracking
- New Relic
- Datadog

## Backup & Recovery

1. **Database Backups**
   - Daily automated backups
   - Point-in-time recovery
   - Off-site storage

2. **Model Backups**
   - Version control
   - S3/Cloud Storage
   - Regular snapshots

3. **Disaster Recovery**
   - Backup procedures
   - Recovery testing
   - RTO/RPO targets

## Security

1. **Network**
   - VPC isolation
   - Security groups
   - WAF rules

2. **Secrets**
   - Secret management service
   - Environment variables
   - No secrets in code

3. **SSL/TLS**
   - Let's Encrypt certificates
   - Certificate rotation
   - HSTS headers

## Cost Optimization

1. **GPU Usage**
   - Spot instances for training
   - Auto-scaling
   - Model quantization

2. **Storage**
   - Lifecycle policies
   - Compression
   - Tiered storage

3. **Compute**
   - Right-sizing instances
   - Reserved instances
   - Auto-shutdown

