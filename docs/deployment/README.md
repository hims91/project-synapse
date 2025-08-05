# Project Synapse Deployment Guide

This guide covers deploying Project Synapse in various environments, from development to production.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Database Setup](#database-setup)
6. [Application Deployment](#application-deployment)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Security Configuration](#security-configuration)
9. [Scaling and Performance](#scaling-and-performance)
10. [Troubleshooting](#troubleshooting)

## Quick Start

### Docker Compose (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/project-synapse/synapse.git
cd synapse

# Copy environment configuration
cp .env.example .env

# Edit configuration
nano .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### One-Click Cloud Deployment

[![Deploy to Railway](https://railway.app/button.svg)](https://railway.app/template/synapse)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/project-synapse/synapse)
[![Deploy to Heroku](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy?template=https://github.com/project-synapse/synapse)

## Architecture Overview

Project Synapse follows a brain-inspired, microservices architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                 Axon Interface (FastAPI)                   │
│                    - REST API                              │
│                    - WebSocket Server                      │
│                    - Authentication                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                Central Cortex (Hub Server)                 │
│                    - Request Routing                       │
│                    - Rate Limiting                         │
│                    - Monitoring                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │        Thalamus (NLP Engine)         │
│  ┌──────────────────┴──────────────────────────────────────┤
│  │              Signal Relay (Task Dispatcher)             │
│  └──────────────────┬──────────────────────────────────────┤
│                     │                                      │
│  ┌──────────────────┴──────────────────────────────────────┤
│  │           Synaptic Vesicle (Database Layer)             │
│  └──────────────────┬──────────────────────────────────────┤
│                     │                                      │
│  ┌──────────────────┴──────────────────────────────────────┤
│  │              Spinal Cord (Fallback System)              │
│  └─────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│                External Services                           │
│  - PostgreSQL Database                                     │
│  - Redis Cache                                             │
│  - Cloudflare R2 Storage                                   │
│  - Cloudflare Workers                                      │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB SSD
- Network: 100 Mbps

**Recommended for Production:**
- CPU: 8 cores
- RAM: 16GB
- Storage: 100GB SSD
- Network: 1 Gbps

### Software Dependencies

- **Python**: 3.10 or higher
- **Node.js**: 18 or higher (for Cloudflare Workers)
- **Docker**: 20.10 or higher
- **Docker Compose**: 2.0 or higher

### External Services

- **PostgreSQL**: 14 or higher
- **Redis**: 6.0 or higher
- **Cloudflare Account**: For R2 storage and Workers
- **Optional**: Vercel account for edge functions

## Environment Setup

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Application Settings
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-secret-key-here
API_HOST=0.0.0.0
API_PORT=8000

# Database Configuration
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/synapse
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_POOL_SIZE=10

# Cloudflare Configuration
CLOUDFLARE_ACCOUNT_ID=your-account-id
CLOUDFLARE_API_TOKEN=your-api-token
CLOUDFLARE_R2_BUCKET=synapse-storage
CLOUDFLARE_R2_ACCESS_KEY=your-access-key
CLOUDFLARE_R2_SECRET_KEY=your-secret-key

# Security Settings
JWT_SECRET_KEY=your-jwt-secret
API_KEY_ENCRYPTION_KEY=your-encryption-key
ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REDIS_URL=redis://localhost:6379/1

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO
METRICS_ENABLED=true

# External APIs
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
```

### Configuration Validation

```bash
# Validate configuration
python -c "from src.shared.config import get_settings; print('✅ Configuration valid')"

# Test database connection
python -c "from src.shared.database import test_connection; test_connection()"

# Test Redis connection
python -c "from src.shared.caching import test_redis; test_redis()"
```

## Database Setup

### PostgreSQL Installation

#### Using Docker

```bash
# Start PostgreSQL container
docker run -d \
  --name synapse-postgres \
  -e POSTGRES_DB=synapse \
  -e POSTGRES_USER=synapse \
  -e POSTGRES_PASSWORD=your-password \
  -p 5432:5432 \
  -v synapse_postgres_data:/var/lib/postgresql/data \
  postgres:14
```

#### Using Package Manager

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql-server postgresql-contrib

# macOS
brew install postgresql
```

### Database Migration

```bash
# Install Alembic
pip install alembic

# Initialize migrations (first time only)
alembic init migrations

# Create migration
alembic revision --autogenerate -m "Initial migration"

# Apply migrations
alembic upgrade head

# Verify migration
python -c "from src.shared.database import verify_schema; verify_schema()"
```

### Database Optimization

```sql
-- Create indexes for performance
CREATE INDEX CONCURRENTLY idx_articles_created_at ON articles(created_at);
CREATE INDEX CONCURRENTLY idx_articles_sentiment ON articles((analysis->>'sentiment'));
CREATE INDEX CONCURRENTLY idx_articles_topics ON articles USING GIN((analysis->'topics'));

-- Enable full-text search
CREATE INDEX CONCURRENTLY idx_articles_fts ON articles USING GIN(to_tsvector('english', title || ' ' || content));

-- Analyze tables
ANALYZE articles;
ANALYZE scraping_recipes;
ANALYZE task_queue;
```

## Application Deployment

### Docker Deployment

#### Build Images

```bash
# Build main application
docker build -t synapse-api:latest .

# Build worker image
docker build -f Dockerfile.worker -t synapse-worker:latest .

# Build frontend (if applicable)
docker build -f frontend/Dockerfile -t synapse-frontend:latest frontend/
```

#### Docker Compose Production

```yaml
version: '3.8'

services:
  api:
    image: synapse-api:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://synapse:${DB_PASSWORD}@postgres:5432/synapse
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  worker:
    image: synapse-worker:latest
    environment:
      - DATABASE_URL=postgresql+asyncpg://synapse:${DB_PASSWORD}@postgres:5432/synapse
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=synapse
      - POSTGRES_USER=synapse
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

#### Namespace and ConfigMap

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: synapse

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: synapse-config
  namespace: synapse
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
```

#### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: synapse-api
  namespace: synapse
spec:
  replicas: 3
  selector:
    matchLabels:
      app: synapse-api
  template:
    metadata:
      labels:
        app: synapse-api
    spec:
      containers:
      - name: api
        image: synapse-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: synapse-config
        - secretRef:
            name: synapse-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

#### Service and Ingress

```yaml
apiVersion: v1
kind: Service
metadata:
  name: synapse-api-service
  namespace: synapse
spec:
  selector:
    app: synapse-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: synapse-ingress
  namespace: synapse
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.projectsynapse.com
    secretName: synapse-tls
  rules:
  - host: api.projectsynapse.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: synapse-api-service
            port:
              number: 80
```

### Cloud Platform Deployment

#### Railway

```toml
# railway.toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[[services]]
name = "api"
source = "."

[[services]]
name = "worker"
source = "."
dockerfile = "Dockerfile.worker"
```

#### Render

```yaml
# render.yaml
services:
- type: web
  name: synapse-api
  env: docker
  dockerfilePath: ./Dockerfile
  healthCheckPath: /health
  envVars:
  - key: DATABASE_URL
    fromDatabase:
      name: synapse-db
      property: connectionString
  - key: REDIS_URL
    fromService:
      type: redis
      name: synapse-redis
      property: connectionString

- type: worker
  name: synapse-worker
  env: docker
  dockerfilePath: ./Dockerfile.worker
  envVars:
  - key: DATABASE_URL
    fromDatabase:
      name: synapse-db
      property: connectionString

databases:
- name: synapse-db
  databaseName: synapse
  user: synapse

- name: synapse-redis
  type: redis
```

## Monitoring and Logging

### Prometheus Metrics

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'synapse-api'
  static_configs:
  - targets: ['localhost:8000']
  metrics_path: '/metrics'
  scrape_interval: 30s
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Project Synapse Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Log Aggregation

```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200

volumes:
  elasticsearch_data:
```

## Security Configuration

### SSL/TLS Setup

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.projectsynapse.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    location / {
        proxy_pass http://api:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw deny 5432/tcp  # PostgreSQL (internal only)
sudo ufw deny 6379/tcp  # Redis (internal only)
sudo ufw enable

# iptables
iptables -A INPUT -p tcp --dport 22 -j ACCEPT
iptables -A INPUT -p tcp --dport 80 -j ACCEPT
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 5432 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 6379 -s 10.0.0.0/8 -j ACCEPT
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
version: '3.8'

services:
  api:
    image: synapse-api:latest
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G

  worker:
    image: synapse-worker:latest
    deploy:
      replicas: 3
```

### Load Balancing

```nginx
# nginx-lb.conf
upstream synapse_api {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
    server api3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location / {
        proxy_pass http://synapse_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: synapse-api-hpa
  namespace: synapse
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: synapse-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues

#### Database Connection Issues

```bash
# Check database connectivity
pg_isready -h localhost -p 5432 -U synapse

# Check connection pool
python -c "
from src.shared.database import get_database_manager
import asyncio
async def test():
    db = get_database_manager()
    await db.initialize()
    print('✅ Database connected')
asyncio.run(test())
"
```

#### Redis Connection Issues

```bash
# Test Redis connection
redis-cli ping

# Check Redis memory usage
redis-cli info memory

# Monitor Redis commands
redis-cli monitor
```

#### High Memory Usage

```bash
# Check memory usage
docker stats

# Profile Python memory usage
python -m memory_profiler src/axon_interface/main.py

# Check for memory leaks
valgrind --tool=memcheck python src/axon_interface/main.py
```

### Health Checks

```bash
# API health check
curl -f http://localhost:8000/health

# Database health check
curl -f http://localhost:8000/health/database

# Cache health check
curl -f http://localhost:8000/health/cache

# Full system check
curl -f http://localhost:8000/health/full
```

### Log Analysis

```bash
# View application logs
docker-compose logs -f api

# Search for errors
docker-compose logs api | grep ERROR

# Monitor real-time logs
tail -f /var/log/synapse/api.log | grep -E "(ERROR|WARN)"
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
pg_dump -h localhost -U synapse synapse > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U synapse synapse | gzip > $BACKUP_DIR/synapse_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "synapse_*.sql.gz" -mtime +7 -delete
```

### Restore Database

```bash
# Restore from backup
gunzip -c backup_20240108_120000.sql.gz | psql -h localhost -U synapse synapse
```

### Configuration Backup

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml nginx.conf

# Backup to cloud storage
aws s3 cp config_backup_$(date +%Y%m%d).tar.gz s3://synapse-backups/config/
```

## Support

For deployment support:
- **Documentation**: [docs.projectsynapse.com/deployment](https://docs.projectsynapse.com/deployment)
- **Support Email**: [support@projectsynapse.com](mailto:support@projectsynapse.com)
- **Discord**: [discord.gg/projectsynapse](https://discord.gg/projectsynapse)
- **GitHub Issues**: [github.com/project-synapse/synapse/issues](https://github.com/project-synapse/synapse/issues)