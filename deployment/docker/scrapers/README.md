# Project Synapse Scrapers - Docker Deployment

This directory contains Docker containerization for Project Synapse lightweight scrapers (Neurons layer). The containerized scrapers provide production-ready deployment with comprehensive health monitoring, graceful shutdown, and environment management.

## 🏗️ Architecture

The Docker deployment includes:

- **Multi-stage Dockerfile** with optimized production and development builds
- **Comprehensive health checks** for all system components
- **Graceful shutdown handling** with proper signal management
- **Environment configuration** with validation and defaults
- **Security hardening** with non-root user and minimal attack surface
- **Resource management** with configurable limits and monitoring

## 📁 Files Overview

```
deployment/docker/scrapers/
├── Dockerfile              # Multi-stage container definition
├── docker-compose.yml      # Complete deployment orchestration
├── entrypoint.sh           # Container initialization and lifecycle management
├── healthcheck.py          # Comprehensive health check system
├── deploy.sh               # Deployment automation script
├── init-db.sql            # Database initialization
├── .env.example           # Environment configuration template
└── README.md              # This documentation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 2. Production Deployment

```bash
# Build and start all services
./deploy.sh build
./deploy.sh up

# Or in one command
./deploy.sh -b up
```

### 3. Development Environment

```bash
# Start development environment
./deploy.sh -e development up

# Open development shell
./deploy.sh -e development shell scrapers-dev
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `DATABASE_URL` | PostgreSQL connection string | - | ✅ |
| `REDIS_URL` | Redis connection string | - | ✅ |
| `SCRAPER_WORKERS` | Number of concurrent workers | 4 | ❌ |
| `SCRAPER_TIMEOUT` | Request timeout in seconds | 30 | ❌ |
| `SCRAPER_MAX_RETRIES` | Maximum retry attempts | 3 | ❌ |
| `LOG_LEVEL` | Logging level (DEBUG/INFO/WARN/ERROR) | INFO | ❌ |
| `HEALTH_CHECK_PORT` | Health check server port | 8080 | ❌ |
| `RUN_MIGRATIONS` | Run database migrations on startup | false | ❌ |
| `DEBUG` | Enable debug mode | false | ❌ |

### Docker Compose Profiles

- **Default**: Production scrapers with PostgreSQL and Redis
- **Development**: Development scrapers with debugging tools
- **Monitoring**: Adds Prometheus and Grafana for observability

## 🛠️ Deployment Script Usage

The `deploy.sh` script provides comprehensive deployment management:

### Basic Commands

```bash
# Build images
./deploy.sh build

# Start services
./deploy.sh up

# Stop services  
./deploy.sh down

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs scrapers

# Check health
./deploy.sh health

# Run database migrations
./deploy.sh migrate

# Run tests
./deploy.sh test

# Clean up everything
./deploy.sh clean
```

### Advanced Usage

```bash
# Development environment
./deploy.sh -e development up

# Force rebuild with no cache
./deploy.sh --no-cache build

# Detached mode
./deploy.sh -d up

# Verbose output
./deploy.sh -v build

# Open shell in container
./deploy.sh shell scrapers-dev
```

## 🏥 Health Monitoring

### Health Check Endpoints

The scrapers expose comprehensive health check endpoints:

```bash
# Overall health status
curl http://localhost:8080/health

# Readiness check
curl http://localhost:8080/ready
```

### Health Check Components

The health check system monitors:

- **Database connectivity** - PostgreSQL connection and query performance
- **Redis connectivity** - Cache server availability and response time
- **Scraper components** - Recipe engine and HTTP client functionality
- **Filesystem health** - Disk space and directory permissions
- **Environment validation** - Required configuration variables

### Health Check Output

```json
{
  "overall_status": "healthy",
  "timestamp": 1705312200.123,
  "uptime_seconds": 3600.45,
  "checks": {
    "database": {
      "status": "healthy",
      "details": {
        "connection_pool": "active",
        "query_time_ms": 12.3
      }
    },
    "redis": {
      "status": "healthy",
      "host": "redis",
      "port": 6379
    },
    "scraper_components": {
      "status": "healthy",
      "components": {
        "http_scraper": {
          "timeout": 30,
          "max_retries": 3,
          "user_agents": 8
        },
        "recipe_engine": {
          "cache_ttl": "1:00:00",
          "validation_working": true
        }
      }
    }
  }
}
```

## 🔒 Security Features

### Container Security

- **Non-root user** - Runs as dedicated `synapse` user
- **Minimal base image** - Python slim image with only required packages
- **Read-only filesystem** - Application code mounted read-only
- **Resource limits** - CPU and memory constraints
- **Health checks** - Automatic restart on failure

### Network Security

- **Internal network** - Services communicate on isolated Docker network
- **Port exposure** - Only health check port exposed externally
- **Environment isolation** - Secrets managed through environment variables

### Data Security

- **Volume encryption** - Persistent data stored in named volumes
- **Secrets management** - Database and Redis passwords via environment
- **Log sanitization** - Sensitive data filtered from logs

## 📊 Monitoring and Observability

### Logging

Structured logging with contextual information:

```bash
# View real-time logs
./deploy.sh logs scrapers

# Filter by log level
docker-compose logs scrapers | grep ERROR

# Export logs
docker-compose logs --no-color scrapers > scrapers.log
```

### Metrics Collection

Optional Prometheus and Grafana integration:

```bash
# Start with monitoring
./deploy.sh -e monitoring up

# Access Grafana
open http://localhost:3000

# Access Prometheus
open http://localhost:9090
```

### Performance Monitoring

```bash
# Container resource usage
./deploy.sh status

# Detailed container stats
docker stats synapse-scrapers

# Health check with metrics
curl http://localhost:8080/health | jq .
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
./deploy.sh test

# Run specific test file
docker-compose run --rm scrapers python -m pytest tests/test_http_scraper.py -v
```

### Integration Tests

```bash
# Test with real services
./deploy.sh up -d postgres redis
./deploy.sh test

# Test health checks
./deploy.sh health
```

### Load Testing

```bash
# Start scrapers
./deploy.sh up -d

# Run load test (requires additional tools)
# ab -n 1000 -c 10 http://localhost:8080/health
```

## 🚨 Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
./deploy.sh logs scrapers

# Check health
./deploy.sh health

# Verify environment
docker-compose config
```

#### Database Connection Issues

```bash
# Check database status
./deploy.sh health postgres

# Test connection manually
docker-compose exec postgres psql -U synapse -d synapse -c "SELECT 1;"

# Check environment variables
docker-compose exec scrapers env | grep DATABASE
```

#### Memory Issues

```bash
# Check resource usage
./deploy.sh status

# Increase memory limits in docker-compose.yml
# deploy:
#   resources:
#     limits:
#       memory: 2G
```

#### Permission Issues

```bash
# Check volume permissions
docker-compose exec scrapers ls -la /app/

# Fix permissions
docker-compose exec --user root scrapers chown -R synapse:synapse /app/logs
```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Set debug environment
echo "DEBUG=true" >> .env

# Restart with debug logging
./deploy.sh restart

# View debug logs
./deploy.sh logs scrapers | grep DEBUG
```

### Container Shell Access

```bash
# Production container
./deploy.sh shell scrapers

# Development container
./deploy.sh shell scrapers-dev

# Root access (for debugging)
docker-compose exec --user root scrapers /bin/bash
```

## 🔄 Maintenance

### Updates and Upgrades

```bash
# Pull latest images
./deploy.sh --pull build

# Backup data before upgrade
docker-compose exec postgres pg_dump -U synapse synapse > backup.sql

# Upgrade with zero downtime
./deploy.sh build
./deploy.sh up --no-deps scrapers
```

### Backup and Recovery

```bash
# Database backup
docker-compose exec postgres pg_dump -U synapse synapse | gzip > backup_$(date +%Y%m%d).sql.gz

# Redis backup
docker-compose exec redis redis-cli BGSAVE

# Volume backup
docker run --rm -v synapse_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

### Log Rotation

```bash
# Configure log rotation in docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale scrapers
docker-compose up -d --scale scrapers=3

# Load balancer configuration needed for multiple instances
```

### Vertical Scaling

```bash
# Increase resources in docker-compose.yml
deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 2G
    reservations:
      cpus: '1.0'
      memory: 512M
```

## 🤝 Contributing

When contributing to the Docker deployment:

1. Test changes in development environment
2. Update documentation for new features
3. Ensure health checks cover new components
4. Test deployment script changes
5. Validate security implications

## 📄 License

This Docker deployment is part of Project Synapse and follows the same licensing terms.