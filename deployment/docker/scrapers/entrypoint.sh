#!/bin/bash
set -e

# Project Synapse Scrapers Container Entrypoint
# Handles initialization, configuration, and graceful shutdown

# Color codes for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Signal handlers for graceful shutdown
shutdown_handler() {
    log_info "Received shutdown signal, initiating graceful shutdown..."
    
    # Kill background processes
    if [[ -n "$SCRAPER_PID" ]]; then
        log_info "Stopping scraper process (PID: $SCRAPER_PID)..."
        kill -TERM "$SCRAPER_PID" 2>/dev/null || true
        wait "$SCRAPER_PID" 2>/dev/null || true
    fi
    
    # Cleanup temporary files
    cleanup_temp_files
    
    log_info "Graceful shutdown completed"
    exit 0
}

# Cleanup function
cleanup_temp_files() {
    log_debug "Cleaning up temporary files..."
    rm -rf /app/tmp/* 2>/dev/null || true
}

# Trap signals
trap shutdown_handler SIGTERM SIGINT SIGQUIT

# Environment validation
validate_environment() {
    log_info "Validating environment configuration..."
    
    # Required environment variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        exit 1
    fi
    
    # Optional variables with defaults
    export SCRAPER_WORKERS="${SCRAPER_WORKERS:-4}"
    export SCRAPER_TIMEOUT="${SCRAPER_TIMEOUT:-30}"
    export SCRAPER_MAX_RETRIES="${SCRAPER_MAX_RETRIES:-3}"
    export LOG_LEVEL="${LOG_LEVEL:-INFO}"
    export HEALTH_CHECK_PORT="${HEALTH_CHECK_PORT:-8080}"
    
    log_info "Environment validation completed"
}

# Database connectivity check
check_database() {
    log_info "Checking database connectivity..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if python -c "
import asyncio
import sys
from src.synaptic_vesicle.database import DatabaseManager

async def check_db():
    try:
        db_manager = DatabaseManager()
        health = await db_manager.health_check()
        if health['status'] == 'healthy':
            print('Database connection successful')
            return True
        else:
            print(f'Database unhealthy: {health}')
            return False
    except Exception as e:
        print(f'Database connection failed: {e}')
        return False
    finally:
        await db_manager.close()

result = asyncio.run(check_db())
sys.exit(0 if result else 1)
        "; then
            log_info "Database connectivity check passed"
            return 0
        fi
        
        log_warn "Database connectivity check failed (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_error "Database connectivity check failed after $max_attempts attempts"
    exit 1
}

# Redis connectivity check
check_redis() {
    log_info "Checking Redis connectivity..."
    
    if python -c "
import redis
import sys
from urllib.parse import urlparse

try:
    redis_url = '$REDIS_URL'
    parsed = urlparse(redis_url)
    
    r = redis.Redis(
        host=parsed.hostname,
        port=parsed.port or 6379,
        password=parsed.password,
        socket_connect_timeout=5,
        socket_timeout=5
    )
    
    r.ping()
    print('Redis connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Redis connection failed: {e}')
    sys.exit(1)
    "; then
        log_info "Redis connectivity check passed"
    else
        log_error "Redis connectivity check failed"
        exit 1
    fi
}

# Initialize application
initialize_app() {
    log_info "Initializing Project Synapse Scrapers..."
    
    # Create necessary directories
    mkdir -p /app/logs /app/data /app/tmp
    
    # Set up logging configuration
    export PYTHONPATH="/app:$PYTHONPATH"
    
    # Run database migrations if needed
    if [[ "${RUN_MIGRATIONS:-false}" == "true" ]]; then
        log_info "Running database migrations..."
        alembic upgrade head || {
            log_error "Database migration failed"
            exit 1
        }
    fi
    
    log_info "Application initialization completed"
}

# Start health check server
start_health_server() {
    log_info "Starting health check server on port $HEALTH_CHECK_PORT..."
    
    python -c "
import asyncio
import json
from aiohttp import web
from src.neurons.http_scraper import http_scraper
from src.synaptic_vesicle.database import DatabaseManager

async def health_check(request):
    try:
        # Check database
        db_manager = DatabaseManager()
        db_health = await db_manager.health_check()
        
        # Check scraper
        scraper_health = {
            'status': 'healthy',
            'user_agents': len(http_scraper.USER_AGENTS),
            'timeout': http_scraper.timeout
        }
        
        overall_status = 'healthy' if db_health['status'] == 'healthy' else 'unhealthy'
        
        response = {
            'status': overall_status,
            'timestamp': '$(date -Iseconds)',
            'version': '${VERSION:-unknown}',
            'components': {
                'database': db_health,
                'scraper': scraper_health
            }
        }
        
        status_code = 200 if overall_status == 'healthy' else 503
        return web.json_response(response, status=status_code)
        
    except Exception as e:
        return web.json_response({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': '$(date -Iseconds)'
        }, status=503)

async def ready_check(request):
    return web.json_response({'status': 'ready'})

app = web.Application()
app.router.add_get('/health', health_check)
app.router.add_get('/ready', ready_check)

web.run_app(app, host='0.0.0.0', port=$HEALTH_CHECK_PORT, access_log=None)
    " &
    
    HEALTH_SERVER_PID=$!
    log_debug "Health server started with PID: $HEALTH_SERVER_PID"
}

# Start scraper worker
start_scraper() {
    log_info "Starting scraper worker with $SCRAPER_WORKERS workers..."
    
    python -c "
import asyncio
import signal
import sys
from src.neurons.http_scraper import http_scraper
from src.neurons.recipe_engine import recipe_engine
from src.signal_relay.task_dispatcher import get_task_dispatcher

class ScraperWorker:
    def __init__(self):
        self.running = True
        self.task_dispatcher = None
    
    async def start(self):
        try:
            # Initialize components
            self.task_dispatcher = await get_task_dispatcher()
            
            # Register signal handlers
            signal.signal(signal.SIGTERM, self.shutdown_handler)
            signal.signal(signal.SIGINT, self.shutdown_handler)
            
            print('Scraper worker started, waiting for tasks...')
            
            # Main worker loop
            while self.running:
                try:
                    await asyncio.sleep(1)
                    # Task processing happens in the background via task dispatcher
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f'Worker error: {e}')
                    await asyncio.sleep(5)
            
        except Exception as e:
            print(f'Failed to start scraper worker: {e}')
            sys.exit(1)
        finally:
            await self.cleanup()
    
    def shutdown_handler(self, signum, frame):
        print(f'Received signal {signum}, shutting down...')
        self.running = False
    
    async def cleanup(self):
        try:
            if self.task_dispatcher:
                await self.task_dispatcher.shutdown()
            await http_scraper.close()
        except Exception as e:
            print(f'Cleanup error: {e}')

# Run the worker
worker = ScraperWorker()
asyncio.run(worker.start())
    " &
    
    SCRAPER_PID=$!
    log_info "Scraper worker started with PID: $SCRAPER_PID"
}

# Main execution
main() {
    log_info "Starting Project Synapse Scrapers Container"
    log_info "Version: ${VERSION:-unknown}"
    log_info "Build Date: ${BUILD_DATE:-unknown}"
    log_info "VCS Ref: ${VCS_REF:-unknown}"
    
    # Parse command line arguments
    local command="${1:-scraper}"
    
    case "$command" in
        "scraper")
            validate_environment
            check_database
            check_redis
            initialize_app
            start_health_server
            start_scraper
            
            # Wait for processes
            wait $SCRAPER_PID
            ;;
            
        "health-check")
            python healthcheck.py
            ;;
            
        "migrate")
            validate_environment
            check_database
            log_info "Running database migrations..."
            alembic upgrade head
            ;;
            
        "shell")
            validate_environment
            log_info "Starting interactive shell..."
            exec /bin/bash
            ;;
            
        "test")
            log_info "Running tests..."
            python -m pytest tests/ -v
            ;;
            
        *)
            log_info "Running custom command: $*"
            exec "$@"
            ;;
    esac
}

# Execute main function with all arguments
main "$@"