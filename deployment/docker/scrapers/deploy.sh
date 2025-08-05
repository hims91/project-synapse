#!/bin/bash
set -e

# Project Synapse Scrapers Deployment Script
# Handles building, deployment, and management of scraper containers

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
ENV_FILE="$SCRIPT_DIR/.env"

# Default values
ENVIRONMENT="production"
ACTION=""
SERVICE=""
BUILD_ARGS=""

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
Project Synapse Scrapers Deployment Script

Usage: $0 [OPTIONS] ACTION [SERVICE]

ACTIONS:
    build           Build container images
    up              Start services
    down            Stop services
    restart         Restart services
    logs            Show service logs
    shell           Open shell in service container
    health          Check service health
    migrate         Run database migrations
    test            Run tests in container
    clean           Clean up containers and volumes
    status          Show service status

SERVICES:
    scrapers        Main scraper service (default)
    scrapers-dev    Development scraper service
    postgres        PostgreSQL database
    redis           Redis cache
    all             All services

OPTIONS:
    -e, --env ENV       Environment (production|development) [default: production]
    -f, --file FILE     Docker compose file [default: docker-compose.yml]
    -d, --detach        Run in detached mode
    -b, --build         Force rebuild images
    --no-cache          Build without cache
    --pull              Pull latest base images
    -v, --verbose       Verbose output
    -h, --help          Show this help

EXAMPLES:
    $0 build                    # Build production images
    $0 -e development up        # Start development environment
    $0 logs scrapers            # Show scraper logs
    $0 shell scrapers-dev       # Open shell in dev container
    $0 health                   # Check all service health
    $0 migrate                  # Run database migrations
    $0 clean                    # Clean up everything

ENVIRONMENT VARIABLES:
    Set these in .env file or environment:
    
    POSTGRES_PASSWORD           PostgreSQL password
    REDIS_PASSWORD             Redis password
    SCRAPER_WORKERS            Number of scraper workers
    LOG_LEVEL                  Logging level (DEBUG|INFO|WARN|ERROR)
    VERSION                    Application version
    BUILD_DATE                 Build timestamp
    VCS_REF                    Git commit hash

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -f|--file)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -d|--detach)
                BUILD_ARGS="$BUILD_ARGS -d"
                shift
                ;;
            -b|--build)
                BUILD_ARGS="$BUILD_ARGS --build"
                shift
                ;;
            --no-cache)
                BUILD_ARGS="$BUILD_ARGS --no-cache"
                shift
                ;;
            --pull)
                BUILD_ARGS="$BUILD_ARGS --pull"
                shift
                ;;
            -v|--verbose)
                DEBUG="true"
                BUILD_ARGS="$BUILD_ARGS --verbose"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            -*)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                if [[ -z "$ACTION" ]]; then
                    ACTION="$1"
                elif [[ -z "$SERVICE" ]]; then
                    SERVICE="$1"
                else
                    log_error "Too many arguments"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "$ACTION" ]]; then
        log_error "Action is required"
        show_help
        exit 1
    fi
    
    # Set default service
    if [[ -z "$SERVICE" ]]; then
        case "$ACTION" in
            up|down|restart|logs|health|status)
                SERVICE="all"
                ;;
            shell)
                SERVICE="scrapers"
                ;;
            *)
                SERVICE="scrapers"
                ;;
        esac
    fi
}

# Setup environment
setup_environment() {
    log_info "Setting up environment: $ENVIRONMENT"
    
    # Load environment file if it exists
    if [[ -f "$ENV_FILE" ]]; then
        log_debug "Loading environment from $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    fi
    
    # Set build information
    export BUILD_DATE="${BUILD_DATE:-$(date -u +'%Y-%m-%dT%H:%M:%SZ')}"
    export VCS_REF="${VCS_REF:-$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}"
    export VERSION="${VERSION:-latest}"
    
    # Set environment-specific profiles
    case "$ENVIRONMENT" in
        development)
            export COMPOSE_PROFILES="development"
            ;;
        production)
            export COMPOSE_PROFILES=""
            ;;
        monitoring)
            export COMPOSE_PROFILES="monitoring"
            ;;
        *)
            log_error "Unknown environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
    
    log_debug "Build Date: $BUILD_DATE"
    log_debug "VCS Ref: $VCS_REF"
    log_debug "Version: $VERSION"
    log_debug "Profiles: $COMPOSE_PROFILES"
}

# Docker compose wrapper
docker_compose() {
    local cmd="docker-compose -f $COMPOSE_FILE"
    
    if [[ -n "$COMPOSE_PROFILES" ]]; then
        cmd="$cmd --profile $COMPOSE_PROFILES"
    fi
    
    log_debug "Running: $cmd $*"
    $cmd "$@"
}

# Build images
build_images() {
    log_info "Building container images..."
    
    if [[ "$SERVICE" == "all" ]]; then
        docker_compose build $BUILD_ARGS
    else
        docker_compose build $BUILD_ARGS "$SERVICE"
    fi
    
    log_info "Build completed successfully"
}

# Start services
start_services() {
    log_info "Starting services..."
    
    if [[ "$SERVICE" == "all" ]]; then
        docker_compose up $BUILD_ARGS
    else
        docker_compose up $BUILD_ARGS "$SERVICE"
    fi
}

# Stop services
stop_services() {
    log_info "Stopping services..."
    
    if [[ "$SERVICE" == "all" ]]; then
        docker_compose down
    else
        docker_compose stop "$SERVICE"
    fi
    
    log_info "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting services..."
    
    if [[ "$SERVICE" == "all" ]]; then
        docker_compose restart
    else
        docker_compose restart "$SERVICE"
    fi
    
    log_info "Services restarted"
}

# Show logs
show_logs() {
    log_info "Showing logs for $SERVICE..."
    
    if [[ "$SERVICE" == "all" ]]; then
        docker_compose logs -f
    else
        docker_compose logs -f "$SERVICE"
    fi
}

# Open shell
open_shell() {
    log_info "Opening shell in $SERVICE..."
    
    case "$SERVICE" in
        scrapers)
            docker_compose exec scrapers /bin/bash
            ;;
        scrapers-dev)
            docker_compose exec scrapers-dev /bin/bash
            ;;
        postgres)
            docker_compose exec postgres psql -U synapse -d synapse
            ;;
        redis)
            docker_compose exec redis redis-cli
            ;;
        *)
            log_error "Shell not supported for service: $SERVICE"
            exit 1
            ;;
    esac
}

# Check health
check_health() {
    log_info "Checking service health..."
    
    if [[ "$SERVICE" == "all" ]]; then
        docker_compose ps
        echo
        log_info "Detailed health check:"
        docker_compose exec scrapers python healthcheck.py || true
    else
        case "$SERVICE" in
            scrapers|scrapers-dev)
                docker_compose exec "$SERVICE" python healthcheck.py
                ;;
            postgres)
                docker_compose exec postgres pg_isready -U synapse -d synapse
                ;;
            redis)
                docker_compose exec redis redis-cli ping
                ;;
            *)
                docker_compose ps "$SERVICE"
                ;;
        esac
    fi
}

# Run migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Ensure database is running
    docker_compose up -d postgres
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    timeout=60
    while ! docker_compose exec postgres pg_isready -U synapse -d synapse >/dev/null 2>&1; do
        sleep 2
        timeout=$((timeout - 2))
        if [[ $timeout -le 0 ]]; then
            log_error "Database failed to start within timeout"
            exit 1
        fi
    done
    
    # Run migrations
    docker_compose run --rm scrapers migrate
    
    log_info "Migrations completed"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Start dependencies
    docker_compose up -d postgres redis
    
    # Run tests
    docker_compose run --rm scrapers test
    
    log_info "Tests completed"
}

# Clean up
cleanup() {
    log_info "Cleaning up containers and volumes..."
    
    # Stop and remove containers
    docker_compose down -v --remove-orphans
    
    # Remove images if requested
    if [[ "${CLEAN_IMAGES:-false}" == "true" ]]; then
        log_info "Removing images..."
        docker_compose down --rmi all
    fi
    
    # Prune unused resources
    docker system prune -f
    
    log_info "Cleanup completed"
}

# Show status
show_status() {
    log_info "Service status:"
    docker_compose ps
    
    echo
    log_info "Resource usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
}

# Main execution
main() {
    parse_args "$@"
    setup_environment
    
    log_info "Project Synapse Scrapers Deployment"
    log_info "Action: $ACTION, Service: $SERVICE, Environment: $ENVIRONMENT"
    
    case "$ACTION" in
        build)
            build_images
            ;;
        up)
            start_services
            ;;
        down)
            stop_services
            ;;
        restart)
            restart_services
            ;;
        logs)
            show_logs
            ;;
        shell)
            open_shell
            ;;
        health)
            check_health
            ;;
        migrate)
            run_migrations
            ;;
        test)
            run_tests
            ;;
        clean)
            cleanup
            ;;
        status)
            show_status
            ;;
        *)
            log_error "Unknown action: $ACTION"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"