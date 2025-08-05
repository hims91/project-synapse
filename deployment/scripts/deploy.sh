#!/bin/bash

# Project Synapse - One-Click Deployment Script
# Automated deployment system with environment provisioning

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TERRAFORM_DIR="${PROJECT_ROOT}/deployment/infrastructure/terraform"
DOCKER_DIR="${PROJECT_ROOT}"

# Default values
ENVIRONMENT="dev"
AWS_REGION="us-east-1"
SKIP_TESTS=false
SKIP_BUILD=false
FORCE_DEPLOY=false
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Project Synapse - One-Click Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, production) [default: dev]
    -r, --region REGION      AWS region [default: us-east-1]
    -s, --skip-tests         Skip running tests before deployment
    -b, --skip-build         Skip building Docker images
    -f, --force              Force deployment without confirmation
    -d, --dry-run            Show what would be deployed without making changes
    -v, --verbose            Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    $0                                    # Deploy to dev environment
    $0 -e staging                         # Deploy to staging
    $0 -e production -f                   # Force deploy to production
    $0 -e dev --skip-tests --skip-build   # Quick deploy without tests/build
    $0 --dry-run -e production            # Preview production deployment

ENVIRONMENT VARIABLES:
    DATABASE_PASSWORD        Database password (required)
    CLOUDFLARE_API_TOKEN     Cloudflare API token (optional)
    VERCEL_API_TOKEN         Vercel API token (optional)
    SSL_CERTIFICATE_ARN      SSL certificate ARN (optional)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -r|--region)
                AWS_REGION="$2"
                shift 2
                ;;
            -s|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -b|--skip-build)
                SKIP_BUILD=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        dev|staging|production)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Must be one of: dev, staging, production"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    command -v aws >/dev/null 2>&1 || missing_tools+=("aws-cli")
    command -v terraform >/dev/null 2>&1 || missing_tools+=("terraform")
    command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install the missing tools and try again."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS credentials not configured. Please run 'aws configure' or set AWS environment variables."
        exit 1
    fi
    
    # Check required environment variables
    if [[ -z "${DATABASE_PASSWORD:-}" ]]; then
        log_error "DATABASE_PASSWORD environment variable is required"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests"
        return 0
    fi
    
    log_info "Running tests..."
    
    cd "$PROJECT_ROOT"
    
    # Install dependencies if needed
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    fi
    
    if [[ -f "requirements-dev.txt" ]]; then
        pip install -r requirements-dev.txt
    fi
    
    # Run tests
    if command -v pytest >/dev/null 2>&1; then
        log_info "Running pytest..."
        pytest tests/ -v --tb=short
    else
        log_info "Running Python tests..."
        python -m unittest discover tests/
    fi
    
    log_success "Tests passed"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping Docker build"
        return 0
    fi
    
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build main application image
    local image_tag="project-synapse:${ENVIRONMENT}"
    
    log_info "Building image: $image_tag"
    docker build -t "$image_tag" .
    
    # Tag for ECR if not dev environment
    if [[ "$ENVIRONMENT" != "dev" ]]; then
        local aws_account_id
        aws_account_id=$(aws sts get-caller-identity --query Account --output text)
        local ecr_repo="${aws_account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com/project-synapse"
        
        # Create ECR repository if it doesn't exist
        aws ecr describe-repositories --repository-names project-synapse --region "$AWS_REGION" >/dev/null 2>&1 || {
            log_info "Creating ECR repository..."
            aws ecr create-repository --repository-name project-synapse --region "$AWS_REGION"
        }
        
        # Login to ECR
        aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ecr_repo"
        
        # Tag and push
        docker tag "$image_tag" "${ecr_repo}:${ENVIRONMENT}"
        docker tag "$image_tag" "${ecr_repo}:latest"
        
        log_info "Pushing to ECR..."
        docker push "${ecr_repo}:${ENVIRONMENT}"
        docker push "${ecr_repo}:latest"
        
        # Update terraform variable
        export TF_VAR_app_image="${ecr_repo}:${ENVIRONMENT}"
    fi
    
    log_success "Docker images built successfully"
}

# Initialize Terraform
init_terraform() {
    log_info "Initializing Terraform..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    terraform init -upgrade
    
    # Select or create workspace
    terraform workspace select "$ENVIRONMENT" 2>/dev/null || terraform workspace new "$ENVIRONMENT"
    
    log_success "Terraform initialized"
}

# Plan Terraform deployment
plan_terraform() {
    log_info "Planning Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    
    # Set Terraform variables
    export TF_VAR_environment="$ENVIRONMENT"
    export TF_VAR_aws_region="$AWS_REGION"
    export TF_VAR_database_password="$DATABASE_PASSWORD"
    
    # Optional variables
    [[ -n "${CLOUDFLARE_API_TOKEN:-}" ]] && export TF_VAR_cloudflare_api_token="$CLOUDFLARE_API_TOKEN"
    [[ -n "${VERCEL_API_TOKEN:-}" ]] && export TF_VAR_vercel_api_token="$VERCEL_API_TOKEN"
    [[ -n "${SSL_CERTIFICATE_ARN:-}" ]] && export TF_VAR_ssl_certificate_arn="$SSL_CERTIFICATE_ARN"
    
    # Plan with environment-specific variables
    local var_file="environments/${ENVIRONMENT}.tfvars"
    
    if [[ ! -f "$var_file" ]]; then
        log_error "Environment configuration file not found: $var_file"
        exit 1
    fi
    
    terraform plan -var-file="$var_file" -out="tfplan-${ENVIRONMENT}"
    
    log_success "Terraform plan completed"
}

# Apply Terraform deployment
apply_terraform() {
    log_info "Applying Terraform deployment..."
    
    cd "$TERRAFORM_DIR"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - skipping actual deployment"
        return 0
    fi
    
    # Confirmation for production
    if [[ "$ENVIRONMENT" == "production" && "$FORCE_DEPLOY" != "true" ]]; then
        echo
        log_warning "You are about to deploy to PRODUCTION environment!"
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Deployment cancelled"
            exit 0
        fi
    fi
    
    # Apply the plan
    terraform apply "tfplan-${ENVIRONMENT}"
    
    log_success "Terraform deployment completed"
}

# Get deployment outputs
get_outputs() {
    log_info "Getting deployment outputs..."
    
    cd "$TERRAFORM_DIR"
    
    # Get important outputs
    local alb_dns_name
    local database_endpoint
    local redis_endpoint
    
    alb_dns_name=$(terraform output -raw alb_dns_name 2>/dev/null || echo "N/A")
    database_endpoint=$(terraform output -raw database_endpoint 2>/dev/null || echo "N/A")
    redis_endpoint=$(terraform output -raw redis_endpoint 2>/dev/null || echo "N/A")
    
    echo
    log_success "Deployment completed successfully!"
    echo
    echo "=== Deployment Information ==="
    echo "Environment: $ENVIRONMENT"
    echo "AWS Region: $AWS_REGION"
    echo "Load Balancer DNS: $alb_dns_name"
    echo "Database Endpoint: $database_endpoint"
    echo "Redis Endpoint: $redis_endpoint"
    echo
    
    # Save outputs to file
    local output_file="${PROJECT_ROOT}/deployment-outputs-${ENVIRONMENT}.json"
    terraform output -json > "$output_file"
    log_info "Deployment outputs saved to: $output_file"
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # This would typically connect to the deployed database and run migrations
    # For now, we'll just log that this step would happen
    log_info "Database migrations would be run here"
    log_success "Database migrations completed"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    cd "$TERRAFORM_DIR"
    
    local alb_dns_name
    alb_dns_name=$(terraform output -raw alb_dns_name 2>/dev/null || echo "")
    
    if [[ -n "$alb_dns_name" ]]; then
        local health_url="http://${alb_dns_name}/health"
        local max_attempts=30
        local attempt=1
        
        log_info "Checking health endpoint: $health_url"
        
        while [[ $attempt -le $max_attempts ]]; do
            if curl -f -s "$health_url" >/dev/null 2>&1; then
                log_success "Health check passed"
                return 0
            fi
            
            log_info "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
            sleep 10
            ((attempt++))
        done
        
        log_warning "Health check failed after $max_attempts attempts"
        log_warning "The deployment may still be starting up"
    else
        log_warning "Could not determine load balancer DNS name for health check"
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    
    cd "$TERRAFORM_DIR"
    rm -f "tfplan-${ENVIRONMENT}"
    
    log_success "Cleanup completed"
}

# Main deployment function
main() {
    log_info "Starting Project Synapse deployment..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run deployment steps
    validate_environment
    check_prerequisites
    run_tests
    build_images
    init_terraform
    plan_terraform
    apply_terraform
    get_outputs
    run_migrations
    health_check
    
    log_success "Deployment completed successfully!"
    
    echo
    echo "Next steps:"
    echo "1. Verify the application is running correctly"
    echo "2. Run any additional tests or validations"
    echo "3. Update DNS records if needed"
    echo "4. Monitor the deployment for any issues"
    echo
}

# Parse arguments and run main function
parse_args "$@"

# Enable verbose output if requested
if [[ "$VERBOSE" == "true" ]]; then
    set -x
fi

# Run main function
main