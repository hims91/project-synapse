#!/bin/bash

# Project Synapse - Environment Setup Script
# Initial setup and configuration for deployment

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Default values
ENVIRONMENT="dev"
AWS_REGION="us-east-1"
SKIP_AWS_SETUP=false
SKIP_DOCKER_SETUP=false
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
Project Synapse - Environment Setup Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, production) [default: dev]
    -r, --region REGION      AWS region [default: us-east-1]
    --skip-aws               Skip AWS setup
    --skip-docker            Skip Docker setup
    -v, --verbose            Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    $0                       # Setup dev environment
    $0 -e staging            # Setup staging environment
    $0 --skip-aws            # Setup without AWS configuration

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
            --skip-aws)
                SKIP_AWS_SETUP=true
                shift
                ;;
            --skip-docker)
                SKIP_DOCKER_SETUP=true
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

# Check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    local missing_tools=()
    
    # Check required tools
    command -v python3 >/dev/null 2>&1 || missing_tools+=("python3")
    command -v pip3 >/dev/null 2>&1 || missing_tools+=("pip3")
    command -v git >/dev/null 2>&1 || missing_tools+=("git")
    command -v curl >/dev/null 2>&1 || missing_tools+=("curl")
    
    if [[ "$SKIP_DOCKER_SETUP" != "true" ]]; then
        command -v docker >/dev/null 2>&1 || missing_tools+=("docker")
        command -v docker-compose >/dev/null 2>&1 || missing_tools+=("docker-compose")
    fi
    
    if [[ "$SKIP_AWS_SETUP" != "true" ]]; then
        command -v aws >/dev/null 2>&1 || missing_tools+=("aws-cli")
        command -v terraform >/dev/null 2>&1 || missing_tools+=("terraform")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and run this script again."
        
        # Provide installation hints
        echo
        log_info "Installation hints:"
        for tool in "${missing_tools[@]}"; do
            case $tool in
                python3)
                    echo "  - Python 3: https://www.python.org/downloads/"
                    ;;
                pip3)
                    echo "  - pip3: Usually comes with Python 3"
                    ;;
                docker)
                    echo "  - Docker: https://docs.docker.com/get-docker/"
                    ;;
                docker-compose)
                    echo "  - Docker Compose: https://docs.docker.com/compose/install/"
                    ;;
                aws-cli)
                    echo "  - AWS CLI: https://aws.amazon.com/cli/"
                    ;;
                terraform)
                    echo "  - Terraform: https://www.terraform.io/downloads"
                    ;;
            esac
        done
        
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Setup Python environment
setup_python_environment() {
    log_info "Setting up Python environment..."
    
    cd "$PROJECT_ROOT"
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing Python dependencies..."
        pip install -r requirements.txt
    fi
    
    if [[ -f "requirements-dev.txt" ]]; then
        log_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    log_success "Python environment setup completed"
}

# Setup AWS configuration
setup_aws_configuration() {
    if [[ "$SKIP_AWS_SETUP" == "true" ]]; then
        log_warning "Skipping AWS setup"
        return 0
    fi
    
    log_info "Setting up AWS configuration..."
    
    # Check if AWS credentials are configured
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_warning "AWS credentials not configured"
        log_info "Please run 'aws configure' to set up your AWS credentials"
        
        read -p "Do you want to configure AWS credentials now? (y/n): " -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            aws configure
        else
            log_warning "AWS setup skipped - you'll need to configure credentials later"
            return 0
        fi
    fi
    
    # Verify AWS access
    local aws_account_id
    aws_account_id=$(aws sts get-caller-identity --query Account --output text)
    log_info "AWS Account ID: $aws_account_id"
    
    # Create S3 bucket for Terraform state (if it doesn't exist)
    local state_bucket="synapse-terraform-state-${aws_account_id}"
    
    if ! aws s3 ls "s3://${state_bucket}" >/dev/null 2>&1; then
        log_info "Creating S3 bucket for Terraform state..."
        aws s3 mb "s3://${state_bucket}" --region "$AWS_REGION"
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "$state_bucket" \
            --versioning-configuration Status=Enabled
        
        # Enable encryption
        aws s3api put-bucket-encryption \
            --bucket "$state_bucket" \
            --server-side-encryption-configuration '{
                "Rules": [
                    {
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }
                ]
            }'
    fi
    
    log_success "AWS configuration completed"
}

# Setup Docker environment
setup_docker_environment() {
    if [[ "$SKIP_DOCKER_SETUP" == "true" ]]; then
        log_warning "Skipping Docker setup"
        return 0
    fi
    
    log_info "Setting up Docker environment..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    # Build development image
    log_info "Building Docker development image..."
    cd "$PROJECT_ROOT"
    docker build -t project-synapse:dev --target development .
    
    log_success "Docker environment setup completed"
}

# Create environment configuration
create_environment_config() {
    log_info "Creating environment configuration..."
    
    local config_dir="${PROJECT_ROOT}/.env"
    mkdir -p "$config_dir"
    
    local env_file="${config_dir}/${ENVIRONMENT}.env"
    
    if [[ ! -f "$env_file" ]]; then
        log_info "Creating environment file: $env_file"
        
        cat > "$env_file" << EOF
# Project Synapse - ${ENVIRONMENT} Environment Configuration
# Generated on $(date)

# Environment
ENVIRONMENT=${ENVIRONMENT}
AWS_REGION=${AWS_REGION}

# Database Configuration
DATABASE_URL=postgresql://synapse:synapse@localhost:5432/synapse_${ENVIRONMENT}
DATABASE_PASSWORD=synapse

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Application Configuration
SECRET_KEY=$(openssl rand -hex 32)
LOG_LEVEL=$([ "$ENVIRONMENT" == "production" ] && echo "INFO" || echo "DEBUG")
CORS_ORIGINS=*
RATE_LIMIT_ENABLED=true
METRICS_ENABLED=true

# External Services (configure as needed)
# CLOUDFLARE_API_TOKEN=
# VERCEL_API_TOKEN=
# SSL_CERTIFICATE_ARN=

# Feature Flags
ENABLE_DEBUG_MODE=$([ "$ENVIRONMENT" == "dev" ] && echo "true" || echo "false")
ENABLE_TEST_DATA=$([ "$ENVIRONMENT" == "dev" ] && echo "true" || echo "false")
EOF
        
        log_info "Environment file created. Please review and update as needed."
    else
        log_info "Environment file already exists: $env_file"
    fi
    
    # Create .env symlink for convenience
    local env_symlink="${PROJECT_ROOT}/.env"
    if [[ ! -L "$env_symlink" ]]; then
        ln -sf ".env/${ENVIRONMENT}.env" "$env_symlink"
        log_info "Created .env symlink pointing to ${ENVIRONMENT}.env"
    fi
    
    log_success "Environment configuration completed"
}

# Setup development tools
setup_development_tools() {
    log_info "Setting up development tools..."
    
    cd "$PROJECT_ROOT"
    
    # Setup pre-commit hooks (if available)
    if command -v pre-commit >/dev/null 2>&1 && [[ -f ".pre-commit-config.yaml" ]]; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi
    
    # Create useful scripts
    local scripts_dir="${PROJECT_ROOT}/scripts"
    mkdir -p "$scripts_dir"
    
    # Create development start script
    cat > "${scripts_dir}/dev-start.sh" << 'EOF'
#!/bin/bash
# Start development environment
cd "$(dirname "$0")/.."
source venv/bin/activate
export $(cat .env | xargs)
python -m uvicorn src.axon_interface.main:app --reload --host 0.0.0.0 --port 8000
EOF
    
    # Create test script
    cat > "${scripts_dir}/test.sh" << 'EOF'
#!/bin/bash
# Run tests
cd "$(dirname "$0")/.."
source venv/bin/activate
export $(cat .env | xargs)
python -m pytest tests/ -v
EOF
    
    # Make scripts executable
    chmod +x "${scripts_dir}"/*.sh
    
    log_success "Development tools setup completed"
}

# Verify setup
verify_setup() {
    log_info "Verifying setup..."
    
    local issues=()
    
    # Check Python environment
    if [[ ! -d "${PROJECT_ROOT}/venv" ]]; then
        issues+=("Python virtual environment not found")
    fi
    
    # Check environment configuration
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        issues+=("Environment configuration not found")
    fi
    
    # Check AWS configuration (if not skipped)
    if [[ "$SKIP_AWS_SETUP" != "true" ]] && ! aws sts get-caller-identity >/dev/null 2>&1; then
        issues+=("AWS credentials not configured")
    fi
    
    # Check Docker (if not skipped)
    if [[ "$SKIP_DOCKER_SETUP" != "true" ]] && ! docker info >/dev/null 2>&1; then
        issues+=("Docker not running")
    fi
    
    if [[ ${#issues[@]} -gt 0 ]]; then
        log_warning "Setup verification found issues:"
        for issue in "${issues[@]}"; do
            echo "  - $issue"
        done
        log_warning "Please address these issues before proceeding"
    else
        log_success "Setup verification passed"
    fi
}

# Main setup function
main() {
    log_info "Starting Project Synapse environment setup..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Run setup steps
    check_system_requirements
    setup_python_environment
    setup_aws_configuration
    setup_docker_environment
    create_environment_config
    setup_development_tools
    verify_setup
    
    log_success "Environment setup completed!"
    
    echo
    echo "Setup Summary:"
    echo "- Environment: $ENVIRONMENT"
    echo "- Python virtual environment: Created"
    echo "- AWS configuration: $([ "$SKIP_AWS_SETUP" == "true" ] && echo "Skipped" || echo "Configured")"
    echo "- Docker environment: $([ "$SKIP_DOCKER_SETUP" == "true" ] && echo "Skipped" || echo "Configured")"
    echo
    echo "Next steps:"
    echo "1. Review and update the environment configuration in .env/${ENVIRONMENT}.env"
    echo "2. Start the development environment with: ./scripts/dev-start.sh"
    echo "3. Run tests with: ./scripts/test.sh"
    echo "4. Deploy with: ./deployment/scripts/deploy.sh -e ${ENVIRONMENT}"
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