#!/bin/bash

# Project Synapse - Rollback Script
# Automated rollback system for failed deployments

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TERRAFORM_DIR="${PROJECT_ROOT}/deployment/infrastructure/terraform"

# Default values
ENVIRONMENT="dev"
AWS_REGION="us-east-1"
ROLLBACK_VERSION=""
FORCE_ROLLBACK=false
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
Project Synapse - Rollback Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, production) [default: dev]
    -r, --region REGION      AWS region [default: us-east-1]
    -v, --version VERSION    Specific version to rollback to (optional)
    -f, --force              Force rollback without confirmation
    -d, --dry-run            Show what would be rolled back without making changes
    --verbose                Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    $0                                    # Rollback dev to previous version
    $0 -e staging                         # Rollback staging to previous version
    $0 -e production -v v1.2.3            # Rollback production to specific version
    $0 --dry-run -e production            # Preview production rollback

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
            -v|--version)
                ROLLBACK_VERSION="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_ROLLBACK=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
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
            log_info "Rolling back environment: $ENVIRONMENT"
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
    command -v jq >/dev/null 2>&1 || missing_tools+=("jq")
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        log_error "AWS credentials not configured"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Get current deployment info
get_current_deployment() {
    log_info "Getting current deployment information..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    terraform init -upgrade >/dev/null 2>&1
    terraform workspace select "$ENVIRONMENT" >/dev/null 2>&1
    
    # Get current state
    local current_image
    current_image=$(terraform show -json | jq -r '.values.root_module.resources[] | select(.type == "aws_ecs_task_definition") | .values.container_definitions' | jq -r '.[0].image' 2>/dev/null || echo "unknown")
    
    log_info "Current deployed image: $current_image"
    
    # Get ECS service info
    local cluster_name
    local service_name
    cluster_name=$(terraform output -raw ecs_cluster_name 2>/dev/null || echo "")
    service_name=$(terraform output -raw ecs_service_name 2>/dev/null || echo "")
    
    if [[ -n "$cluster_name" && -n "$service_name" ]]; then
        log_info "ECS Cluster: $cluster_name"
        log_info "ECS Service: $service_name"
        
        # Get service details
        local service_info
        service_info=$(aws ecs describe-services --cluster "$cluster_name" --services "$service_name" --region "$AWS_REGION" 2>/dev/null || echo "{}")
        
        local running_count
        local desired_count
        running_count=$(echo "$service_info" | jq -r '.services[0].runningCount // 0')
        desired_count=$(echo "$service_info" | jq -r '.services[0].desiredCount // 0')
        
        log_info "Service status: $running_count/$desired_count tasks running"
    fi
}

# List available versions
list_available_versions() {
    log_info "Listing available versions for rollback..."
    
    local aws_account_id
    aws_account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_repo="${aws_account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com/project-synapse"
    
    # List ECR images
    local images
    images=$(aws ecr describe-images --repository-name project-synapse --region "$AWS_REGION" --query 'imageDetails[*].[imageTags[0],imageDigest,imagePushedAt]' --output table 2>/dev/null || echo "No images found")
    
    echo "$images"
    
    # Get task definition revisions
    log_info "Available task definition revisions:"
    local task_family="project-synapse-app-${ENVIRONMENT}"
    aws ecs list-task-definitions --family-prefix "$task_family" --status ACTIVE --region "$AWS_REGION" --query 'taskDefinitionArns' --output table 2>/dev/null || log_warning "No task definitions found"
}

# Determine rollback target
determine_rollback_target() {
    log_info "Determining rollback target..."
    
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        log_info "Using specified version: $ROLLBACK_VERSION"
        return 0
    fi
    
    # Get previous task definition
    local task_family="project-synapse-app-${ENVIRONMENT}"
    local task_definitions
    task_definitions=$(aws ecs list-task-definitions --family-prefix "$task_family" --status ACTIVE --region "$AWS_REGION" --query 'taskDefinitionArns' --output text 2>/dev/null || echo "")
    
    if [[ -z "$task_definitions" ]]; then
        log_error "No task definitions found for rollback"
        exit 1
    fi
    
    # Get the second most recent (previous) task definition
    local previous_task_def
    previous_task_def=$(echo "$task_definitions" | tr ' ' '\n' | sort -V | tail -n 2 | head -n 1)
    
    if [[ -n "$previous_task_def" ]]; then
        log_info "Previous task definition: $previous_task_def"
        ROLLBACK_VERSION="$previous_task_def"
    else
        log_error "Could not determine previous version for rollback"
        exit 1
    fi
}

# Perform ECS rollback
perform_ecs_rollback() {
    log_info "Performing ECS service rollback..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - would rollback ECS service to: $ROLLBACK_VERSION"
        return 0
    fi
    
    cd "$TERRAFORM_DIR"
    
    local cluster_name
    local service_name
    cluster_name=$(terraform output -raw ecs_cluster_name 2>/dev/null || echo "")
    service_name=$(terraform output -raw ecs_service_name 2>/dev/null || echo "")
    
    if [[ -z "$cluster_name" || -z "$service_name" ]]; then
        log_error "Could not determine ECS cluster or service name"
        exit 1
    fi
    
    # Update service with previous task definition
    log_info "Updating ECS service to use task definition: $ROLLBACK_VERSION"
    
    aws ecs update-service \
        --cluster "$cluster_name" \
        --service "$service_name" \
        --task-definition "$ROLLBACK_VERSION" \
        --region "$AWS_REGION" >/dev/null
    
    log_success "ECS service rollback initiated"
}

# Wait for rollback completion
wait_for_rollback() {
    log_info "Waiting for rollback to complete..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - skipping rollback wait"
        return 0
    fi
    
    cd "$TERRAFORM_DIR"
    
    local cluster_name
    local service_name
    cluster_name=$(terraform output -raw ecs_cluster_name 2>/dev/null || echo "")
    service_name=$(terraform output -raw ecs_service_name 2>/dev/null || echo "")
    
    if [[ -z "$cluster_name" || -z "$service_name" ]]; then
        log_warning "Could not determine ECS cluster or service name for monitoring"
        return 0
    fi
    
    # Wait for service to stabilize
    log_info "Waiting for service to stabilize..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        local service_info
        service_info=$(aws ecs describe-services --cluster "$cluster_name" --services "$service_name" --region "$AWS_REGION" 2>/dev/null || echo "{}")
        
        local running_count
        local desired_count
        local deployment_status
        
        running_count=$(echo "$service_info" | jq -r '.services[0].runningCount // 0')
        desired_count=$(echo "$service_info" | jq -r '.services[0].desiredCount // 0')
        deployment_status=$(echo "$service_info" | jq -r '.services[0].deployments[0].status // "UNKNOWN"')
        
        log_info "Rollback progress: $running_count/$desired_count tasks running, deployment status: $deployment_status"
        
        if [[ "$running_count" -eq "$desired_count" && "$deployment_status" == "PRIMARY" ]]; then
            log_success "Rollback completed successfully"
            return 0
        fi
        
        sleep 30
        ((attempt++))
    done
    
    log_warning "Rollback did not complete within expected time"
    log_warning "Please check the ECS console for deployment status"
}

# Perform health check
health_check() {
    log_info "Performing post-rollback health check..."
    
    cd "$TERRAFORM_DIR"
    
    local alb_dns_name
    alb_dns_name=$(terraform output -raw alb_dns_name 2>/dev/null || echo "")
    
    if [[ -n "$alb_dns_name" ]]; then
        local health_url="http://${alb_dns_name}/health"
        local max_attempts=10
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
        
        log_error "Health check failed after rollback"
        log_error "The application may not be functioning correctly"
        return 1
    else
        log_warning "Could not determine load balancer DNS name for health check"
    fi
}

# Create rollback report
create_rollback_report() {
    log_info "Creating rollback report..."
    
    local report_file="${PROJECT_ROOT}/rollback-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "rollback_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT",
  "aws_region": "$AWS_REGION",
  "rollback_version": "$ROLLBACK_VERSION",
  "rollback_initiated_by": "$(whoami)",
  "rollback_reason": "Manual rollback via script",
  "rollback_status": "completed"
}
EOF
    
    log_info "Rollback report saved to: $report_file"
}

# Main rollback function
main() {
    log_info "Starting Project Synapse rollback..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Confirmation for production
    if [[ "$ENVIRONMENT" == "production" && "$FORCE_ROLLBACK" != "true" ]]; then
        echo
        log_warning "You are about to rollback PRODUCTION environment!"
        read -p "Are you sure you want to continue? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            log_info "Rollback cancelled"
            exit 0
        fi
    fi
    
    # Run rollback steps
    validate_environment
    check_prerequisites
    get_current_deployment
    list_available_versions
    determine_rollback_target
    perform_ecs_rollback
    wait_for_rollback
    
    # Health check (non-blocking)
    if ! health_check; then
        log_warning "Health check failed, but rollback was completed"
        log_warning "Please investigate the application status manually"
    fi
    
    create_rollback_report
    
    log_success "Rollback completed!"
    
    echo
    echo "Rollback Summary:"
    echo "- Environment: $ENVIRONMENT"
    echo "- Rolled back to: $ROLLBACK_VERSION"
    echo "- Status: Completed"
    echo
    echo "Next steps:"
    echo "1. Verify the application is functioning correctly"
    echo "2. Investigate the root cause of the issue that required rollback"
    echo "3. Plan a fix and re-deployment"
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