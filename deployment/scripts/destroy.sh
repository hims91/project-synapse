#!/bin/bash

# Project Synapse - Infrastructure Destruction Script
# Safely destroy infrastructure with proper safeguards

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TERRAFORM_DIR="${PROJECT_ROOT}/deployment/infrastructure/terraform"

# Default values
ENVIRONMENT="dev"
AWS_REGION="us-east-1"
FORCE_DESTROY=false
SKIP_BACKUP=false
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
Project Synapse - Infrastructure Destruction Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Target environment (dev, staging, production) [default: dev]
    -r, --region REGION      AWS region [default: us-east-1]
    -f, --force              Force destruction without confirmation
    -s, --skip-backup        Skip creating backup before destruction
    -d, --dry-run            Show what would be destroyed without making changes
    --verbose                Enable verbose output
    -h, --help               Show this help message

EXAMPLES:
    $0                                    # Destroy dev environment (with confirmation)
    $0 -e staging -f                      # Force destroy staging environment
    $0 --dry-run -e production            # Preview what would be destroyed in production

WARNING:
    This script will permanently destroy infrastructure and data.
    Make sure you have proper backups before proceeding.

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
            -f|--force)
                FORCE_DESTROY=true
                shift
                ;;
            -s|--skip-backup)
                SKIP_BACKUP=true
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
            log_info "Target environment for destruction: $ENVIRONMENT"
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

# Show destruction plan
show_destruction_plan() {
    log_info "Analyzing infrastructure to be destroyed..."
    
    cd "$TERRAFORM_DIR"
    
    # Initialize Terraform
    terraform init -upgrade >/dev/null 2>&1
    
    # Select workspace
    if ! terraform workspace select "$ENVIRONMENT" >/dev/null 2>&1; then
        log_error "Environment workspace '$ENVIRONMENT' does not exist"
        exit 1
    fi
    
    # Set Terraform variables
    export TF_VAR_environment="$ENVIRONMENT"
    export TF_VAR_aws_region="$AWS_REGION"
    
    # Show destroy plan
    local var_file="environments/${ENVIRONMENT}.tfvars"
    
    if [[ ! -f "$var_file" ]]; then
        log_error "Environment configuration file not found: $var_file"
        exit 1
    fi
    
    log_info "Resources that will be destroyed:"
    terraform plan -destroy -var-file="$var_file"
    
    # Get resource count
    local resource_count
    resource_count=$(terraform state list 2>/dev/null | wc -l || echo "0")
    
    log_warning "Total resources to be destroyed: $resource_count"
}

# Create backup
create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log_warning "Skipping backup creation"
        return 0
    fi
    
    log_info "Creating backup before destruction..."
    
    cd "$TERRAFORM_DIR"
    
    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d-%H%M%S)-${ENVIRONMENT}"
    mkdir -p "$backup_dir"
    
    # Backup Terraform state
    log_info "Backing up Terraform state..."
    terraform state pull > "${backup_dir}/terraform.tfstate"
    
    # Backup configuration
    log_info "Backing up configuration..."
    cp -r . "${backup_dir}/terraform-config"
    
    # Get resource information
    log_info "Backing up resource information..."
    terraform show -json > "${backup_dir}/terraform-show.json" 2>/dev/null || true
    terraform output -json > "${backup_dir}/terraform-outputs.json" 2>/dev/null || true
    
    # Backup database (if accessible)
    local database_endpoint
    database_endpoint=$(terraform output -raw database_endpoint 2>/dev/null || echo "")
    
    if [[ -n "$database_endpoint" && "$ENVIRONMENT" != "production" ]]; then
        log_info "Creating database backup..."
        # This would typically create a database dump
        # For security, we'll just log that this step would happen
        echo "Database backup would be created here" > "${backup_dir}/database-backup.info"
    fi
    
    # Create backup manifest
    cat > "${backup_dir}/backup-manifest.json" << EOF
{
  "backup_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT",
  "aws_region": "$AWS_REGION",
  "backup_created_by": "$(whoami)",
  "terraform_version": "$(terraform version -json | jq -r '.terraform_version')",
  "aws_account_id": "$(aws sts get-caller-identity --query Account --output text)"
}
EOF
    
    log_success "Backup created at: $backup_dir"
}

# Perform safety checks
perform_safety_checks() {
    log_info "Performing safety checks..."
    
    # Check if this is production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        log_warning "PRODUCTION ENVIRONMENT DETECTED!"
        
        if [[ "$FORCE_DESTROY" != "true" ]]; then
            echo
            log_error "Production environment destruction requires --force flag"
            log_error "This is a safety measure to prevent accidental destruction"
            exit 1
        fi
    fi
    
    # Check for running services
    cd "$TERRAFORM_DIR"
    
    local cluster_name
    cluster_name=$(terraform output -raw ecs_cluster_name 2>/dev/null || echo "")
    
    if [[ -n "$cluster_name" ]]; then
        local running_services
        running_services=$(aws ecs list-services --cluster "$cluster_name" --region "$AWS_REGION" --query 'serviceArns' --output text 2>/dev/null | wc -w || echo "0")
        
        if [[ "$running_services" -gt 0 ]]; then
            log_warning "Found $running_services running services in ECS cluster"
            log_warning "These services will be terminated during destruction"
        fi
    fi
    
    # Check for data in databases
    local database_endpoint
    database_endpoint=$(terraform output -raw database_endpoint 2>/dev/null || echo "")
    
    if [[ -n "$database_endpoint" ]]; then
        log_warning "Database instance will be destroyed"
        log_warning "Ensure you have proper backups of your data"
    fi
    
    log_success "Safety checks completed"
}

# Get user confirmation
get_confirmation() {
    if [[ "$FORCE_DESTROY" == "true" ]]; then
        log_warning "Force mode enabled - skipping confirmation"
        return 0
    fi
    
    echo
    log_warning "⚠️  DANGER: INFRASTRUCTURE DESTRUCTION ⚠️"
    echo
    echo "You are about to PERMANENTLY DESTROY the following:"
    echo "- Environment: $ENVIRONMENT"
    echo "- AWS Region: $AWS_REGION"
    echo "- All infrastructure resources"
    echo "- All data (databases, storage, etc.)"
    echo
    log_warning "This action CANNOT be undone!"
    echo
    
    # Multiple confirmations for production
    if [[ "$ENVIRONMENT" == "production" ]]; then
        echo "This is a PRODUCTION environment!"
        read -p "Type 'DESTROY PRODUCTION' to confirm: " -r
        if [[ "$REPLY" != "DESTROY PRODUCTION" ]]; then
            log_info "Destruction cancelled"
            exit 0
        fi
        
        read -p "Are you absolutely sure? Type 'YES' to proceed: " -r
        if [[ "$REPLY" != "YES" ]]; then
            log_info "Destruction cancelled"
            exit 0
        fi
    else
        read -p "Type 'DESTROY' to confirm destruction: " -r
        if [[ "$REPLY" != "DESTROY" ]]; then
            log_info "Destruction cancelled"
            exit 0
        fi
    fi
    
    log_warning "Proceeding with destruction in 10 seconds..."
    log_warning "Press Ctrl+C to cancel"
    sleep 10
}

# Destroy infrastructure
destroy_infrastructure() {
    log_info "Starting infrastructure destruction..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "Dry run mode - infrastructure would be destroyed here"
        return 0
    fi
    
    cd "$TERRAFORM_DIR"
    
    # Set Terraform variables
    export TF_VAR_environment="$ENVIRONMENT"
    export TF_VAR_aws_region="$AWS_REGION"
    
    # Destroy with environment-specific variables
    local var_file="environments/${ENVIRONMENT}.tfvars"
    
    log_info "Executing Terraform destroy..."
    terraform destroy -auto-approve -var-file="$var_file"
    
    log_success "Infrastructure destruction completed"
}

# Clean up Terraform workspace
cleanup_workspace() {
    log_info "Cleaning up Terraform workspace..."
    
    cd "$TERRAFORM_DIR"
    
    # Delete workspace (except default)
    if [[ "$ENVIRONMENT" != "default" ]]; then
        terraform workspace select default >/dev/null 2>&1 || true
        terraform workspace delete "$ENVIRONMENT" >/dev/null 2>&1 || true
        log_info "Terraform workspace '$ENVIRONMENT' deleted"
    fi
    
    # Clean up plan files
    rm -f "tfplan-${ENVIRONMENT}" >/dev/null 2>&1 || true
    
    log_success "Workspace cleanup completed"
}

# Create destruction report
create_destruction_report() {
    log_info "Creating destruction report..."
    
    local report_file="${PROJECT_ROOT}/destruction-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "$report_file" << EOF
{
  "destruction_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "environment": "$ENVIRONMENT",
  "aws_region": "$AWS_REGION",
  "destroyed_by": "$(whoami)",
  "destruction_method": "terraform_destroy",
  "backup_created": $([ "$SKIP_BACKUP" == "true" ] && echo "false" || echo "true"),
  "destruction_status": "completed"
}
EOF
    
    log_info "Destruction report saved to: $report_file"
}

# Main destruction function
main() {
    log_info "Starting Project Synapse infrastructure destruction..."
    log_info "Environment: $ENVIRONMENT"
    log_info "AWS Region: $AWS_REGION"
    
    # Run destruction steps
    validate_environment
    check_prerequisites
    show_destruction_plan
    perform_safety_checks
    create_backup
    get_confirmation
    destroy_infrastructure
    cleanup_workspace
    create_destruction_report
    
    log_success "Infrastructure destruction completed!"
    
    echo
    echo "Destruction Summary:"
    echo "- Environment: $ENVIRONMENT"
    echo "- Status: Completed"
    echo "- Backup created: $([ "$SKIP_BACKUP" == "true" ] && echo "No" || echo "Yes")"
    echo
    echo "The infrastructure has been permanently destroyed."
    echo "If you need to recreate it, use the deployment script."
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