#!/bin/bash
set -e

# Project Synapse Learning Scrapers - GitHub Actions Deployment Script
# Manages deployment and execution of learning scrapers in GitHub Actions

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
WORKFLOW_FILE=".github/workflows/learning-scrapers.yml"

# Default values
ACTION=""
URLS=""
SCRAPER_TYPE="playwright"
LEARNING_MODE="true"
USE_PROXY="false"
MAX_CONCURRENT="3"
TIMEOUT_MINUTES="30"

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
Project Synapse Learning Scrapers - GitHub Actions Deployment

Usage: $0 [OPTIONS] ACTION

ACTIONS:
    trigger         Trigger learning scraper workflow
    status          Check workflow status
    logs            Get workflow logs
    artifacts       Download workflow artifacts
    setup           Setup GitHub Actions secrets
    validate        Validate workflow configuration
    monitor         Monitor running workflows

OPTIONS:
    -u, --urls URLS         Comma-separated URLs to scrape
    -t, --type TYPE         Scraper type (playwright|http|both) [default: playwright]
    -l, --learning BOOL     Enable learning mode [default: true]
    -p, --proxy BOOL        Use proxy network [default: false]
    -c, --concurrent NUM    Max concurrent scrapers [default: 3]
    --timeout MINUTES       Timeout in minutes [default: 30]
    --repo REPO             GitHub repository (owner/repo)
    --token TOKEN           GitHub token
    -v, --verbose           Verbose output
    -h, --help              Show this help

EXAMPLES:
    $0 trigger -u "https://example.com,https://test.com" -t playwright
    $0 status --repo owner/repo
    $0 logs --repo owner/repo
    $0 setup --repo owner/repo --token ghp_xxx
    $0 artifacts --repo owner/repo

ENVIRONMENT VARIABLES:
    GITHUB_TOKEN            GitHub personal access token
    GITHUB_REPOSITORY       Repository in format owner/repo
    GITHUB_WORKFLOW_ID      Workflow ID or filename

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--urls)
                URLS="$2"
                shift 2
                ;;
            -t|--type)
                SCRAPER_TYPE="$2"
                shift 2
                ;;
            -l|--learning)
                LEARNING_MODE="$2"
                shift 2
                ;;
            -p|--proxy)
                USE_PROXY="$2"
                shift 2
                ;;
            -c|--concurrent)
                MAX_CONCURRENT="$2"
                shift 2
                ;;
            --timeout)
                TIMEOUT_MINUTES="$2"
                shift 2
                ;;
            --repo)
                GITHUB_REPOSITORY="$2"
                shift 2
                ;;
            --token)
                GITHUB_TOKEN="$2"
                shift 2
                ;;
            -v|--verbose)
                DEBUG="true"
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
}

# Validate environment
validate_environment() {
    log_info "Validating environment..."
    
    # Check required tools
    local required_tools=("curl" "jq" "gh")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Install missing tools:"
        for tool in "${missing_tools[@]}"; do
            case "$tool" in
                gh)
                    log_info "  GitHub CLI: https://cli.github.com/"
                    ;;
                jq)
                    log_info "  jq: https://stedolan.github.io/jq/"
                    ;;
                curl)
                    log_info "  curl: Usually pre-installed"
                    ;;
            esac
        done
        exit 1
    fi
    
    # Check GitHub token
    if [[ -z "$GITHUB_TOKEN" ]]; then
        log_error "GITHUB_TOKEN environment variable is required"
        log_info "Create a personal access token at: https://github.com/settings/tokens"
        exit 1
    fi
    
    # Check repository
    if [[ -z "$GITHUB_REPOSITORY" ]]; then
        # Try to detect from git remote
        if git remote get-url origin &> /dev/null; then
            local remote_url=$(git remote get-url origin)
            GITHUB_REPOSITORY=$(echo "$remote_url" | sed -n 's/.*github\.com[:/]\([^/]*\/[^/]*\)\.git.*/\1/p')
        fi
        
        if [[ -z "$GITHUB_REPOSITORY" ]]; then
            log_error "GITHUB_REPOSITORY is required (format: owner/repo)"
            exit 1
        fi
    fi
    
    log_debug "GitHub Repository: $GITHUB_REPOSITORY"
    log_debug "GitHub Token: ${GITHUB_TOKEN:0:10}..."
}

# GitHub API helper
github_api() {
    local method="$1"
    local endpoint="$2"
    local data="$3"
    
    local url="https://api.github.com/repos/$GITHUB_REPOSITORY/$endpoint"
    
    local curl_args=(
        -X "$method"
        -H "Authorization: token $GITHUB_TOKEN"
        -H "Accept: application/vnd.github.v3+json"
        -H "Content-Type: application/json"
        -s
    )
    
    if [[ -n "$data" ]]; then
        curl_args+=(-d "$data")
    fi
    
    log_debug "API call: $method $url"
    curl "${curl_args[@]}" "$url"
}

# Trigger workflow
trigger_workflow() {
    log_info "Triggering learning scraper workflow..."
    
    # Prepare workflow inputs
    local inputs="{
        \"target_urls\": \"$URLS\",
        \"scraper_type\": \"$SCRAPER_TYPE\",
        \"learning_mode\": $LEARNING_MODE,
        \"use_proxy\": $USE_PROXY,
        \"max_concurrent\": \"$MAX_CONCURRENT\",
        \"timeout_minutes\": \"$TIMEOUT_MINUTES\"
    }"
    
    local payload="{
        \"ref\": \"main\",
        \"inputs\": $inputs
    }"
    
    log_debug "Workflow inputs: $inputs"
    
    # Trigger workflow
    local response=$(github_api "POST" "actions/workflows/learning-scrapers.yml/dispatches" "$payload")
    
    if [[ $? -eq 0 ]]; then
        log_info "Workflow triggered successfully"
        
        # Wait a moment and get the latest run
        sleep 3
        get_workflow_status
    else
        log_error "Failed to trigger workflow"
        echo "$response" | jq -r '.message // .'
        exit 1
    fi
}

# Get workflow status
get_workflow_status() {
    log_info "Getting workflow status..."
    
    local response=$(github_api "GET" "actions/workflows/learning-scrapers.yml/runs?per_page=5")
    
    if [[ $? -eq 0 ]]; then
        echo "$response" | jq -r '
            .workflow_runs[] | 
            "ID: \(.id) | Status: \(.status) | Conclusion: \(.conclusion // "N/A") | Created: \(.created_at) | URL: \(.html_url)"
        ' | head -5
    else
        log_error "Failed to get workflow status"
        echo "$response" | jq -r '.message // .'
        exit 1
    fi
}

# Get workflow logs
get_workflow_logs() {
    log_info "Getting workflow logs..."
    
    # Get latest workflow run
    local latest_run=$(github_api "GET" "actions/workflows/learning-scrapers.yml/runs?per_page=1")
    local run_id=$(echo "$latest_run" | jq -r '.workflow_runs[0].id')
    
    if [[ "$run_id" == "null" || -z "$run_id" ]]; then
        log_error "No workflow runs found"
        exit 1
    fi
    
    log_info "Getting logs for run ID: $run_id"
    
    # Get logs
    local logs_url="https://api.github.com/repos/$GITHUB_REPOSITORY/actions/runs/$run_id/logs"
    
    curl -L \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "$logs_url" \
        -o "workflow_logs_$run_id.zip"
    
    if [[ $? -eq 0 ]]; then
        log_info "Logs downloaded to: workflow_logs_$run_id.zip"
        
        # Extract and display recent logs
        if command -v unzip &> /dev/null; then
            unzip -q "workflow_logs_$run_id.zip" -d "logs_$run_id"
            log_info "Logs extracted to: logs_$run_id/"
            
            # Show recent log entries
            find "logs_$run_id" -name "*.txt" -exec tail -20 {} \; 2>/dev/null | head -50
        fi
    else
        log_error "Failed to download logs"
        exit 1
    fi
}

# Download artifacts
download_artifacts() {
    log_info "Downloading workflow artifacts..."
    
    # Get latest workflow run
    local latest_run=$(github_api "GET" "actions/workflows/learning-scrapers.yml/runs?per_page=1")
    local run_id=$(echo "$latest_run" | jq -r '.workflow_runs[0].id')
    
    if [[ "$run_id" == "null" || -z "$run_id" ]]; then
        log_error "No workflow runs found"
        exit 1
    fi
    
    log_info "Getting artifacts for run ID: $run_id"
    
    # Get artifacts list
    local artifacts=$(github_api "GET" "actions/runs/$run_id/artifacts")
    
    echo "$artifacts" | jq -r '.artifacts[] | "\(.id) \(.name) \(.size_in_bytes)"' | while read -r artifact_id name size; do
        log_info "Downloading artifact: $name ($size bytes)"
        
        local download_url="https://api.github.com/repos/$GITHUB_REPOSITORY/actions/artifacts/$artifact_id/zip"
        
        curl -L \
            -H "Authorization: token $GITHUB_TOKEN" \
            -H "Accept: application/vnd.github.v3+json" \
            "$download_url" \
            -o "${name}.zip"
        
        if [[ $? -eq 0 ]]; then
            log_info "Downloaded: ${name}.zip"
        else
            log_warn "Failed to download: $name"
        fi
    done
}

# Setup GitHub Actions secrets
setup_secrets() {
    log_info "Setting up GitHub Actions secrets..."
    
    # List of secrets to set up
    local secrets=(
        "DATABASE_URL:Database connection string"
        "REDIS_URL:Redis connection string"
        "PROXY_API_KEY:Proxy service API key (optional)"
        "NOTIFICATION_WEBHOOK:Webhook URL for notifications (optional)"
    )
    
    for secret_info in "${secrets[@]}"; do
        local secret_name="${secret_info%%:*}"
        local secret_desc="${secret_info##*:}"
        
        echo
        log_info "Setting up secret: $secret_name"
        log_info "Description: $secret_desc"
        
        read -p "Enter value for $secret_name (or press Enter to skip): " -s secret_value
        echo
        
        if [[ -n "$secret_value" ]]; then
            # Use GitHub CLI to set secret
            if command -v gh &> /dev/null; then
                echo "$secret_value" | gh secret set "$secret_name" --repo "$GITHUB_REPOSITORY"
                log_info "Secret $secret_name set successfully"
            else
                log_warn "GitHub CLI not available. Please set secret manually in repository settings."
            fi
        else
            log_info "Skipped $secret_name"
        fi
    done
    
    log_info "Secret setup completed"
}

# Validate workflow configuration
validate_workflow() {
    log_info "Validating workflow configuration..."
    
    # Check if workflow file exists
    if [[ ! -f "$PROJECT_ROOT/$WORKFLOW_FILE" ]]; then
        log_error "Workflow file not found: $WORKFLOW_FILE"
        exit 1
    fi
    
    # Validate YAML syntax
    if command -v yamllint &> /dev/null; then
        yamllint "$PROJECT_ROOT/$WORKFLOW_FILE"
        if [[ $? -eq 0 ]]; then
            log_info "Workflow YAML syntax is valid"
        else
            log_error "Workflow YAML syntax errors found"
            exit 1
        fi
    else
        log_warn "yamllint not available, skipping YAML validation"
    fi
    
    # Check workflow in repository
    local workflow_check=$(github_api "GET" "actions/workflows/learning-scrapers.yml")
    
    if echo "$workflow_check" | jq -e '.id' > /dev/null; then
        log_info "Workflow found in repository"
        echo "$workflow_check" | jq -r '"Name: \(.name) | State: \(.state) | Created: \(.created_at)"'
    else
        log_error "Workflow not found in repository"
        log_info "Make sure the workflow file is committed and pushed to the repository"
        exit 1
    fi
}

# Monitor workflows
monitor_workflows() {
    log_info "Monitoring workflow runs..."
    
    while true; do
        clear
        echo "=== Learning Scrapers Workflow Monitor ==="
        echo "Repository: $GITHUB_REPOSITORY"
        echo "Time: $(date)"
        echo
        
        # Get recent runs
        local runs=$(github_api "GET" "actions/workflows/learning-scrapers.yml/runs?per_page=10")
        
        echo "$runs" | jq -r '
            .workflow_runs[] | 
            "\(.id) | \(.status) | \(.conclusion // "N/A") | \(.created_at) | \(.head_commit.message // "N/A")"
        ' | while IFS='|' read -r id status conclusion created message; do
            printf "%-12s | %-12s | %-12s | %-20s | %s\n" \
                "${id// /}" "${status// /}" "${conclusion// /}" "${created// /}" "${message// /}"
        done
        
        echo
        echo "Press Ctrl+C to exit monitoring"
        sleep 30
    done
}

# Main execution
main() {
    parse_args "$@"
    validate_environment
    
    log_info "Project Synapse Learning Scrapers - GitHub Actions Deployment"
    log_info "Action: $ACTION"
    log_info "Repository: $GITHUB_REPOSITORY"
    
    case "$ACTION" in
        trigger)
            trigger_workflow
            ;;
        status)
            get_workflow_status
            ;;
        logs)
            get_workflow_logs
            ;;
        artifacts)
            download_artifacts
            ;;
        setup)
            setup_secrets
            ;;
        validate)
            validate_workflow
            ;;
        monitor)
            monitor_workflows
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