# Project Synapse - Development Environment Configuration

# Environment
environment = "dev"
aws_region  = "us-east-1"

# Networking
vpc_cidr = "10.0.0.0/16"
public_subnet_cidrs = ["10.0.1.0/24", "10.0.2.0/24"]
private_subnet_cidrs = ["10.0.10.0/24", "10.0.20.0/24"]
enable_nat_gateway = false  # Cost optimization for dev

# Database Configuration
postgres_version = "15.4"
db_instance_class = "db.t3.micro"
db_allocated_storage = 20
db_max_allocated_storage = 50
database_name = "synapse_dev"
database_username = "synapse_dev"
# database_password should be set via environment variable or terraform.tfvars
db_backup_retention_period = 1  # Minimal backup for dev

# Redis Configuration
redis_node_type = "cache.t3.micro"
redis_num_cache_nodes = 1

# Application Configuration
app_image = "project-synapse:dev"
app_port = 8000
app_cpu = 256
app_memory = 512
app_desired_count = 1
app_min_capacity = 1
app_max_capacity = 2

# Auto Scaling
auto_scaling_target_cpu = 80
auto_scaling_target_memory = 85
auto_scaling_scale_up_cooldown = 300
auto_scaling_scale_down_cooldown = 300

# Monitoring and Logging
log_retention_days = 7
enable_monitoring = true

# Security
allowed_cidr_blocks = ["0.0.0.0/0"]  # Open for development
enable_waf = false
enable_shield = false

# Features
enable_blue_green_deployment = false
enable_canary_deployment = false
enable_multi_region = false
enable_spot_instances = true  # Cost optimization
enable_scheduled_scaling = false

# Development specific
enable_debug_mode = true
enable_test_data = true

# Compliance
enable_encryption_at_rest = true
enable_encryption_in_transit = true
compliance_mode = "none"

# Cost Optimization
enable_automated_backups = false  # Disable for dev to save costs

# Domain Configuration (optional for dev)
domain_name = ""
cloudflare_zone_id = ""
ssl_certificate_arn = ""

# Resource Tagging
additional_tags = {
  Environment = "development"
  Purpose     = "testing"
  AutoShutdown = "true"
}

cost_center = "engineering"
owner = "development-team"
project_code = "SYNAPSE-DEV"

# External Services (set via environment variables)
# cloudflare_api_token = ""
# vercel_api_token = ""