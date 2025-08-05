# Project Synapse - Staging Environment Configuration

# Environment
environment = "staging"
aws_region  = "us-east-1"

# Networking
vpc_cidr = "10.1.0.0/16"
public_subnet_cidrs = ["10.1.1.0/24", "10.1.2.0/24"]
private_subnet_cidrs = ["10.1.10.0/24", "10.1.20.0/24"]
enable_nat_gateway = true

# Database Configuration
postgres_version = "15.4"
db_instance_class = "db.t3.small"
db_allocated_storage = 50
db_max_allocated_storage = 200
database_name = "synapse_staging"
database_username = "synapse_staging"
# database_password should be set via environment variable or terraform.tfvars
db_backup_retention_period = 7

# Redis Configuration
redis_node_type = "cache.t3.small"
redis_num_cache_nodes = 1

# Application Configuration
app_image = "project-synapse:staging"
app_port = 8000
app_cpu = 512
app_memory = 1024
app_desired_count = 2
app_min_capacity = 1
app_max_capacity = 4

# Auto Scaling
auto_scaling_target_cpu = 70
auto_scaling_target_memory = 80
auto_scaling_scale_up_cooldown = 300
auto_scaling_scale_down_cooldown = 300

# Monitoring and Logging
log_retention_days = 14
enable_monitoring = true

# Security
allowed_cidr_blocks = ["0.0.0.0/0"]
enable_waf = true
enable_shield = false

# Features
enable_blue_green_deployment = true
enable_canary_deployment = false
enable_multi_region = false
enable_spot_instances = false
enable_scheduled_scaling = false

# Development specific
enable_debug_mode = false
enable_test_data = false

# Compliance
enable_encryption_at_rest = true
enable_encryption_in_transit = true
compliance_mode = "none"

# Backup Configuration
enable_automated_backups = true
backup_schedule = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC

# Domain Configuration
domain_name = "staging.projectsynapse.dev"
# cloudflare_zone_id should be set via environment variable
ssl_certificate_arn = ""  # Will be created automatically

# Resource Tagging
additional_tags = {
  Environment = "staging"
  Purpose     = "pre-production-testing"
  AutoShutdown = "false"
}

cost_center = "engineering"
owner = "devops-team"
project_code = "SYNAPSE-STAGING"

# External Services (set via environment variables)
# cloudflare_api_token = ""
# vercel_api_token = ""