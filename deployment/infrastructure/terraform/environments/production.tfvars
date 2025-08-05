# Project Synapse - Production Environment Configuration

# Environment
environment = "production"
aws_region  = "us-east-1"

# Networking
vpc_cidr = "10.2.0.0/16"
public_subnet_cidrs = ["10.2.1.0/24", "10.2.2.0/24", "10.2.3.0/24"]
private_subnet_cidrs = ["10.2.10.0/24", "10.2.20.0/24", "10.2.30.0/24"]
enable_nat_gateway = true

# Database Configuration
postgres_version = "15.4"
db_instance_class = "db.t3.medium"
db_allocated_storage = 100
db_max_allocated_storage = 1000
database_name = "synapse_production"
database_username = "synapse_prod"
# database_password should be set via environment variable or terraform.tfvars
db_backup_retention_period = 30

# Redis Configuration
redis_node_type = "cache.t3.medium"
redis_num_cache_nodes = 2

# Application Configuration
app_image = "project-synapse:latest"
app_port = 8000
app_cpu = 1024
app_memory = 2048
app_desired_count = 3
app_min_capacity = 2
app_max_capacity = 10

# Auto Scaling
auto_scaling_target_cpu = 60
auto_scaling_target_memory = 70
auto_scaling_scale_up_cooldown = 300
auto_scaling_scale_down_cooldown = 600

# Monitoring and Logging
log_retention_days = 90
enable_monitoring = true

# Security
allowed_cidr_blocks = ["0.0.0.0/0"]
enable_waf = true
enable_shield = true

# Features
enable_blue_green_deployment = true
enable_canary_deployment = true
enable_multi_region = false  # Can be enabled for global deployment
enable_spot_instances = false  # Reliability over cost in production
enable_scheduled_scaling = true

# Development specific
enable_debug_mode = false
enable_test_data = false

# Compliance
enable_encryption_at_rest = true
enable_encryption_in_transit = true
compliance_mode = "none"  # Set to "hipaa", "pci", or "sox" as needed

# Backup Configuration
enable_automated_backups = true
backup_schedule = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC

# Domain Configuration
domain_name = "api.projectsynapse.com"
# cloudflare_zone_id should be set via environment variable
# ssl_certificate_arn should be set via environment variable

# Resource Tagging
additional_tags = {
  Environment = "production"
  Purpose     = "production-workload"
  AutoShutdown = "false"
  CriticalSystem = "true"
}

cost_center = "production"
owner = "platform-team"
project_code = "SYNAPSE-PROD"

# External Services (set via environment variables)
# cloudflare_api_token = ""
# vercel_api_token = ""