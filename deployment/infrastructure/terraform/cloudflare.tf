# Cloudflare Infrastructure for Project Synapse
# Manages Workers, R2 Storage, DNS, and CDN

# Provider configuration
provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

# Variables
variable "cloudflare_api_token" {
  description = "Cloudflare API Token"
  type        = string
  sensitive   = true
}

variable "cloudflare_account_id" {
  description = "Cloudflare Account ID"
  type        = string
}

variable "r2_bucket_name" {
  description = "R2 bucket name for fallback storage"
  type        = string
  default     = "synapse-fallback-storage"
}

# R2 Bucket for Spinal Cord (Fallback System)
resource "cloudflare_r2_bucket" "fallback_storage" {
  account_id = var.cloudflare_account_id
  name       = var.r2_bucket_name
  location   = "auto"
}

# KV Namespaces for Workers
resource "cloudflare_workers_kv_namespace" "feed_cache" {
  account_id = var.cloudflare_account_id
  title      = "synapse-feed-cache"
}

resource "cloudflare_workers_kv_namespace" "task_queue" {
  account_id = var.cloudflare_account_id
  title      = "synapse-task-queue"
}

resource "cloudflare_workers_kv_namespace" "metrics" {
  account_id = var.cloudflare_account_id
  title      = "synapse-metrics"
}

# Dendrites - Feed Poller Worker
resource "cloudflare_worker_script" "feed_poller" {
  account_id = var.cloudflare_account_id
  name       = "synapse-feed-poller"
  content    = file("${path.module}/../../cloudflare/workers/feed-poller/src/index.js")
  
  kv_namespace_binding {
    name         = "FEED_CACHE"
    namespace_id = cloudflare_workers_kv_namespace.feed_cache.id
  }
  
  kv_namespace_binding {
    name         = "METRICS"
    namespace_id = cloudflare_workers_kv_namespace.metrics.id
  }
  
  r2_bucket_binding {
    name        = "FALLBACK_STORAGE"
    bucket_name = cloudflare_r2_bucket.fallback_storage.name
  }
  
  plain_text_binding {
    name  = "ENVIRONMENT"
    text  = var.environment
  }
  
  secret_text_binding {
    name = "API_BASE_URL"
    text = "https://${var.api_domain}"
  }
}

# Signal Relay - Task Dispatcher Worker
resource "cloudflare_worker_script" "task_dispatcher" {
  account_id = var.cloudflare_account_id
  name       = "synapse-task-dispatcher"
  content    = file("${path.module}/../../cloudflare/workers/task-dispatcher/src/index.js")
  
  kv_namespace_binding {
    name         = "TASK_QUEUE"
    namespace_id = cloudflare_workers_kv_namespace.task_queue.id
  }
  
  kv_namespace_binding {
    name         = "METRICS"
    namespace_id = cloudflare_workers_kv_namespace.metrics.id
  }
  
  r2_bucket_binding {
    name        = "FALLBACK_STORAGE"
    bucket_name = cloudflare_r2_bucket.fallback_storage.name
  }
  
  plain_text_binding {
    name  = "ENVIRONMENT"
    text  = var.environment
  }
  
  secret_text_binding {
    name = "API_BASE_URL"
    text = "https://${var.api_domain}"
  }
  
  secret_text_binding {
    name = "GITHUB_TOKEN"
    text = var.github_token
  }
}

# Cron Triggers for Feed Poller
resource "cloudflare_worker_cron_trigger" "feed_poller_high_priority" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.feed_poller.name
  cron        = "*/5 * * * *"  # Every 5 minutes for high priority feeds
}

resource "cloudflare_worker_cron_trigger" "feed_poller_normal_priority" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.feed_poller.name
  cron        = "*/15 * * * *"  # Every 15 minutes for normal priority feeds
}

resource "cloudflare_worker_cron_trigger" "feed_poller_low_priority" {
  account_id  = var.cloudflare_account_id
  script_name = cloudflare_worker_script.feed_poller.name
  cron        = "0 * * * *"  # Every hour for low priority feeds
}

# Custom domains for Workers
resource "cloudflare_worker_domain" "feed_poller" {
  account_id = var.cloudflare_account_id
  hostname   = "feeds.${var.domain_name}"
  service    = cloudflare_worker_script.feed_poller.name
  zone_id    = data.cloudflare_zone.main.id
}

resource "cloudflare_worker_domain" "task_dispatcher" {
  account_id = var.cloudflare_account_id
  hostname   = "tasks.${var.domain_name}"
  service    = cloudflare_worker_script.task_dispatcher.name
  zone_id    = data.cloudflare_zone.main.id
}

# DNS Records
resource "cloudflare_record" "api" {
  zone_id = data.cloudflare_zone.main.id
  name    = "api"
  value   = render_service.central_cortex.service_url
  type    = "CNAME"
  proxied = true
  
  depends_on = [render_service.central_cortex]
}

resource "cloudflare_record" "dashboard" {
  zone_id = data.cloudflare_zone.main.id
  name    = "@"
  value   = render_static_site.frontend.service_url
  type    = "CNAME"
  proxied = true
  
  depends_on = [render_static_site.frontend]
}

# Page Rules for optimization
resource "cloudflare_page_rule" "api_cache" {
  zone_id  = data.cloudflare_zone.main.id
  target   = "${var.api_domain}/v1/health"
  priority = 1
  
  actions {
    cache_level = "cache_everything"
    edge_cache_ttl = 300  # 5 minutes
  }
}

resource "cloudflare_page_rule" "static_assets" {
  zone_id  = data.cloudflare_zone.main.id
  target   = "${var.domain_name}/static/*"
  priority = 2
  
  actions {
    cache_level = "cache_everything"
    edge_cache_ttl = 86400  # 24 hours
  }
}

# Security settings
resource "cloudflare_zone_settings_override" "security" {
  zone_id = data.cloudflare_zone.main.id
  
  settings {
    ssl                      = "strict"
    always_use_https        = "on"
    min_tls_version         = "1.2"
    opportunistic_encryption = "on"
    tls_1_3                 = "zrt"
    automatic_https_rewrites = "on"
    security_level          = "medium"
    challenge_ttl           = 1800
    browser_check           = "on"
    hotlink_protection      = "on"
    ip_geolocation          = "on"
    email_obfuscation       = "on"
    server_side_exclude     = "on"
    brotli                  = "on"
    minify {
      css  = "on"
      js   = "on"
      html = "on"
    }
  }
}

# Rate limiting rules
resource "cloudflare_rate_limit" "api_protection" {
  zone_id   = data.cloudflare_zone.main.id
  threshold = 100
  period    = 60
  
  match {
    request {
      url_pattern = "${var.api_domain}/v1/*"
      schemes     = ["HTTPS"]
      methods     = ["GET", "POST", "PUT", "DELETE"]
    }
  }
  
  action {
    mode    = "challenge"
    timeout = 86400
  }
  
  correlate {
    by = "nat"
  }
  
  disabled    = false
  description = "Rate limiting for API endpoints"
}

# Outputs
output "r2_bucket_name" {
  description = "R2 bucket name"
  value       = cloudflare_r2_bucket.fallback_storage.name
}

output "worker_domains" {
  description = "Worker custom domains"
  value = {
    feed_poller     = cloudflare_worker_domain.feed_poller.hostname
    task_dispatcher = cloudflare_worker_domain.task_dispatcher.hostname
  }
}

output "kv_namespaces" {
  description = "KV namespace IDs"
  value = {
    feed_cache = cloudflare_workers_kv_namespace.feed_cache.id
    task_queue = cloudflare_workers_kv_namespace.task_queue.id
    metrics    = cloudflare_workers_kv_namespace.metrics.id
  }
}