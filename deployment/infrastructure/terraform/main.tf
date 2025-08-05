# Project Synapse - Main Terraform Configuration
# This is the master infrastructure definition for the entire system

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
    render = {
      source  = "render-oss/render"
      version = "~> 1.0"
    }
    vercel = {
      source  = "vercel/vercel"
      version = "~> 0.15"
    }
    github = {
      source  = "integrations/github"
      version = "~> 5.0"
    }
  }

  backend "remote" {
    organization = "project-synapse"
    workspaces {
      name = "synapse-production"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "project-synapse"
}

variable "docker_registry" {
  description = "Docker registry URL"
  type        = string
  default     = "ghcr.io/project-synapse"
}

variable "domain_name" {
  description = "Primary domain name"
  type        = string
  default     = "projectsynapse.com"
}

variable "api_domain" {
  description = "API domain name"
  type        = string
  default     = "api.projectsynapse.com"
}

# Data sources
data "cloudflare_zone" "main" {
  name = var.domain_name
}

# Local values
locals {
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  
  image_tags = {
    central_cortex = "${var.docker_registry}/synapse-cortex:latest"
    neurons        = "${var.docker_registry}/synapse-neuron:latest"
    frontend       = "${var.docker_registry}/synapse-frontend:latest"
  }
}

# Outputs
output "api_url" {
  description = "API URL"
  value       = "https://${var.api_domain}"
}

output "dashboard_url" {
  description = "Dashboard URL"
  value       = "https://${var.domain_name}"
}

output "cloudflare_zone_id" {
  description = "Cloudflare Zone ID"
  value       = data.cloudflare_zone.main.id
}

output "deployment_info" {
  description = "Deployment information"
  value = {
    environment    = var.environment
    project_name   = var.project_name
    api_domain     = var.api_domain
    dashboard_domain = var.domain_name
    docker_images  = local.image_tags
  }
}