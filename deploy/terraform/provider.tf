# AluminatAI Terraform Provider Configuration
#
# This provider manages AluminatAI resources via the REST API.
# Requires an API key with admin privileges.
#
# Example usage:
#   terraform {
#     required_providers {
#       aluminatai = {
#         source  = "aluminatai/aluminatai"
#         version = "~> 0.1"
#       }
#     }
#   }
#
#   provider "aluminatai" {
#     api_key  = var.aluminatai_api_key
#     endpoint = "https://www.aluminatai.com"
#   }

terraform {
  required_version = ">= 1.0"
}

variable "aluminatai_api_key" {
  type        = string
  sensitive   = true
  description = "AluminatAI API key (alum_...)"
}

variable "aluminatai_endpoint" {
  type        = string
  default     = "https://www.aluminatai.com"
  description = "AluminatAI API endpoint"
}

# Local values for API calls
locals {
  api_headers = {
    "Content-Type" = "application/json"
    "X-API-Key"    = var.aluminatai_api_key
  }
  base_url = var.aluminatai_endpoint
}
