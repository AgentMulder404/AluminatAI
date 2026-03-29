# AluminatAI Terraform Resource Definitions
#
# These resources use the restapi provider (or curl-based provisioners)
# to manage AluminatAI configuration as infrastructure-as-code.
#
# Resources:
#   - aluminatai_cost_rate     → Electricity rate configuration
#   - aluminatai_budget        → Budget threshold with alerts
#   - aluminatai_webhook       → Event webhook endpoint
#   - aluminatai_team          → Team (org) management
#   - aluminatai_export_config → S3/GCS export configuration

# ── Cost Rate ────────────────────────────────────────────────────────────────

variable "cost_rates" {
  type = list(object({
    rate_type      = string # "electricity", "cloud_gpu"
    rate_per_kwh   = number
    currency       = optional(string, "USD")
    label          = optional(string, "")
    is_default     = optional(bool, true)
  }))
  default     = []
  description = "Electricity rates for cost calculation"
}

resource "null_resource" "cost_rates" {
  for_each = { for i, r in var.cost_rates : i => r }

  triggers = {
    rate_type    = each.value.rate_type
    rate_per_kwh = each.value.rate_per_kwh
    is_default   = each.value.is_default
  }

  provisioner "local-exec" {
    command = <<-EOT
      curl -s -X POST "${local.base_url}/api/cost/rates" \
        -H "Content-Type: application/json" \
        -H "Cookie: ${var.aluminatai_api_key}" \
        -d '${jsonencode({
          rate_type    = each.value.rate_type
          rate_per_kwh = each.value.rate_per_kwh
          currency     = each.value.currency
          label        = each.value.label
          is_default   = each.value.is_default
        })}'
    EOT
  }
}

# ── Budget ───────────────────────────────────────────────────────────────────

variable "budgets" {
  type = list(object({
    name            = string
    scope_type      = string # "global", "team", "cluster", "gpu_model"
    scope_value     = optional(string, "")
    period          = string # "daily", "weekly", "monthly"
    limit_usd       = number
    warn_pct        = optional(number, 80)
    notify_channels = optional(list(object({
      type   = string
      target = string
    })), [])
  }))
  default     = []
  description = "Budget thresholds with notification channels"
}

resource "null_resource" "budgets" {
  for_each = { for i, b in var.budgets : b.name => b }

  triggers = {
    name      = each.value.name
    limit_usd = each.value.limit_usd
    period    = each.value.period
  }

  provisioner "local-exec" {
    command = <<-EOT
      curl -s -X POST "${local.base_url}/api/budgets" \
        -H "Content-Type: application/json" \
        -H "Cookie: ${var.aluminatai_api_key}" \
        -d '${jsonencode({
          name            = each.value.name
          scope_type      = each.value.scope_type
          scope_value     = each.value.scope_value
          period          = each.value.period
          limit_usd       = each.value.limit_usd
          warn_pct        = each.value.warn_pct
          notify_channels = each.value.notify_channels
        })}'
    EOT
  }
}

# ── Webhook ──────────────────────────────────────────────────────────────────

variable "webhooks" {
  type = list(object({
    url         = string
    description = optional(string, "")
    event_types = optional(list(string), [])
  }))
  default     = []
  description = "Event webhook endpoints"
}

resource "null_resource" "webhooks" {
  for_each = { for i, w in var.webhooks : w.url => w }

  triggers = {
    url         = each.value.url
    event_types = join(",", each.value.event_types)
  }

  provisioner "local-exec" {
    command = <<-EOT
      curl -s -X POST "${local.base_url}/api/webhooks" \
        -H "Content-Type: application/json" \
        -H "Cookie: ${var.aluminatai_api_key}" \
        -d '${jsonencode({
          url         = each.value.url
          description = each.value.description
          event_types = each.value.event_types
        })}'
    EOT
  }
}

# ── Team ─────────────────────────────────────────────────────────────────────

variable "teams" {
  type = list(object({
    name    = string
    members = optional(list(object({
      email = string
      role  = optional(string, "viewer")
    })), [])
  }))
  default     = []
  description = "Teams and member invitations"
}

resource "null_resource" "teams" {
  for_each = { for t in var.teams : t.name => t }

  triggers = {
    name = each.value.name
  }

  provisioner "local-exec" {
    command = <<-EOT
      TEAM_RESPONSE=$(curl -s -X POST "${local.base_url}/api/teams" \
        -H "Content-Type: application/json" \
        -H "Cookie: ${var.aluminatai_api_key}" \
        -d '${jsonencode({ name = each.value.name })}')

      TEAM_ID=$(echo "$TEAM_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

      %{for member in each.value.members}
      if [ -n "$TEAM_ID" ]; then
        curl -s -X POST "${local.base_url}/api/teams/$TEAM_ID/members" \
          -H "Content-Type: application/json" \
          -H "Cookie: ${var.aluminatai_api_key}" \
          -d '${jsonencode({ email = member.email, role = member.role })}'
      fi
      %{endfor}
    EOT
  }
}
