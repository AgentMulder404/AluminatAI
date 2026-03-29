# Example AluminatAI Terraform configuration
# Copy this to terraform.tfvars and fill in your values.
#
# Usage:
#   terraform init
#   terraform plan -var-file=terraform.tfvars
#   terraform apply -var-file=terraform.tfvars

aluminatai_api_key  = "alum_your_api_key_here"
aluminatai_endpoint = "https://www.aluminatai.com"

cost_rates = [
  {
    rate_type    = "electricity"
    rate_per_kwh = 0.08
    currency     = "USD"
    label        = "US-West-2 data center rate"
    is_default   = true
  }
]

budgets = [
  {
    name        = "Production GPU Budget"
    scope_type  = "cluster"
    scope_value = "prod"
    period      = "monthly"
    limit_usd   = 5000
    warn_pct    = 80
    notify_channels = [
      { type = "slack", target = "https://hooks.slack.com/services/T.../B.../xxx" },
      { type = "pagerduty", target = "your-pagerduty-routing-key" }
    ]
  },
  {
    name        = "Dev/Test Daily Limit"
    scope_type  = "cluster"
    scope_value = "dev"
    period      = "daily"
    limit_usd   = 100
    warn_pct    = 90
    notify_channels = [
      { type = "email", target = "team@example.com" }
    ]
  }
]

webhooks = [
  {
    url         = "https://your-service.com/webhooks/aluminatai"
    description = "Production event sink"
    event_types = ["budget.exceeded", "waste.detected", "agent.offline"]
  }
]

teams = [
  {
    name = "ML Platform"
    members = [
      { email = "alice@example.com", role = "admin" },
      { email = "bob@example.com", role = "viewer" },
      { email = "carol@example.com", role = "billing" }
    ]
  }
]
