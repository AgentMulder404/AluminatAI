// Plan definitions, feature limits, and gating helpers.
// Central source of truth for what each tier includes.

import { createSupabaseServerClient } from "@/lib/supabase-client";

export type PlanTier = "free" | "team" | "enterprise";

export interface PlanLimits {
  max_gpus: number;
  max_teams: number;
  max_team_members: number;
  max_budgets: number;
  max_webhooks: number;
  max_export_configs: number;
  retention_days: number;
  slack_integration: boolean;
  pagerduty_opsgenie: boolean;
  sla_dashboard: boolean;
  api_rate_limit: number;
  priority_support: boolean;
}

export const PLAN_LIMITS: Record<PlanTier, PlanLimits> = {
  free: {
    max_gpus: 2,
    max_teams: 1,
    max_team_members: 3,
    max_budgets: 1,
    max_webhooks: 0,
    max_export_configs: 0,
    retention_days: 7,
    slack_integration: false,
    pagerduty_opsgenie: false,
    sla_dashboard: false,
    api_rate_limit: 60,
    priority_support: false,
  },
  team: {
    max_gpus: -1,
    max_teams: 5,
    max_team_members: 25,
    max_budgets: 20,
    max_webhooks: 10,
    max_export_configs: 3,
    retention_days: 90,
    slack_integration: true,
    pagerduty_opsgenie: true,
    sla_dashboard: false,
    api_rate_limit: 300,
    priority_support: false,
  },
  enterprise: {
    max_gpus: -1,
    max_teams: -1,
    max_team_members: -1,
    max_budgets: -1,
    max_webhooks: -1,
    max_export_configs: -1,
    retention_days: 365,
    slack_integration: true,
    pagerduty_opsgenie: true,
    sla_dashboard: true,
    api_rate_limit: 1000,
    priority_support: true,
  },
};

// Stripe Price IDs — set these in environment variables
export const STRIPE_PRICES = {
  team_monthly: process.env.STRIPE_PRICE_TEAM_MONTHLY ?? "",
  team_yearly: process.env.STRIPE_PRICE_TEAM_YEARLY ?? "",
  enterprise_monthly: process.env.STRIPE_PRICE_ENTERPRISE_MONTHLY ?? "",
  enterprise_yearly: process.env.STRIPE_PRICE_ENTERPRISE_YEARLY ?? "",
};

export const PLAN_DISPLAY: Record<
  PlanTier,
  { name: string; price_monthly: number; price_yearly: number; tagline: string; per_gpu: boolean }
> = {
  free: {
    name: "Free",
    price_monthly: 0,
    price_yearly: 0,
    tagline: "For individual devs and academics",
    per_gpu: false,
  },
  team: {
    name: "Team",
    price_monthly: 5,
    price_yearly: 48, // $4/GPU/mo billed yearly
    tagline: "For teams running 10-100 GPUs",
    per_gpu: true,
  },
  enterprise: {
    name: "Enterprise",
    price_monthly: 15,
    price_yearly: 144, // $12/GPU/mo billed yearly
    tagline: "For organizations running 100+ GPU fleets",
    per_gpu: true,
  },
};

/**
 * Get a user's current plan and limits.
 */
export async function getUserPlan(
  userId: string
): Promise<{ plan: PlanTier; limits: PlanLimits; periodEnd: string | null; cancelAtPeriodEnd: boolean }> {
  const supabase = createSupabaseServerClient();

  const { data } = await supabase
    .from("users")
    .select("plan, plan_period_end, plan_cancel_at_period_end, trial_ends_at")
    .eq("id", userId)
    .single();

  if (!data) {
    return { plan: "free", limits: PLAN_LIMITS.free, periodEnd: null, cancelAtPeriodEnd: false };
  }

  let plan = (data.plan as PlanTier) ?? "free";

  // Backward compat: treat legacy "pro" as "team"
  if ((plan as string) === "pro") plan = "team";

  // If paid plan has expired and not renewed, downgrade to free
  if (plan !== "free" && data.plan_period_end) {
    if (new Date(data.plan_period_end) < new Date()) {
      plan = "free";
    }
  }

  return {
    plan,
    limits: PLAN_LIMITS[plan],
    periodEnd: data.plan_period_end,
    cancelAtPeriodEnd: data.plan_cancel_at_period_end ?? false,
  };
}

/**
 * Check if a user's plan allows a specific feature.
 * Returns { allowed: true } or { allowed: false, reason, upgrade_to }.
 */
export async function requirePlan(
  userId: string,
  feature: keyof PlanLimits
): Promise<{ allowed: boolean; reason?: string; upgrade_to?: PlanTier }> {
  const { plan, limits } = await getUserPlan(userId);

  const value = limits[feature];

  // Boolean features
  if (typeof value === "boolean") {
    if (!value) {
      const upgradeTo = plan === "free" ? "team" : "enterprise";
      return {
        allowed: false,
        reason: `${String(feature)} requires a ${upgradeTo} plan`,
        upgrade_to: upgradeTo as PlanTier,
      };
    }
    return { allowed: true };
  }

  // Numeric limits (-1 = unlimited)
  if (typeof value === "number" && value === -1) {
    return { allowed: true };
  }

  // For numeric limits, the caller needs to check count themselves
  return { allowed: true };
}

/**
 * Check if a user has reached the count limit for a feature.
 * Pass the current count; returns allowed if under the limit.
 */
export async function checkCountLimit(
  userId: string,
  feature: keyof PlanLimits,
  currentCount: number
): Promise<{ allowed: boolean; limit: number; reason?: string }> {
  const { plan, limits } = await getUserPlan(userId);
  const limit = limits[feature] as number;

  if (limit === -1) {
    return { allowed: true, limit: -1 };
  }

  if (currentCount >= limit) {
    const upgradeTo = plan === "free" ? "team" : "enterprise";
    return {
      allowed: false,
      limit,
      reason: `${String(feature)} limit reached (${currentCount}/${limit}). Upgrade to ${upgradeTo} for more.`,
    };
  }

  return { allowed: true, limit };
}
