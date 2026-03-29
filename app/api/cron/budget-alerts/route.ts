// GET /api/cron/budget-alerts
// Schedule: 0 * * * * (every hour)
// Checks spend against budget thresholds and fires notifications.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate, getPeriodStart } from "@/lib/cost";
import { dispatchBudgetAlert } from "@/lib/notifications";
import { dispatchWebhook } from "@/lib/webhooks";

export const runtime = "edge";

interface Budget {
  id: string;
  user_id: string;
  name: string;
  scope_type: string;
  scope_value: string | null;
  period: "daily" | "weekly" | "monthly";
  limit_usd: number;
  warn_pct: number;
  notify_channels: Array<{ type: string; target: string }>;
}

export async function GET(req: NextRequest) {
  const auth = req.headers.get("authorization") ?? "";
  const secret = process.env.CRON_SECRET;
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Fetch all active budgets
  const { data: budgets, error: budgetsErr } = await supabase
    .from("budgets")
    .select("id, user_id, name, scope_type, scope_value, period, limit_usd, warn_pct, notify_channels")
    .eq("is_active", true);

  if (budgetsErr || !budgets) {
    return NextResponse.json({ error: budgetsErr?.message ?? "No budgets" }, { status: 500 });
  }

  let alertsSent = 0;
  let budgetsChecked = 0;

  for (const budget of budgets as Budget[]) {
    budgetsChecked++;

    const periodStart = getPeriodStart(budget.period);
    const kwhRate = await getUserKwhRate(budget.user_id);

    // Build query for this budget's scope
    let query = supabase
      .from("gpu_metrics")
      .select("energy_delta_j, gpu_fraction")
      .eq("user_id", budget.user_id)
      .gte("time", periodStart.toISOString());

    // Apply scope filter
    if (budget.scope_type === "team" && budget.scope_value) {
      query = query.eq("team_id", budget.scope_value);
    } else if (budget.scope_type === "cluster" && budget.scope_value) {
      query = query.eq("cluster_tag", budget.scope_value);
    } else if (budget.scope_type === "gpu_model" && budget.scope_value) {
      query = query.eq("gpu_name", budget.scope_value);
    }
    // "global" and "project" don't filter further

    const { data: metrics } = await query.limit(100000);

    if (!metrics || metrics.length === 0) continue;

    // Calculate total spend
    let totalJ = 0;
    for (const m of metrics) {
      const frac = (m.gpu_fraction ?? 1) as number;
      totalJ += ((m.energy_delta_j ?? 0) as number) * frac;
    }
    const spendUsd = (totalJ / 3_600_000) * kwhRate;

    // Check warn threshold
    const warnThreshold = (budget.warn_pct / 100) * budget.limit_usd;

    const scopeLabel =
      budget.scope_type === "global"
        ? "All resources"
        : `${budget.scope_type}: ${budget.scope_value}`;

    // Check if exceeded
    if (spendUsd >= budget.limit_usd) {
      // Try to insert alert (dedup by budget + type + period)
      const { error: insertErr } = await supabase
        .from("budget_alerts")
        .insert({
          budget_id: budget.id,
          user_id: budget.user_id,
          alert_type: "exceeded",
          spend_usd: Math.round(spendUsd * 100) / 100,
          limit_usd: budget.limit_usd,
          period_start: periodStart.toISOString(),
          channels: budget.notify_channels,
        });

      // If insert succeeds (no dedup conflict), send notifications + webhooks
      if (!insertErr) {
        const alertPayload = {
          budget_name: budget.name,
          alert_type: "exceeded" as const,
          spend_usd: Math.round(spendUsd * 100) / 100,
          limit_usd: budget.limit_usd,
          period: budget.period,
          scope: scopeLabel,
        };
        await dispatchBudgetAlert(budget.notify_channels, alertPayload);
        await dispatchWebhook("budget.exceeded", budget.user_id, alertPayload);
        alertsSent++;
      }
    }
    // Check warn threshold
    else if (spendUsd >= warnThreshold) {
      const { error: insertErr } = await supabase
        .from("budget_alerts")
        .insert({
          budget_id: budget.id,
          user_id: budget.user_id,
          alert_type: "warn",
          spend_usd: Math.round(spendUsd * 100) / 100,
          limit_usd: budget.limit_usd,
          period_start: periodStart.toISOString(),
          channels: budget.notify_channels,
        });

      if (!insertErr) {
        const alertPayload = {
          budget_name: budget.name,
          alert_type: "warn" as const,
          spend_usd: Math.round(spendUsd * 100) / 100,
          limit_usd: budget.limit_usd,
          period: budget.period,
          scope: scopeLabel,
        };
        await dispatchBudgetAlert(budget.notify_channels, alertPayload);
        await dispatchWebhook("budget.warning", budget.user_id, alertPayload);
        alertsSent++;
      }
    }
  }

  return NextResponse.json({
    budgets_checked: budgetsChecked,
    alerts_sent: alertsSent,
  });
}
