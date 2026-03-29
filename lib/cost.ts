// Shared cost helper — centralizes electricity rate lookups.
// Replaces the hardcoded $0.12/kWh in dashboard API routes.

import { createSupabaseServerClient } from "@/lib/supabase-client";

const DEFAULT_KWH_RATE = 0.12; // USD per kWh — fallback when no user rate configured

/**
 * Returns the user's configured electricity rate in USD/kWh.
 * Falls back to DEFAULT_KWH_RATE if no custom rate is set.
 */
export async function getUserKwhRate(userId: string): Promise<number> {
  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("cost_rates")
    .select("rate_usd")
    .eq("user_id", userId)
    .eq("rate_type", "electricity")
    .eq("is_default", true)
    .single();

  if (error || !data) {
    return DEFAULT_KWH_RATE;
  }

  return Number(data.rate_usd);
}

/**
 * Returns the period start date for a given budget period.
 */
export function getPeriodStart(period: "daily" | "weekly" | "monthly"): Date {
  const now = new Date();
  switch (period) {
    case "daily": {
      const d = new Date(now);
      d.setUTCHours(0, 0, 0, 0);
      return d;
    }
    case "weekly": {
      const d = new Date(now);
      d.setUTCDate(d.getUTCDate() - d.getUTCDay());
      d.setUTCHours(0, 0, 0, 0);
      return d;
    }
    case "monthly": {
      const d = new Date(now);
      d.setUTCDate(1);
      d.setUTCHours(0, 0, 0, 0);
      return d;
    }
  }
}
