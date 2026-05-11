// GET /api/cost/recommendations
// Returns scheduling, GPU sizing recommendations, and savings calculator.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { requireRole } from "@/lib/rbac";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`cost-recs:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const teamId = req.nextUrl.searchParams.get("team_id") ?? "";
  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  const supabase = createSupabaseServerClient();

  // 1. Fetch stored recommendations
  const { data: recs } = await supabase
    .from("scheduling_recommendations")
    .select("id, recommendation_type, title, description, estimated_savings_usd, estimated_savings_pct, context, created_at")
    .eq("user_id", user.id)
    .eq("dismissed", false)
    .order("created_at", { ascending: false })
    .limit(20);

  // 2. Build savings calculator: compare user's GPU models against reference pricing
  const kwhRate = await getUserKwhRate(user.id);

  // Get user's distinct GPU models and their avg power from last 30 days
  const thirtyDaysAgo = new Date(Date.now() - 30 * 86400000).toISOString();

  const { data: userGpus } = await supabase
    .from("gpu_metrics")
    .select("gpu_name, power_draw_w, energy_delta_j")
    .eq("user_id", user.id)
    .gte("time", thirtyDaysAgo)
    .not("gpu_name", "is", null)
    .limit(50000);

  // Aggregate by GPU model
  const gpuStats = new Map<
    string,
    { totalJ: number; count: number; totalPowerW: number }
  >();

  for (const row of userGpus ?? []) {
    const name = row.gpu_name as string;
    if (!name) continue;
    let stat = gpuStats.get(name);
    if (!stat) {
      stat = { totalJ: 0, count: 0, totalPowerW: 0 };
      gpuStats.set(name, stat);
    }
    stat.totalJ += (row.energy_delta_j ?? 0) as number;
    stat.totalPowerW += (row.power_draw_w ?? 0) as number;
    stat.count++;
  }

  // Get reference pricing
  const { data: refPricing } = await supabase
    .from("gpu_reference_pricing")
    .select("gpu_model, provider, instance_type, rate_usd_per_gpu_hour, spot_rate_usd_per_gpu_hour, tdp_watts, memory_gb");

  // Build savings suggestions
  const savingsCalculator: Array<{
    current_gpu: string;
    current_monthly_cost_usd: number;
    alternatives: Array<{
      gpu_model: string;
      provider: string;
      instance_type: string | null;
      monthly_cost_usd: number;
      monthly_savings_usd: number;
      savings_pct: number;
      is_spot: boolean;
    }>;
  }> = [];

  for (const [gpuName, stat] of gpuStats) {
    const avgPowerW = stat.totalPowerW / stat.count;
    const totalKwh = stat.totalJ / 3_600_000;
    const monthlyCost = totalKwh * kwhRate;
    const monthlyHours = (stat.count * 5) / 3600; // approx GPU-hours (5s sample interval)

    if (monthlyCost < 1) continue; // skip negligible costs

    // Find cheaper alternatives from reference pricing
    const alternatives: typeof savingsCalculator[0]["alternatives"] = [];

    for (const ref of refPricing ?? []) {
      // On-demand comparison
      const refMonthlyCost = (ref.rate_usd_per_gpu_hour as number) * monthlyHours;
      if (refMonthlyCost < monthlyCost * 0.8) {
        alternatives.push({
          gpu_model: ref.gpu_model as string,
          provider: ref.provider as string,
          instance_type: ref.instance_type as string | null,
          monthly_cost_usd: Math.round(refMonthlyCost * 100) / 100,
          monthly_savings_usd: Math.round((monthlyCost - refMonthlyCost) * 100) / 100,
          savings_pct: Math.round(((monthlyCost - refMonthlyCost) / monthlyCost) * 100),
          is_spot: false,
        });
      }

      // Spot pricing comparison
      if (ref.spot_rate_usd_per_gpu_hour) {
        const spotMonthlyCost = (ref.spot_rate_usd_per_gpu_hour as number) * monthlyHours;
        if (spotMonthlyCost < monthlyCost * 0.6) {
          alternatives.push({
            gpu_model: ref.gpu_model as string,
            provider: ref.provider as string,
            instance_type: ref.instance_type as string | null,
            monthly_cost_usd: Math.round(spotMonthlyCost * 100) / 100,
            monthly_savings_usd: Math.round((monthlyCost - spotMonthlyCost) * 100) / 100,
            savings_pct: Math.round(((monthlyCost - spotMonthlyCost) / monthlyCost) * 100),
            is_spot: true,
          });
        }
      }
    }

    // Sort by savings descending, take top 5
    alternatives.sort((a, b) => b.monthly_savings_usd - a.monthly_savings_usd);

    if (alternatives.length > 0) {
      savingsCalculator.push({
        current_gpu: gpuName,
        current_monthly_cost_usd: Math.round(monthlyCost * 100) / 100,
        alternatives: alternatives.slice(0, 5),
      });
    }
  }

  return NextResponse.json({
    recommendations: recs ?? [],
    savings_calculator: savingsCalculator,
  });
}
