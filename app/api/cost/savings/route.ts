// GET /api/cost/savings
// Computes "You saved $X vs cloud" by comparing actual electricity cost
// against cloud GPU on-demand rates from gpu_reference_pricing.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { requireRole } from "@/lib/rbac";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
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

  const rl = await rateLimit(`cost-savings:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const params = req.nextUrl.searchParams;
  const days = Math.min(Number(params.get("days") ?? 30), 365);
  const teamId = params.get("team_id") ?? "";

  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  const supabase = createSupabaseServerClient();
  const cutoff = new Date(Date.now() - days * 86_400_000).toISOString();

  // Get user's metrics grouped by GPU
  let query = supabase
    .from("gpu_metrics")
    .select("gpu_name, energy_delta_j, gpu_fraction, time")
    .eq("user_id", user.id)
    .gte("time", cutoff)
    .not("gpu_name", "is", null);

  if (teamId) query = query.eq("team_id", teamId);

  const { data: metrics, error } = await query.limit(100000);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  // Aggregate by GPU model: total kWh and estimated GPU-hours
  const gpuAgg = new Map<
    string,
    { totalJ: number; count: number; firstTime: string; lastTime: string }
  >();

  for (const row of metrics ?? []) {
    const name = row.gpu_name as string;
    if (!name) continue;
    const frac = (row.gpu_fraction ?? 1) as number;
    const energyJ = ((row.energy_delta_j ?? 0) as number) * frac;

    let agg = gpuAgg.get(name);
    if (!agg) {
      agg = { totalJ: 0, count: 0, firstTime: row.time, lastTime: row.time };
      gpuAgg.set(name, agg);
    }
    agg.totalJ += energyJ;
    agg.count++;
    if (row.time > agg.lastTime) agg.lastTime = row.time;
    if (row.time < agg.firstTime) agg.firstTime = row.time;
  }

  // Get reference pricing (highest on-demand rate per GPU model for "worst case" cloud cost)
  const { data: refPricing } = await supabase
    .from("gpu_reference_pricing")
    .select("gpu_model, provider, rate_usd_per_gpu_hour");

  // Build a map: gpu_model → { max_rate, provider }
  const refMap = new Map<string, { rate: number; provider: string }>();
  for (const ref of refPricing ?? []) {
    const model = ref.gpu_model as string;
    const rate = ref.rate_usd_per_gpu_hour as number;
    const existing = refMap.get(model);
    if (!existing || rate > existing.rate) {
      refMap.set(model, { rate, provider: ref.provider as string });
    }
  }

  const kwhRate = await getUserKwhRate(user.id);

  // Calculate savings per GPU
  const byGpu: Array<{
    gpu_name: string;
    hours: number;
    actual_usd: number;
    cloud_equivalent_usd: number;
    savings_usd: number;
    savings_pct: number;
    reference_provider: string;
    reference_rate_hr: number;
  }> = [];

  let totalActual = 0;
  let totalCloudEquivalent = 0;

  for (const [gpuName, agg] of gpuAgg) {
    const totalKwh = agg.totalJ / 3_600_000;
    const actualCost = totalKwh * kwhRate;

    // Estimate GPU-hours from sample count (5-second intervals typical)
    const gpuHours = (agg.count * 5) / 3600;

    // Find reference pricing — try exact match first, then fuzzy match
    let ref = refMap.get(gpuName);
    if (!ref) {
      // Try matching by partial name (e.g., "NVIDIA A100-SXM4-80GB" → "A100-SXM4-80GB")
      for (const [model, r] of refMap) {
        if (gpuName.includes(model) || model.includes(gpuName.replace("NVIDIA ", ""))) {
          ref = r;
          break;
        }
      }
    }

    if (!ref) continue; // No reference price for this GPU — skip

    const cloudEquivalent = ref.rate * gpuHours;
    const savings = cloudEquivalent - actualCost;

    totalActual += actualCost;
    totalCloudEquivalent += cloudEquivalent;

    byGpu.push({
      gpu_name: gpuName,
      hours: Math.round(gpuHours * 10) / 10,
      actual_usd: Math.round(actualCost * 100) / 100,
      cloud_equivalent_usd: Math.round(cloudEquivalent * 100) / 100,
      savings_usd: Math.round(savings * 100) / 100,
      savings_pct:
        cloudEquivalent > 0
          ? Math.round((savings / cloudEquivalent) * 10000) / 100
          : 0,
      reference_provider: ref.provider,
      reference_rate_hr: ref.rate,
    });
  }

  // Sort by savings descending
  byGpu.sort((a, b) => b.savings_usd - a.savings_usd);

  const totalSavings = totalCloudEquivalent - totalActual;

  return NextResponse.json({
    period_days: days,
    total_actual_usd: Math.round(totalActual * 100) / 100,
    total_cloud_equivalent_usd: Math.round(totalCloudEquivalent * 100) / 100,
    total_savings_usd: Math.round(totalSavings * 100) / 100,
    savings_pct:
      totalCloudEquivalent > 0
        ? Math.round((totalSavings / totalCloudEquivalent) * 10000) / 100
        : 0,
    by_gpu: byGpu,
  });
}
