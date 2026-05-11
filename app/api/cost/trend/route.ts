// GET /api/cost/trend
// Daily/weekly/monthly cost timeseries with end-of-month projection.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { requireRole } from "@/lib/rbac";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

type Granularity = "day" | "week" | "month";

function bucketKey(time: string, granularity: Granularity): string {
  const d = new Date(time);
  switch (granularity) {
    case "day":
      return d.toISOString().slice(0, 10); // YYYY-MM-DD
    case "week": {
      // ISO week start (Monday)
      const day = d.getUTCDay();
      const diff = d.getUTCDate() - day + (day === 0 ? -6 : 1);
      const monday = new Date(d);
      monday.setUTCDate(diff);
      return monday.toISOString().slice(0, 10);
    }
    case "month":
      return d.toISOString().slice(0, 7); // YYYY-MM
  }
}

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`cost-trend:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const params = req.nextUrl.searchParams;
  const granularity = (params.get("granularity") ?? "day") as Granularity;
  const clusterTag = params.get("cluster_tag") ?? "";
  const teamId = params.get("team_id") ?? "";

  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  if (!["day", "week", "month"].includes(granularity)) {
    return NextResponse.json(
      { error: "granularity must be 'day', 'week', or 'month'" },
      { status: 400 }
    );
  }

  // Default ranges based on granularity
  const now = new Date();
  const defaultDays = granularity === "day" ? 30 : granularity === "week" ? 90 : 365;
  const from = params.get("from")
    ? new Date(params.get("from")!)
    : new Date(now.getTime() - defaultDays * 86400000);
  const to = params.get("to") ? new Date(params.get("to")!) : now;

  const supabase = createSupabaseServerClient();

  const includeCloud = params.get("include_cloud") === "true";

  let query = supabase
    .from("gpu_metrics")
    .select(
      includeCloud
        ? "time, energy_delta_j, gpu_fraction, carbon_g_per_kwh, gpu_name"
        : "time, energy_delta_j, gpu_fraction, carbon_g_per_kwh"
    )
    .eq("user_id", user.id)
    .gte("time", from.toISOString())
    .lte("time", to.toISOString())
    .order("time", { ascending: true });

  if (teamId) {
    query = query.eq("team_id", teamId);
  }
  if (clusterTag) {
    query = query.eq("cluster_tag", clusterTag);
  }

  const { data, error } = await query.limit(100000);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  const kwhRate = await getUserKwhRate(user.id);

  // Load reference pricing if cloud comparison requested
  let refMap = new Map<string, number>(); // gpu_model → highest on-demand rate
  if (includeCloud) {
    const { data: refs } = await supabase
      .from("gpu_reference_pricing")
      .select("gpu_model, rate_usd_per_gpu_hour");
    for (const r of refs ?? []) {
      const model = r.gpu_model as string;
      const rate = r.rate_usd_per_gpu_hour as number;
      if (!refMap.has(model) || rate > refMap.get(model)!) {
        refMap.set(model, rate);
      }
    }
  }

  function findRefRate(gpuName: string | null): number | null {
    if (!gpuName) return null;
    const direct = refMap.get(gpuName);
    if (direct) return direct;
    for (const [model, rate] of refMap) {
      if (gpuName.includes(model) || model.includes(gpuName.replace("NVIDIA ", ""))) {
        return rate;
      }
    }
    return null;
  }

  // Bucket metrics by period
  const buckets = new Map<
    string,
    { totalJ: number; totalCo2eG: number | null; cloudSamples: number; cloudRate: number | null }
  >();

  for (const rawRow of data ?? []) {
    const row = rawRow as unknown as Record<string, unknown>;
    const key = bucketKey(row.time as string, granularity);
    const frac = (row.gpu_fraction as number) ?? 1;
    const energyJ = ((row.energy_delta_j as number) ?? 0) * frac;

    let bucket = buckets.get(key);
    if (!bucket) {
      bucket = { totalJ: 0, totalCo2eG: null, cloudSamples: 0, cloudRate: null };
      buckets.set(key, bucket);
    }
    bucket.totalJ += energyJ;

    if (row.carbon_g_per_kwh != null) {
      const co2g = (energyJ / 3_600_000) * (row.carbon_g_per_kwh as number);
      bucket.totalCo2eG = (bucket.totalCo2eG ?? 0) + co2g;
    }

    if (includeCloud) {
      const rate = findRefRate(row.gpu_name as string | null);
      if (rate) {
        bucket.cloudRate = rate;
        bucket.cloudSamples++;
      }
    }
  }

  const points = [...buckets.entries()].map(([period, b]) => {
    const kwh = b.totalJ / 3_600_000;
    const point: { period: string; cost_usd: number; kwh: number; co2e_g: number | null; cloud_equivalent_usd?: number } = {
      period,
      cost_usd: Math.round(kwh * kwhRate * 100) / 100,
      kwh: Math.round(kwh * 1000) / 1000,
      co2e_g: b.totalCo2eG != null ? Math.round(b.totalCo2eG * 100) / 100 : null,
    };

    if (includeCloud && b.cloudRate) {
      const gpuHours = (b.cloudSamples * 5) / 3600;
      point.cloud_equivalent_usd = Math.round(b.cloudRate * gpuHours * 100) / 100;
    }

    return point;
  });

  // Monthly projection: extrapolate current month's spend
  let projection = null;
  if (granularity === "day" && points.length > 0) {
    const today = now.toISOString().slice(0, 10);
    const monthStart = now.toISOString().slice(0, 7);
    const daysInMonth = new Date(now.getUTCFullYear(), now.getUTCMonth() + 1, 0).getUTCDate();
    const dayOfMonth = now.getUTCDate();

    const thisMonthSpend = points
      .filter((p) => p.period.startsWith(monthStart) && p.period <= today)
      .reduce((sum, p) => sum + p.cost_usd, 0);

    if (dayOfMonth > 1) {
      const dailyAvg = thisMonthSpend / dayOfMonth;
      const projected = dailyAvg * daysInMonth;
      const trendPct =
        points.length >= 2
          ? ((points[points.length - 1].cost_usd - points[points.length - 2].cost_usd) /
              Math.max(points[points.length - 2].cost_usd, 0.01)) *
            100
          : 0;

      projection = {
        end_of_month_usd: Math.round(projected * 100) / 100,
        trend_pct: Math.round(trendPct * 10) / 10,
      };
    }
  }

  return NextResponse.json({ points, projection });
}
