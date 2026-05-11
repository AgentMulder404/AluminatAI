// GET /api/cost/breakdown
// Cost breakdown by team, model, gpu, or cluster within a date range.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { requireRole } from "@/lib/rbac";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

const VALID_DIMENSIONS = ["team", "model", "gpu", "cluster"] as const;
type Dimension = (typeof VALID_DIMENSIONS)[number];

const DIMENSION_COLUMN: Record<Dimension, string> = {
  team: "team_id",
  model: "model_tag",
  gpu: "gpu_name",
  cluster: "cluster_tag",
};

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`cost-breakdown:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const params = req.nextUrl.searchParams;
  const dimension = params.get("dimension") as Dimension | null;
  const from = params.get("from");
  const to = params.get("to");
  const clusterTag = params.get("cluster_tag") ?? "";
  const teamId = params.get("team_id") ?? "";

  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  if (!dimension || !VALID_DIMENSIONS.includes(dimension)) {
    return NextResponse.json(
      { error: `dimension must be one of: ${VALID_DIMENSIONS.join(", ")}` },
      { status: 400 }
    );
  }

  // Default to last 30 days if no range specified
  const now = new Date();
  const fromDate = from ? new Date(from) : new Date(now.getTime() - 30 * 86400000);
  const toDate = to ? new Date(to) : now;

  const col = DIMENSION_COLUMN[dimension];
  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("gpu_metrics")
    .select(`${col}, energy_delta_j, gpu_fraction, job_id`)
    .eq("user_id", user.id)
    .gte("time", fromDate.toISOString())
    .lte("time", toDate.toISOString());

  if (teamId) {
    query = query.eq("team_id", teamId);
  }
  if (clusterTag) {
    query = query.eq("cluster_tag", clusterTag);
  }

  const { data, error } = await query.limit(50000);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  const kwhRate = await getUserKwhRate(user.id);

  // Aggregate by dimension value
  const groups = new Map<
    string,
    { totalJ: number; jobIds: Set<string> }
  >();

  let grandTotalJ = 0;

  for (const rawRow of data ?? []) {
    const row = rawRow as unknown as Record<string, unknown>;
    const key = row[col] as string | null;
    const label = key || "unattributed";
    const frac = (row.gpu_fraction as number) ?? 1;
    const energyJ = ((row.energy_delta_j as number) ?? 0) * frac;

    grandTotalJ += energyJ;

    let group = groups.get(label);
    if (!group) {
      group = { totalJ: 0, jobIds: new Set() };
      groups.set(label, group);
    }
    group.totalJ += energyJ;
    if (row.job_id) group.jobIds.add(row.job_id as string);
  }

  const grandTotalKwh = grandTotalJ / 3_600_000;
  const grandTotalCost = grandTotalKwh * kwhRate;

  const breakdown = [...groups.entries()]
    .map(([dimensionValue, g]) => {
      const kwh = g.totalJ / 3_600_000;
      const cost = kwh * kwhRate;
      return {
        dimension_value: dimensionValue,
        total_kwh: Math.round(kwh * 1000) / 1000,
        cost_usd: Math.round(cost * 100) / 100,
        pct_of_total: grandTotalCost > 0 ? Math.round((cost / grandTotalCost) * 10000) / 100 : 0,
        job_count: g.jobIds.size,
      };
    })
    .sort((a, b) => b.cost_usd - a.cost_usd);

  return NextResponse.json({
    breakdown,
    total_cost_usd: Math.round(grandTotalCost * 100) / 100,
    period: { from: fromDate.toISOString(), to: toDate.toISOString() },
  });
}
