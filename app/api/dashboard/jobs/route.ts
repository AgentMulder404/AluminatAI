import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { requireRole } from "@/lib/rbac";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

interface MetricRow {
  job_id: string | null;
  team_id: string | null;
  model_tag: string | null;
  scheduler_source: string | null;
  cluster_tag: string | null;
  machine_id: string | null;
  gpu_uuid: string | null;
  gpu_name: string | null;
  energy_delta_j: number | null;
  gpu_fraction: number | null;
  carbon_g_per_kwh: number | null;
  grid_zone: string | null;
  time: string;
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

  const rl = await rateLimit(`dash-jobs:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const clusterTag = req.nextUrl.searchParams.get("cluster_tag") ?? "";
  const teamId = req.nextUrl.searchParams.get("team_id") ?? "";

  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("gpu_metrics")
    .select(
      "job_id, team_id, model_tag, scheduler_source, cluster_tag, machine_id, " +
        "gpu_uuid, gpu_name, energy_delta_j, gpu_fraction, carbon_g_per_kwh, grid_zone, time"
    )
    .eq("user_id", user.id)
    .not("job_id", "is", null)
    .order("time", { ascending: false })
    .limit(2000);

  if (teamId) {
    query = query.eq("team_id", teamId);
  }
  if (clusterTag) {
    query = query.eq("cluster_tag", clusterTag);
  }

  const { data, error } = await query;

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  // Group by job_id and aggregate per-job stats
  type JobAgg = {
    job_id: string;
    team_id: string | null;
    model_tag: string | null;
    scheduler_source: string | null;
    cluster_tag: string | null;
    machine_id: string | null;
    gpu_uuids: Set<string>;
    gpu_name: string | null;
    total_energy_j: number;
    total_co2e_g: number | null;
    carbon_samples: number;
    grid_zone: string | null;
    first_time: string;
    last_time: string;
  };

  const jobMap = new Map<string, JobAgg>();

  for (const row of (data ?? []) as MetricRow[]) {
    if (!row.job_id) continue;

    const frac = row.gpu_fraction ?? 1;
    const energyJ = (row.energy_delta_j ?? 0) * frac;

    let agg = jobMap.get(row.job_id);
    if (!agg) {
      agg = {
        job_id: row.job_id,
        team_id: row.team_id,
        model_tag: row.model_tag,
        scheduler_source: row.scheduler_source,
        cluster_tag: row.cluster_tag,
        machine_id: row.machine_id,
        gpu_uuids: new Set(),
        gpu_name: row.gpu_name,
        total_energy_j: 0,
        total_co2e_g: null,
        carbon_samples: 0,
        grid_zone: row.grid_zone,
        first_time: row.time,
        last_time: row.time,
      };
      jobMap.set(row.job_id, agg);
    }

    if (row.gpu_uuid) agg.gpu_uuids.add(row.gpu_uuid);
    agg.total_energy_j += energyJ;

    if (row.carbon_g_per_kwh != null) {
      const co2g = (energyJ / 3_600_000) * row.carbon_g_per_kwh;
      agg.total_co2e_g = (agg.total_co2e_g ?? 0) + co2g;
      agg.carbon_samples++;
    }

    if (row.time < agg.first_time) agg.first_time = row.time;
    if (row.time > agg.last_time) agg.last_time = row.time;
  }

  const kwhRate = await getUserKwhRate(user.id);

  const jobs = [...jobMap.values()].map((agg) => {
    const totalKwh = agg.total_energy_j / 3_600_000;
    return {
      job_id: agg.job_id,
      team_id: agg.team_id,
      model_tag: agg.model_tag,
      scheduler_source: agg.scheduler_source,
      cluster_tag: agg.cluster_tag,
      machine_id: agg.machine_id,
      gpu_count: agg.gpu_uuids.size,
      gpu_name: agg.gpu_name,
      total_energy_j: agg.total_energy_j,
      total_kwh: totalKwh,
      cost_usd: totalKwh * kwhRate,
      total_co2e_g: agg.total_co2e_g,
      grid_zone: agg.grid_zone,
      start_time: agg.first_time,
      end_time: agg.last_time,
    };
  });

  // Sort by most recent end_time first
  jobs.sort((a, b) => (a.end_time < b.end_time ? 1 : -1));

  return NextResponse.json(jobs);
}
