// GET /api/dashboard/jobs/[jobId]
// Returns raw gpu_metrics timeseries for a single job (power timeline).
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";

export const runtime = "edge";

export async function GET(
  req: NextRequest,
  { params }: { params: { jobId: string } }
) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { jobId } = params;

  if (!jobId) {
    return NextResponse.json({ error: "jobId is required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("gpu_metrics")
    .select(
      "time, gpu_uuid, gpu_name, gpu_index, power_draw_w, energy_delta_j, " +
        "utilization_gpu_pct, utilization_memory_pct, temperature_c, " +
        "memory_used_mb, memory_total_mb, gpu_fraction, " +
        "team_id, model_tag, scheduler_source, cluster_tag, " +
        "carbon_g_per_kwh, grid_zone"
    )
    .eq("user_id", user.id)
    .eq("job_id", jobId)
    .order("time", { ascending: true })
    .limit(5000);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  if (!data || data.length === 0) {
    return NextResponse.json({ error: "Job not found" }, { status: 404 });
  }

  const kwhRate = await getUserKwhRate(user.id);

  // Compute summary stats
  let totalJ = 0;
  let totalCo2eG = 0;
  let maxPowerW = 0;
  let peakUtil = 0;
  const gpuUuids = new Set<string>();

  for (const row of data) {
    const frac = (row.gpu_fraction ?? 1) as number;
    const energyJ = ((row.energy_delta_j ?? 0) as number) * frac;
    totalJ += energyJ;

    if ((row.power_draw_w as number) > maxPowerW) maxPowerW = row.power_draw_w as number;
    if ((row.utilization_gpu_pct as number) > peakUtil) peakUtil = row.utilization_gpu_pct as number;
    if (row.gpu_uuid) gpuUuids.add(row.gpu_uuid as string);

    if (row.carbon_g_per_kwh != null) {
      totalCo2eG += (energyJ / 3_600_000) * (row.carbon_g_per_kwh as number);
    }
  }

  const totalKwh = totalJ / 3_600_000;
  const firstRow = data[0];
  const lastRow = data[data.length - 1];

  return NextResponse.json({
    job_id: jobId,
    team_id: firstRow.team_id,
    model_tag: firstRow.model_tag,
    scheduler_source: firstRow.scheduler_source,
    cluster_tag: firstRow.cluster_tag,
    gpu_count: gpuUuids.size,
    gpu_name: firstRow.gpu_name,
    start_time: firstRow.time,
    end_time: lastRow.time,
    summary: {
      total_kwh: Math.round(totalKwh * 1000) / 1000,
      cost_usd: Math.round(totalKwh * kwhRate * 100) / 100,
      total_co2e_g: Math.round(totalCo2eG * 100) / 100,
      max_power_w: Math.round(maxPowerW),
      peak_utilization_pct: Math.round(peakUtil),
      sample_count: data.length,
    },
    timeseries: data,
  });
}
