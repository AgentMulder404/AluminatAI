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
  { params }: { params: Promise<{ jobId: string }> }
) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { jobId } = await params;

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

  const rows = data as unknown as Record<string, unknown>[];

  for (const row of rows) {
    const frac = (row.gpu_fraction as number) ?? 1;
    const energyJ = ((row.energy_delta_j as number) ?? 0) * frac;
    totalJ += energyJ;

    if ((row.power_draw_w as number) > maxPowerW) maxPowerW = row.power_draw_w as number;
    if ((row.utilization_gpu_pct as number) > peakUtil) peakUtil = row.utilization_gpu_pct as number;
    if (row.gpu_uuid) gpuUuids.add(row.gpu_uuid as string);

    if (row.carbon_g_per_kwh != null) {
      totalCo2eG += (energyJ / 3_600_000) * (row.carbon_g_per_kwh as number);
    }
  }

  const totalKwh = totalJ / 3_600_000;
  const firstRow = rows[0];
  const lastRow = rows[rows.length - 1];
  const costUsd = Math.round(totalKwh * kwhRate * 100) / 100;

  // Cloud cost comparison
  let cloudComparison = null;
  const gpuName = firstRow.gpu_name as string | null;
  if (gpuName) {
    const { data: refs } = await supabase
      .from("gpu_reference_pricing")
      .select("gpu_model, provider, rate_usd_per_gpu_hour")
      .order("rate_usd_per_gpu_hour", { ascending: false });

    // Find matching reference price (highest on-demand = worst-case cloud cost)
    const match = ((refs ?? []) as unknown as Record<string, unknown>[]).find(
      (r) =>
        gpuName.includes(r.gpu_model as string) ||
        (r.gpu_model as string).includes(gpuName.replace("NVIDIA ", ""))
    );

    if (match) {
      const durationHours =
        (new Date(lastRow.time).getTime() - new Date(firstRow.time).getTime()) / 3_600_000;
      const gpuHours = durationHours * Math.max(gpuUuids.size, 1);
      const cloudCost =
        Math.round((match.rate_usd_per_gpu_hour as number) * gpuHours * 100) / 100;
      const savingsUsd = Math.round((cloudCost - costUsd) * 100) / 100;

      cloudComparison = {
        cloud_equivalent_usd: cloudCost,
        savings_usd: savingsUsd,
        savings_pct:
          cloudCost > 0
            ? Math.round((savingsUsd / cloudCost) * 10000) / 100
            : 0,
        reference_provider: match.provider,
        reference_rate_hr: match.rate_usd_per_gpu_hour,
      };
    }
  }

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
      cost_usd: costUsd,
      total_co2e_g: Math.round(totalCo2eG * 100) / 100,
      max_power_w: Math.round(maxPowerW),
      peak_utilization_pct: Math.round(peakUtil),
      sample_count: rows.length,
    },
    cloud_comparison: cloudComparison,
    timeseries: rows,
  });
}
