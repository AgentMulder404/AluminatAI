import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export const runtime = "edge";

interface FleetGpuRow {
  machine_id: string;
  gpu_index: number;
  gpu_name: string | null;
  power_draw_w: number | null;
  power_limit_w: number | null;
  utilization_gpu_pct: number | null;
  temperature_c: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  energy_j_last_hour: number | null;
  model_tag: string | null;
  job_id: string | null;
  grid_zone: string | null;
  carbon_g_per_kwh: number | null;
  sample_time: string;
}

export async function GET(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`fleet-state:${auth.userId}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();
  const tenMinAgo = new Date(Date.now() - 600_000).toISOString();

  // Get all active machines (heartbeat within last 10 minutes)
  const { data: heartbeats } = await supabase
    .from("agent_heartbeats")
    .select("machine_id, hostname, gpu_count, cluster_tag, agent_version, last_seen_at, gpu_names")
    .eq("user_id", auth.userId)
    .gte("last_seen_at", tenMinAgo);

  if (!heartbeats || heartbeats.length === 0) {
    return NextResponse.json({ machines: [] });
  }

  // Use DB function for pre-aggregated latest metrics (no row limit)
  const { data: gpuRows, error: rpcError } = await supabase
    .rpc("get_fleet_gpu_latest", { p_user_id: auth.userId });

  const rows = (rpcError ? [] : (gpuRows ?? [])) as unknown as FleetGpuRow[];

  // Index GPU data by machine_id for O(1) lookup
  const gpusByMachine = new Map<string, FleetGpuRow[]>();
  for (const row of rows) {
    const arr = gpusByMachine.get(row.machine_id);
    if (arr) arr.push(row);
    else gpusByMachine.set(row.machine_id, [row]);
  }

  const now = Date.now();
  const machines = heartbeats.map((hb) => {
    const machineGpus = gpusByMachine.get(hb.machine_id) ?? [];
    const gpus = machineGpus.map((g) => ({
      gpu_index: g.gpu_index,
      gpu_name: g.gpu_name ?? "",
      power_draw_w: g.power_draw_w ?? 0,
      power_limit_w: g.power_limit_w ?? 0,
      utilization_pct: g.utilization_gpu_pct ?? 0,
      temperature_c: g.temperature_c ?? 0,
      memory_used_mb: g.memory_used_mb ?? 0,
      memory_total_mb: g.memory_total_mb ?? 0,
      energy_j_last_hour: g.energy_j_last_hour ?? 0,
      model_tag: g.model_tag ?? null,
      job_id: g.job_id ?? null,
      is_idle: (g.utilization_gpu_pct ?? 0) < 5,
    }));

    const firstGpu = machineGpus[0];

    return {
      machine_id: hb.machine_id,
      hostname: hb.hostname ?? "",
      cluster_tag: hb.cluster_tag ?? "",
      gpu_count: hb.gpu_count ?? gpus.length,
      gpus,
      last_heartbeat_age_s: Math.round((now - new Date(hb.last_seen_at).getTime()) / 1000),
      carbon_intensity_gco2e: firstGpu?.carbon_g_per_kwh ?? 0,
      grid_zone: firstGpu?.grid_zone ?? "",
      agent_version: hb.agent_version ?? "",
    };
  });

  return NextResponse.json({ machines });
}
