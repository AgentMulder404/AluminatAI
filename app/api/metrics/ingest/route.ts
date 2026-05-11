import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

// Field length limits for new identity fields
const MAX_IDENTITY_LEN = 64;

// Grid zone: matches Electricity Maps zone codes (e.g. US-CAL-CISO, DE, FR)
const ZONE_RE = /^[A-Z0-9\-]+$/i;
function sanitizeZone(val: unknown): string | null {
  if (typeof val !== "string") return null;
  if (val.length > 32) return null;
  if (!ZONE_RE.test(val)) return null;
  return val.toUpperCase();
}

interface MetricPayload {
  timestamp: string;
  gpu_uuid?: string;
  gpu_index?: number;
  gpu_name?: string;
  power_draw_w?: number;
  temperature_c?: number;
  utilization_gpu_pct?: number;
  memory_used_mb?: number;
  memory_total_mb?: number;
  energy_delta_j?: number;
  fan_speed_pct?: number;
  clocks_sm_mhz?: number;
  clocks_mem_mhz?: number;
  pcie_tx_kb?: number;
  pcie_rx_kb?: number;
  job_id?: string;
  team_id?: string;
  model_tag?: string;
  scheduler_source?: string;
  gpu_fraction?: number;
  attribution_confidence?: string;
  attribution_confidence_score?: number;
  // Phase 2 — machine identity
  machine_id?: string;
  cluster_tag?: string;
  // Carbon tracking
  grid_zone?: string;
}

function sanitizeIdentity(val: unknown): string | null {
  if (typeof val !== "string") return null;
  if (val.length > MAX_IDENTITY_LEN) return null;
  if (/[\r\n]/.test(val)) return null;
  return val;
}

export async function POST(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Rate limit: 100 req/min
  const rl = await rateLimit(`ingest:${auth.userId}`, 100);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  let metrics: MetricPayload[];
  try {
    const body = await req.json();
    metrics = Array.isArray(body) ? body : [body];
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (metrics.length === 0) {
    return NextResponse.json({ inserted: 0 });
  }

  // Cap batch size to prevent abuse
  if (metrics.length > 1000) {
    return NextResponse.json(
      { error: "Batch too large — max 1000 metrics per request" },
      { status: 400 }
    );
  }

  // Collect unique grid zones from this batch for a single carbon intensity lookup
  const supabase = createSupabaseServerClient();
  const zoneIntensityMap = new Map<string, number | null>();

  const uniqueZones = new Set(
    metrics.map((m) => sanitizeZone(m.grid_zone)).filter(Boolean) as string[]
  );

  for (const zone of uniqueZones) {
    const { data } = await supabase
      .from("carbon_intensities")
      .select("carbon_g_per_kwh")
      .eq("zone", zone)
      .order("fetched_at", { ascending: false })
      .limit(1)
      .maybeSingle();
    zoneIntensityMap.set(zone, data?.carbon_g_per_kwh ?? null);
  }

  // Validate + build insert rows
  const rows = metrics
    .map((m) => {
      // Basic server-side validation
      if (
        typeof m.power_draw_w === "number" &&
        (m.power_draw_w < 0 || m.power_draw_w > 1500)
      )
        return null;
      if (
        typeof m.temperature_c === "number" &&
        (m.temperature_c < 0 || m.temperature_c > 120)
      )
        return null;
      if (
        typeof m.utilization_gpu_pct === "number" &&
        (m.utilization_gpu_pct < 0 || m.utilization_gpu_pct > 100)
      )
        return null;
      if (
        typeof m.memory_used_mb === "number" &&
        (m.memory_used_mb < 0 || m.memory_used_mb > 1_000_000)
      )
        return null;
      if (
        typeof m.gpu_fraction === "number" &&
        (m.gpu_fraction < 0 || m.gpu_fraction > 1)
      )
        return null;

      // Reject timestamps more than 5 minutes from server time
      if (m.timestamp) {
        const metricTime = new Date(m.timestamp).getTime();
        if (isNaN(metricTime) || Math.abs(Date.now() - metricTime) > 5 * 60 * 1000)
          return null;
      }

      return {
        user_id: auth.userId,
        timestamp: m.timestamp,
        gpu_uuid: m.gpu_uuid ?? null,
        gpu_index: m.gpu_index ?? null,
        gpu_name: m.gpu_name ?? null,
        power_draw_w: m.power_draw_w ?? null,
        temperature_c: m.temperature_c ?? null,
        utilization_gpu_pct: m.utilization_gpu_pct ?? null,
        memory_used_mb: m.memory_used_mb ?? null,
        memory_total_mb: m.memory_total_mb ?? null,
        energy_delta_j: m.energy_delta_j ?? null,
        fan_speed_pct: m.fan_speed_pct ?? null,
        clocks_sm_mhz: m.clocks_sm_mhz ?? null,
        clocks_mem_mhz: m.clocks_mem_mhz ?? null,
        pcie_tx_kb: m.pcie_tx_kb ?? null,
        pcie_rx_kb: m.pcie_rx_kb ?? null,
        job_id: m.job_id ?? null,
        team_id: m.team_id ?? null,
        model_tag: m.model_tag ?? null,
        scheduler_source: m.scheduler_source ?? null,
        gpu_fraction: m.gpu_fraction ?? null,
        attribution_confidence: m.attribution_confidence ?? null,
        attribution_confidence_score: m.attribution_confidence_score ?? null,
        machine_id: sanitizeIdentity(m.machine_id),
        cluster_tag: sanitizeIdentity(m.cluster_tag) ?? null,
        grid_zone: sanitizeZone(m.grid_zone),
        carbon_g_per_kwh: (() => {
          const z = sanitizeZone(m.grid_zone);
          return z ? (zoneIntensityMap.get(z) ?? null) : null;
        })(),
      };
    })
    .filter(Boolean);

  if (rows.length === 0) {
    return NextResponse.json({ inserted: 0 });
  }

  const { error } = await supabase.from("gpu_metrics").insert(rows);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json(
    { inserted: rows.length },
    { headers: getRateLimitHeaders(rl) }
  );
}
