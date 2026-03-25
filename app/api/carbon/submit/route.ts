// POST /api/carbon/submit
// Cookie-auth. Requires benchmark_opt_in = true (shared flag).
// Queries last 30 days of gpu_metrics where carbon_g_per_kwh IS NOT NULL,
// groups by (gpu_name, model_tag, grid_zone), computes gCO₂e/GPU-hr,
// and upserts into carbon_submissions.
// User ID is never stored — only an HMAC-SHA256 hash.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export const runtime = "edge";

async function computeUserHash(userId: string): Promise<string> {
  const salt = process.env.BENCHMARK_SALT ?? "";
  if (!salt) throw new Error("BENCHMARK_SALT not configured");

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(salt),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const buf = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(userId));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

interface MetricRow {
  power_draw_w: number | null;
  energy_delta_j: number | null;
  gpu_fraction: number | null;
  gpu_name: string | null;
  model_tag: string | null;
  grid_zone: string | null;
  carbon_g_per_kwh: number | null;
}

export async function POST(_req: NextRequest) {
  if (!process.env.BENCHMARK_SALT) {
    return NextResponse.json(
      { error: "Carbon submissions not configured (missing BENCHMARK_SALT)" },
      { status: 500 }
    );
  }

  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`carbon-submit:${user.id}`, 10);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();

  const { data: profile } = await supabase
    .from("users")
    .select("benchmark_opt_in")
    .eq("id", user.id)
    .single();

  if (!profile?.benchmark_opt_in) {
    return NextResponse.json({ error: "Opt-in required" }, { status: 403 });
  }

  const since = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
  const { data: metrics, error: metricsError } = await supabase
    .from("gpu_metrics")
    .select("power_draw_w, energy_delta_j, gpu_fraction, gpu_name, model_tag, grid_zone, carbon_g_per_kwh")
    .eq("user_id", user.id)
    .not("carbon_g_per_kwh", "is", null)
    .not("model_tag", "is", null)
    .not("gpu_name", "is", null)
    .not("grid_zone", "is", null)
    .gte("time", since);

  if (metricsError) {
    return NextResponse.json({ error: metricsError.message }, { status: 500 });
  }

  if (!metrics || metrics.length === 0) {
    return NextResponse.json({ submitted: 0, reason: "no_carbon_data" });
  }

  // Group by (gpu_name, model_tag, grid_zone)
  type GroupKey = string;
  const groups = new Map<
    GroupKey,
    {
      powerSamples: number[];
      carbonSamples: number[];
      gpu_name: string;
      model_tag: string;
      grid_zone: string;
    }
  >();

  for (const m of metrics as MetricRow[]) {
    if (!m.gpu_name || !m.model_tag || !m.grid_zone || m.carbon_g_per_kwh == null) continue;
    const power = m.power_draw_w;
    if (typeof power !== "number" || power <= 0 || power > 1500) continue;

    const key: GroupKey = `${m.gpu_name}|||${m.model_tag}|||${m.grid_zone}`;
    const grp = groups.get(key) ?? {
      powerSamples: [],
      carbonSamples: [],
      gpu_name: m.gpu_name,
      model_tag: m.model_tag,
      grid_zone: m.grid_zone,
    };
    grp.powerSamples.push(power);
    grp.carbonSamples.push(m.carbon_g_per_kwh);
    groups.set(key, grp);
  }

  if (groups.size === 0) {
    return NextResponse.json({ submitted: 0 });
  }

  let userHash: string;
  try {
    userHash = await computeUserHash(user.id);
  } catch {
    return NextResponse.json({ error: "Carbon submissions not configured" }, { status: 500 });
  }

  const optInAt = new Date().toISOString();
  let submitted = 0;

  for (const [, grp] of groups) {
    if (grp.powerSamples.length === 0) continue;

    // Median power
    const sorted = [...grp.powerSamples].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const medianPower =
      sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

    // Average carbon intensity
    const avgCarbon =
      grp.carbonSamples.reduce((a, b) => a + b, 0) / grp.carbonSamples.length;

    // gCO₂e per GPU-hour = avg_power_W * avg_carbon_g_per_kWh * 3.6e-3
    //   (W * g/kWh * 3600s/hr * 1kWh/3.6e6J = g/hr)
    const co2eGPerGpuHour = medianPower * avgCarbon * 3.6e-3;
    const kwhPerGpuHour = (medianPower * 3600) / 3_600_000; // = W/1000

    if (co2eGPerGpuHour <= 0 || co2eGPerGpuHour > 1e7) continue;

    const { error: upsertError } = await supabase
      .from("carbon_submissions")
      .upsert(
        {
          user_hash: userHash,
          gpu_arch: grp.gpu_name,
          model_tag: grp.model_tag,
          grid_zone: grp.grid_zone,
          co2e_g_per_gpu_hour: co2eGPerGpuHour,
          kwh_per_gpu_hour: kwhPerGpuHour,
          carbon_g_per_kwh: avgCarbon,
          duration_seconds: grp.powerSamples.length,
          opt_in_at: optInAt,
          updated_at: optInAt,
        },
        { onConflict: "user_hash,gpu_arch,model_tag,grid_zone" }
      );

    if (!upsertError) submitted++;
  }

  return NextResponse.json({ submitted });
}
