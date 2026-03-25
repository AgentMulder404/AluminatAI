// POST /api/benchmarks/submit
// Cookie-auth. Computes J/GPU-hr from the user's gpu_metrics (last 30 days),
// groups by (gpu_name, model_tag), and upserts into benchmark_submissions.
// The user's ID is never stored — only an HMAC-SHA256 hash.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

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
  const buf = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(userId)
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

interface MetricRow {
  power_draw_w: number | null;
  energy_delta_j: number | null;
  gpu_name: string | null;
  model_tag: string | null;
  gpu_fraction: number | null;
  time: string;
}

export async function POST(_req: NextRequest) {
  // Guard: BENCHMARK_SALT must be set to avoid all users mapping to same hash
  if (!process.env.BENCHMARK_SALT) {
    return NextResponse.json(
      { error: "Benchmarking not configured (missing BENCHMARK_SALT)" },
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

  // Check opt-in
  const supabase = createSupabaseServerClient();
  const { data: profile } = await supabase
    .from("users")
    .select("benchmark_opt_in")
    .eq("id", user.id)
    .single();

  if (!profile?.benchmark_opt_in) {
    return NextResponse.json({ error: "Opt-in required" }, { status: 403 });
  }

  // Query last 30 days of gpu_metrics with model_tag set
  const since = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
  const { data: metrics, error: metricsError } = await supabase
    .from("gpu_metrics")
    .select("power_draw_w, energy_delta_j, gpu_name, model_tag, gpu_fraction, time")
    .eq("user_id", user.id)
    .not("model_tag", "is", null)
    .not("gpu_name", "is", null)
    .gte("time", since);

  if (metricsError) {
    return NextResponse.json({ error: metricsError.message }, { status: 500 });
  }

  if (!metrics || metrics.length === 0) {
    return NextResponse.json({ submitted: 0 });
  }

  // Group by (gpu_name, model_tag) → compute avg_power_w and J/GPU-hr
  type GroupKey = string;
  const groups = new Map<
    GroupKey,
    { powerSamples: number[]; gpu_name: string; model_tag: string }
  >();

  for (const m of metrics as MetricRow[]) {
    if (!m.gpu_name || !m.model_tag) continue;
    const key: GroupKey = `${m.gpu_name}|||${m.model_tag}`;
    const power = m.power_draw_w;
    if (typeof power !== "number" || power <= 0 || power > 1500) continue;

    const grp = groups.get(key) ?? {
      powerSamples: [],
      gpu_name: m.gpu_name,
      model_tag: m.model_tag,
    };
    grp.powerSamples.push(power);
    groups.set(key, grp);
  }

  if (groups.size === 0) {
    return NextResponse.json({ submitted: 0 });
  }

  // Compute user_hash (HMAC of user.id)
  let userHash: string;
  try {
    userHash = await computeUserHash(user.id);
  } catch {
    return NextResponse.json(
      { error: "Benchmarking not configured" },
      { status: 500 }
    );
  }

  const optInAt = new Date().toISOString();
  let submitted = 0;

  for (const [, grp] of groups) {
    if (grp.powerSamples.length === 0) continue;

    // Median power for robustness against spikes
    const sorted = [...grp.powerSamples].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const medianPower =
      sorted.length % 2 === 0
        ? (sorted[mid - 1] + sorted[mid]) / 2
        : sorted[mid];

    // J/GPU-hour = avg_power_W * 3600 (watts × seconds-per-hour)
    const energyJPerGpuHour = medianPower * 3600;

    // Skip implausible values
    if (energyJPerGpuHour > 1e9 || energyJPerGpuHour <= 0) continue;

    const durationSeconds = grp.powerSamples.length; // ~1 sample/sec

    const { error: upsertError } = await supabase
      .from("benchmark_submissions")
      .upsert(
        {
          user_hash: userHash,
          gpu_arch: grp.gpu_name,
          model_tag: grp.model_tag,
          precision_tag: "unknown",
          avg_power_w: medianPower,
          energy_j_per_gpu_hour: energyJPerGpuHour,
          duration_seconds: durationSeconds,
          gpu_count: 1,
          opt_in_at: optInAt,
          updated_at: optInAt,
        },
        { onConflict: "user_hash,gpu_arch,model_tag" }
      );

    if (!upsertError) submitted++;
  }

  return NextResponse.json({ submitted });
}
