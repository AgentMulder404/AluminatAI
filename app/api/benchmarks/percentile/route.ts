// GET /api/benchmarks/percentile
// Cookie-auth, 60 req/min. Returns the user's best J/GPU-hr submission and
// their percentile rank within their (gpu_arch, model_tag) peer group.

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
  const buf = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(userId)
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export async function GET(_req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`benchmarks-percentile:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  // Check opt-in
  const supabase = createSupabaseServerClient();
  const { data: profile } = await supabase
    .from("users")
    .select("benchmark_opt_in")
    .eq("id", user.id)
    .single();

  if (!profile?.benchmark_opt_in) {
    return NextResponse.json(
      { opt_in_required: true },
      { headers: getRateLimitHeaders(rl) }
    );
  }

  // Compute user's hash
  let userHash: string;
  try {
    userHash = await computeUserHash(user.id);
  } catch {
    return NextResponse.json(
      { error: "Benchmarking not configured" },
      { status: 500 }
    );
  }

  // Get user's best submission (lowest J/GPU-hr = most efficient)
  const { data: best } = await supabase
    .from("benchmark_submissions")
    .select("energy_j_per_gpu_hour, gpu_arch, model_tag, avg_power_w, tokens_per_second, kwh_per_1m_tokens, framework_tag")
    .eq("user_hash", userHash)
    .order("energy_j_per_gpu_hour", { ascending: true })
    .limit(1)
    .maybeSingle();

  if (!best) {
    return NextResponse.json(
      { no_data: true },
      { headers: getRateLimitHeaders(rl) }
    );
  }

  const { gpu_arch, model_tag, energy_j_per_gpu_hour } = best;

  // Count peers in same group
  const { count: peerCount } = await supabase
    .from("benchmark_submissions")
    .select("*", { count: "exact", head: true })
    .eq("gpu_arch", gpu_arch)
    .eq("model_tag", model_tag);

  const total = peerCount ?? 0;

  if (total < 10) {
    return NextResponse.json(
      { insufficient_data: true, peer_count: total },
      { headers: getRateLimitHeaders(rl) }
    );
  }

  // Count submissions at or below user's value (rank)
  const { count: rankCount } = await supabase
    .from("benchmark_submissions")
    .select("*", { count: "exact", head: true })
    .eq("gpu_arch", gpu_arch)
    .eq("model_tag", model_tag)
    .lte("energy_j_per_gpu_hour", energy_j_per_gpu_hour);

  const rank = rankCount ?? 1;
  const rawPercentile = (rank / total) * 100;
  const topPct = Math.round(100 - rawPercentile);

  return NextResponse.json(
    {
      top_pct: topPct,
      raw_percentile: Math.round(rawPercentile),
      user_value: Number(energy_j_per_gpu_hour),
      peer_count: total,
      gpu_arch,
      model_tag,
      kwh_per_1m_tokens: best.kwh_per_1m_tokens ?? null,
      framework_tag: best.framework_tag ?? null,
    },
    { headers: getRateLimitHeaders(rl) }
  );
}
