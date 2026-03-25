// GET /api/carbon/rank?gpu_arch=A100&model_tag=llama3&grid_zone=US-CAL-CISO
// Cookie-auth. Returns the user's CO₂e percentile within their peer group.

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

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`carbon-rank:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  if (!process.env.BENCHMARK_SALT) {
    return NextResponse.json({ error: "Not configured" }, { status: 500 });
  }

  const gpu_arch = req.nextUrl.searchParams.get("gpu_arch") ?? "";
  const model_tag = req.nextUrl.searchParams.get("model_tag") ?? "";
  const grid_zone = req.nextUrl.searchParams.get("grid_zone") ?? "";

  if (!gpu_arch || !model_tag || !grid_zone) {
    return NextResponse.json(
      { error: "gpu_arch, model_tag, and grid_zone are required" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  // Check opt-in
  const { data: profile } = await supabase
    .from("users")
    .select("benchmark_opt_in")
    .eq("id", user.id)
    .single();

  if (!profile?.benchmark_opt_in) {
    return NextResponse.json({ opt_in_required: true });
  }

  const userHash = await computeUserHash(user.id).catch(() => null);
  if (!userHash) {
    return NextResponse.json({ error: "Not configured" }, { status: 500 });
  }

  // Get user's own submission
  const { data: own } = await supabase
    .from("carbon_submissions")
    .select("co2e_g_per_gpu_hour")
    .eq("user_hash", userHash)
    .eq("gpu_arch", gpu_arch)
    .eq("model_tag", model_tag)
    .eq("grid_zone", grid_zone)
    .maybeSingle();

  if (!own) {
    return NextResponse.json({ no_data: true });
  }

  // Get all submissions in the peer group
  const { data: peers } = await supabase
    .from("carbon_submissions")
    .select("co2e_g_per_gpu_hour")
    .eq("gpu_arch", gpu_arch)
    .eq("model_tag", model_tag)
    .eq("grid_zone", grid_zone);

  const values = (peers ?? []).map((p) => Number(p.co2e_g_per_gpu_hour)).sort((a, b) => a - b);

  if (values.length < 5) {
    return NextResponse.json({ insufficient_data: true, peer_count: values.length });
  }

  const userVal = Number(own.co2e_g_per_gpu_hour);
  const rank = values.filter((v) => v <= userVal).length;
  const rawPercentile = rank / values.length;
  // Lower CO₂e is better — top_pct = how much better than peers
  const topPct = Math.round((1 - rawPercentile) * 100);

  return NextResponse.json({
    top_pct: topPct,
    raw_percentile: rawPercentile,
    user_value: userVal,
    peer_count: values.length,
    gpu_arch,
    model_tag,
    grid_zone,
  });
}
