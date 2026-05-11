import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

export async function POST(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`swarm-lease:${auth.userId}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { machine_id, cluster_tag = "" } = body as {
    machine_id?: string;
    cluster_tag?: string;
  };

  if (!machine_id) {
    return NextResponse.json({ error: "machine_id required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();
  const now = new Date();
  const leaseDurationMs = 10 * 60 * 1000; // 10 minutes (2x the default 5-min eval interval)
  const expiresAt = new Date(now.getTime() + leaseDurationMs).toISOString();

  // Check for existing lease
  const { data: existing } = await supabase
    .from("swarm_leader_leases")
    .select("machine_id, lease_token, expires_at")
    .eq("user_id", auth.userId)
    .eq("cluster_tag", cluster_tag ?? "")
    .single();

  // If lease exists and is still valid for a different machine, deny
  if (existing && existing.machine_id !== machine_id) {
    const expiresTime = new Date(existing.expires_at).getTime();
    if (expiresTime > now.getTime()) {
      return NextResponse.json({
        leader: false,
        current_leader: existing.machine_id,
        expires_in_s: Math.round((expiresTime - now.getTime()) / 1000),
      });
    }
  }

  // Grant or renew lease
  const leaseToken = crypto.randomUUID();
  const { error } = await supabase
    .from("swarm_leader_leases")
    .upsert({
      user_id: auth.userId,
      cluster_tag: cluster_tag ?? "",
      machine_id,
      lease_token: leaseToken,
      acquired_at: now.toISOString(),
      expires_at: expiresAt,
    }, { onConflict: "user_id,cluster_tag" });

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({
    leader: true,
    lease_token: leaseToken,
    expires_at: expiresAt,
  });
}
