import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

const ONLINE_THRESHOLD_MIN = 10;

export async function GET(req: NextRequest) {
  // Auth via dashboard session
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Rate limit: 60 req/min per user
  const rl = await rateLimit(`agents:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase
    .from("agent_heartbeats")
    .select(
      "hostname, machine_id, cluster_tag, location_hint, gpu_count, gpu_names, agent_version, scheduler, last_seen"
    )
    .eq("user_id", user.id)
    .order("last_seen", { ascending: false });

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  const cutoff = new Date(Date.now() - ONLINE_THRESHOLD_MIN * 60 * 1000);
  const agents = ((data ?? []) as unknown as Record<string, unknown>[]).map((row) => ({
    ...row,
    is_online: new Date(row.last_seen as string) > cutoff,
  }));

  return NextResponse.json(agents, {
    headers: getRateLimitHeaders(rl),
  });
}
