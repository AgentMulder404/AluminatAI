// GET /api/cost/waste
// Returns waste detection events for the authenticated user.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requireRole } from "@/lib/rbac";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`cost-waste:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const params = req.nextUrl.searchParams;
  const from = params.get("from");
  const to = params.get("to");
  const clusterTag = params.get("cluster_tag") ?? "";
  const teamId = params.get("team_id") ?? "";
  const showDismissed = params.get("dismissed") === "true";

  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("waste_events")
    .select(
      "id, gpu_uuid, gpu_name, job_id, team_id, cluster_tag, waste_type, " +
        "avg_utilization_pct, duration_hours, estimated_waste_usd, detected_at, dismissed"
    )
    .eq("user_id", user.id)
    .order("detected_at", { ascending: false })
    .limit(200);

  if (!showDismissed) {
    query = query.eq("dismissed", false);
  }

  if (teamId) {
    query = query.eq("team_id", teamId);
  }
  if (from) {
    query = query.gte("detected_at", new Date(from).toISOString());
  }
  if (to) {
    query = query.lte("detected_at", new Date(to).toISOString());
  }
  if (clusterTag) {
    query = query.eq("cluster_tag", clusterTag);
  }

  const { data, error } = await query;

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  const events = (data ?? []) as unknown as Record<string, unknown>[];
  const totalWaste = events.reduce(
    (sum, e) => sum + Number(e.estimated_waste_usd),
    0
  );

  return NextResponse.json({
    events,
    total_waste_usd: Math.round(totalWaste * 100) / 100,
  });
}
