import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

const ONLINE_THRESHOLD_MIN = 10;

export async function GET(_req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Cluster summary from view
  const { data: summary, error: summaryError } = await supabase
    .from("cluster_summary")
    .select(
      "cluster_tag, machine_count, gpu_count, total_kwh, avg_power_w, avg_utilization_pct"
    )
    .eq("user_id", user.id);

  if (summaryError) {
    return NextResponse.json({ error: summaryError.message }, { status: 500 });
  }

  if (!summary || summary.length < 2) {
    // Single-cluster users see no filter UI
    return NextResponse.json([]);
  }

  // Count online agents per cluster (last_seen within 10 min)
  const cutoff = new Date(Date.now() - ONLINE_THRESHOLD_MIN * 60 * 1000).toISOString();
  const { data: hearts } = await supabase
    .from("agent_heartbeats")
    .select("cluster_tag")
    .eq("user_id", user.id)
    .gte("last_seen", cutoff);

  const onlineByCluster = new Map<string, number>();
  for (const h of hearts ?? []) {
    const key = h.cluster_tag || "default";
    onlineByCluster.set(key, (onlineByCluster.get(key) ?? 0) + 1);
  }

  const result = summary.map((row) => ({
    ...row,
    online_agent_count: onlineByCluster.get(row.cluster_tag) ?? 0,
  }));

  return NextResponse.json(result);
}
