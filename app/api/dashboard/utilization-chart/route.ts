import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const cookieClient = createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const clusterTag = req.nextUrl.searchParams.get("cluster_tag") ?? "";

  const supabase = createSupabaseServerClient();

  // Last 24 hours, 5-minute buckets (288 points max)
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();

  let query = supabase
    .from("gpu_metrics")
    .select("time, utilization_gpu_pct, power_draw_w")
    .eq("user_id", user.id)
    .gte("time", since)
    .order("time", { ascending: true });

  if (clusterTag) {
    query = query.eq("cluster_tag", clusterTag);
  }

  const { data, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data ?? []);
}
