// GET /api/benchmarks/leaderboard
// Public — no auth required.
// Calls benchmark_leaderboard_stats() RPC (only groups with ≥10 submissions).
// Cached 1 hour at the CDN edge.

import { NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET() {
  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.rpc("benchmark_leaderboard_stats");

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data ?? [], {
    headers: {
      "Cache-Control": "public, s-maxage=3600, stale-while-revalidate=86400",
    },
  });
}
