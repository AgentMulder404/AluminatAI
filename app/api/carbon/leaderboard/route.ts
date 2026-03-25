// GET /api/carbon/leaderboard
// Public — no auth required. 1-hour CDN cache.
// Calls carbon_leaderboard_stats() — groups with <5 submissions are excluded (k-anonymity).

import { NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET() {
  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.rpc("carbon_leaderboard_stats");

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data ?? [], {
    headers: {
      "Cache-Control": "public, max-age=3600, stale-while-revalidate=300",
    },
  });
}
