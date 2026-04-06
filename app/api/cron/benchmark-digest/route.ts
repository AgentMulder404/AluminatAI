// GET /api/cron/benchmark-digest
// Schedule: 0 10 * * 1 (Monday 10:00 UTC)
// Warms the benchmark_leaderboard_stats() query plan.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { verifyCronSecret } from "@/lib/auth-helpers";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.rpc("benchmark_leaderboard_stats");

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({
    refreshed: true,
    groups: (data ?? []).length,
  });
}
