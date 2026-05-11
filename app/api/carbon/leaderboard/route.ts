// GET /api/carbon/leaderboard
// Public — no auth required. 1-hour CDN cache.
// Calls carbon_leaderboard_stats() — groups with <5 submissions are excluded (k-anonymity).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

export async function GET(req: NextRequest) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  const rl = await rateLimit(`pub:carbon:${ip}`, 30);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase.rpc("carbon_leaderboard_stats");

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json(data ?? [], {
    headers: {
      "Cache-Control": "public, max-age=3600, stale-while-revalidate=300",
    },
  });
}
