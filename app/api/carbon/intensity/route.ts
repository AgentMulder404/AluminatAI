// GET /api/carbon/intensity?zone=US-CAL-CISO
// Public — no auth required.
// Returns the latest cached carbon intensity for a grid zone.
// The cron job at /api/cron/carbon-refresh keeps the cache fresh (hourly).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

const ZONE_RE = /^[A-Z0-9\-]+$/i;

export async function GET(req: NextRequest) {
  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "unknown";
  const rl = await rateLimit(`pub:intensity:${ip}`, 30);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const zone = req.nextUrl.searchParams.get("zone") ?? "";

  if (!zone || zone.length > 32 || !ZONE_RE.test(zone)) {
    return NextResponse.json(
      { error: "Missing or invalid zone parameter" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase
    .from("carbon_intensities")
    .select("zone, carbon_g_per_kwh, fetched_at, is_estimated, source")
    .eq("zone", zone.toUpperCase())
    .order("fetched_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  if (!data) {
    return NextResponse.json({ zone: zone.toUpperCase(), carbon_g_per_kwh: null });
  }

  return NextResponse.json(
    data,
    { headers: { "Cache-Control": "public, max-age=3600, stale-while-revalidate=300" } }
  );
}
