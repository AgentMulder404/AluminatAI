// GET /api/cron/carbon-refresh
// Schedule: 0 * * * * (every hour)
// Fetches latest carbon intensity from Electricity Maps for all active grid zones
// and caches the results in carbon_intensities.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { verifyCronSecret } from "@/lib/auth-helpers";

export const runtime = "edge";

interface ElectricityMapsResponse {
  zone: string;
  carbonIntensity: number;
  isEstimated: boolean;
  estimationMethod?: string;
}

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const apiKey = process.env.ELECTRICITY_MAPS_API_KEY ?? "";
  if (!apiKey) {
    return NextResponse.json({ skipped: true, reason: "no api key" });
  }

  const supabase = createSupabaseServerClient();

  // Find distinct zones seen in the last 24 hours
  const since = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString();
  const { data: zoneRows, error: zoneError } = await supabase
    .from("gpu_metrics")
    .select("grid_zone")
    .not("grid_zone", "is", null)
    .gte("time", since);

  if (zoneError) {
    return NextResponse.json({ error: zoneError.message }, { status: 500 });
  }

  const zones = [...new Set((zoneRows ?? []).map((r) => r.grid_zone as string).filter(Boolean))];

  if (zones.length === 0) {
    return NextResponse.json({ refreshed: 0, zones: [] });
  }

  const refreshed: string[] = [];

  for (const zone of zones) {
    try {
      const res = await fetch(
        `https://api.electricitymaps.com/v3/carbon-intensity/latest?zone=${encodeURIComponent(zone)}`,
        { headers: { "auth-token": apiKey } }
      );

      if (!res.ok) continue;

      const body = (await res.json()) as ElectricityMapsResponse;
      const carbonGPerKwh = body.carbonIntensity;
      if (typeof carbonGPerKwh !== "number" || carbonGPerKwh < 0) continue;

      const { error: insertError } = await supabase
        .from("carbon_intensities")
        .insert({
          zone,
          carbon_g_per_kwh: carbonGPerKwh,
          is_estimated: body.isEstimated ?? true,
          source: "electricity_maps",
        });

      if (!insertError) refreshed.push(zone);
    } catch {
      // Skip failed zones — partial success is fine
    }
  }

  // Prune rows older than 7 days
  const pruneOlderThan = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString();
  await supabase
    .from("carbon_intensities")
    .delete()
    .lt("fetched_at", pruneOlderThan);

  return NextResponse.json({ refreshed: refreshed.length, zones: refreshed });
}
