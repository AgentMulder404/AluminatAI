import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { requireRole } from "@/lib/rbac";

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

  const clusterTag = req.nextUrl.searchParams.get("cluster_tag") ?? "";
  const teamId = req.nextUrl.searchParams.get("team_id") ?? "";

  // RBAC check if team_id is provided
  if (teamId) {
    const { allowed } = await requireRole(user.id, teamId, "viewer");
    if (!allowed) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 });
    }
  }

  const supabase = createSupabaseServerClient();
  const startOfDay = new Date();
  startOfDay.setUTCHours(0, 0, 0, 0);

  let query = supabase
    .from("gpu_metrics")
    .select("energy_delta_j, power_draw_w, carbon_g_per_kwh, grid_zone, gpu_fraction")
    .eq("user_id", user.id)
    .gte("time", startOfDay.toISOString());

  if (teamId) {
    query = query.eq("team_id", teamId);
  }
  if (clusterTag) {
    query = query.eq("cluster_tag", clusterTag);
  }

  const { data, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  let totalJ = 0;
  let totalCo2eG: number | null = null;
  let gridZone: string | null = null;

  for (const r of data ?? []) {
    const frac = r.gpu_fraction ?? 1;
    const energyJ = (r.energy_delta_j ?? 0) * frac;
    totalJ += energyJ;

    if (r.carbon_g_per_kwh != null) {
      const co2g = (energyJ / 3_600_000) * r.carbon_g_per_kwh;
      totalCo2eG = (totalCo2eG ?? 0) + co2g;
    }
    if (r.grid_zone && !gridZone) gridZone = r.grid_zone;
  }

  const totalKwh = totalJ / 3_600_000;
  const kwhRate = await getUserKwhRate(user.id);
  const costUsd = totalKwh * kwhRate;

  return NextResponse.json({ cost_usd: costUsd, kwh: totalKwh, co2e_g: totalCo2eG, grid_zone: gridZone });
}
