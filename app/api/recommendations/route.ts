import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user } } = await cookieClient.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const status = req.nextUrl.searchParams.get("status");
  const source = req.nextUrl.searchParams.get("source");
  const priority = req.nextUrl.searchParams.get("priority");
  const limit = Math.min(parseInt(req.nextUrl.searchParams.get("limit") ?? "100"), 500);

  const supabase = createSupabaseServerClient();

  // Exclude expired pending recommendations
  let query = supabase
    .from("optimization_recommendations")
    .select("*")
    .eq("user_id", user.id)
    .or("status.neq.pending,expires_at.is.null,expires_at.gt." + new Date().toISOString())
    .order("created_at", { ascending: false })
    .limit(limit);

  if (status) query = query.eq("status", status);
  if (source) query = query.eq("source", source);
  if (priority) query = query.eq("priority", priority);

  const { data: recs, error } = await query;
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  // Summary always from unfiltered counts so filters don't zero them out
  const { data: summaryRows } = await supabase
    .from("optimization_recommendations")
    .select("status, estimated_savings_pct")
    .eq("user_id", user.id)
    .or("status.neq.pending,expires_at.is.null,expires_at.gt." + new Date().toISOString());

  const all = summaryRows ?? [];
  const pendingRecs = all.filter((r) => r.status === "pending");
  const savingsValues = pendingRecs
    .map((r) => r.estimated_savings_pct ?? 0)
    .filter((v) => v > 0);
  const avgSavingsPct = savingsValues.length > 0
    ? savingsValues.reduce((a, b) => a + b, 0) / savingsValues.length
    : 0;

  return NextResponse.json({
    recommendations: recs ?? [],
    summary: {
      total: all.length,
      pending: pendingRecs.length,
      applied: all.filter((r) => r.status === "applied").length,
      rejected: all.filter((r) => r.status === "rejected").length,
      avg_savings_pct: Math.round(avgSavingsPct * 10) / 10,
    },
  });
}
