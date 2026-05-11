import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const authHeader = req.headers.get("authorization");
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();
  const now = new Date().toISOString();

  const { data: expired, error } = await supabase
    .from("optimization_recommendations")
    .update({ status: "expired" })
    .eq("status", "pending")
    .lt("expires_at", now)
    .select("id, user_id");

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const expiredList = expired ?? [];

  if (expiredList.length > 0) {
    const actions = expiredList.map((rec) => ({
      recommendation_id: rec.id,
      user_id: rec.user_id,
      action: "expired",
      metadata: { reason: "auto_expired", expired_at: now },
    }));

    await supabase.from("recommendation_actions").insert(actions);
  }

  return NextResponse.json({
    ok: true,
    expired_count: expiredList.length,
  });
}
