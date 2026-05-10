import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user } } = await cookieClient.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await params;
  const supabase = createSupabaseServerClient();

  // Verify ownership
  const { data: rec, error: fetchErr } = await supabase
    .from("optimization_recommendations")
    .select("id, status, user_id")
    .eq("id", id)
    .eq("user_id", user.id)
    .single();

  if (fetchErr || !rec) {
    return NextResponse.json({ error: "Recommendation not found" }, { status: 404 });
  }

  if (rec.status !== "pending") {
    return NextResponse.json(
      { error: `Cannot reject recommendation with status '${rec.status}'` },
      { status: 400 }
    );
  }

  await supabase
    .from("optimization_recommendations")
    .update({ status: "rejected" })
    .eq("id", id);

  // Audit trail
  await supabase.from("recommendation_actions").insert({
    recommendation_id: id,
    user_id: user.id,
    action: "rejected",
    metadata: {},
  });

  return NextResponse.json({ ok: true, status: "rejected" });
}
