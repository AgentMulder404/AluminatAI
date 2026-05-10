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

  // Fetch the recommendation
  const { data: rec, error: fetchErr } = await supabase
    .from("optimization_recommendations")
    .select("*")
    .eq("id", id)
    .eq("user_id", user.id)
    .single();

  if (fetchErr || !rec) {
    return NextResponse.json({ error: "Recommendation not found" }, { status: 404 });
  }

  if (rec.status !== "pending") {
    return NextResponse.json(
      { error: `Cannot approve recommendation with status '${rec.status}'` },
      { status: 400 }
    );
  }

  // Update recommendation status
  await supabase
    .from("optimization_recommendations")
    .update({
      status: "approved",
      approved_by: user.id,
      approved_at: new Date().toISOString(),
    })
    .eq("id", id);

  // Create a command for the agent to pick up (if action_payload has a command)
  const payload = rec.action_payload as Record<string, unknown> | null;
  if (payload && payload.command) {
    await supabase.from("agent_commands").insert({
      user_id: user.id,
      machine_id: rec.machine_id,
      recommendation_id: id,
      command_type: payload.command as string,
      params: payload,
      status: "pending",
    });
  }

  // Audit trail
  await supabase.from("recommendation_actions").insert({
    recommendation_id: id,
    user_id: user.id,
    action: "approved",
    metadata: {},
  });

  return NextResponse.json({ ok: true, status: "approved" });
}
