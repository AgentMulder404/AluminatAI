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

  const { data: rec, error: fetchErr } = await supabase
    .from("optimization_recommendations")
    .select("*")
    .eq("id", id)
    .eq("user_id", user.id)
    .single();

  if (fetchErr || !rec) {
    return NextResponse.json({ error: "Recommendation not found" }, { status: 404 });
  }

  if (rec.status !== "applied") {
    return NextResponse.json(
      { error: `Cannot rollback recommendation with status '${rec.status}'` },
      { status: 400 }
    );
  }

  const payload = rec.action_payload as Record<string, unknown> | null;
  if (!payload?.command) {
    return NextResponse.json({ error: "No rollback action available" }, { status: 400 });
  }

  // Determine rollback command from the original
  const rollbackCommand = payload.command === "apply_power_cap"
    ? "rollback_power_cap"
    : null;

  if (!rollbackCommand) {
    return NextResponse.json({ error: "Rollback not supported for this action" }, { status: 400 });
  }

  await supabase
    .from("optimization_recommendations")
    .update({ status: "rolled_back", rolled_back_at: new Date().toISOString() })
    .eq("id", id);

  await supabase.from("agent_commands").insert({
    user_id: user.id,
    machine_id: rec.machine_id,
    recommendation_id: id,
    command_type: rollbackCommand,
    params: { gpu_index: payload.gpu_index },
    status: "pending",
  });

  await supabase.from("recommendation_actions").insert({
    recommendation_id: id,
    user_id: user.id,
    action: "rolled_back",
    metadata: { rollback_command: rollbackCommand },
  });

  return NextResponse.json({ ok: true, status: "rolled_back" });
}
