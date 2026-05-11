import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

export async function POST(
  req: NextRequest,
  { params }: { params: Promise<{ commandId: string }> }
) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`agent-cmd-result:${auth.userId}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const { commandId } = await params;

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { success, message, machine_id } = body as {
    success?: boolean;
    message?: string;
    machine_id?: string;
  };

  const supabase = createSupabaseServerClient();

  // Verify the command belongs to this user AND machine
  const { data: existing } = await supabase
    .from("agent_commands")
    .select("id, machine_id")
    .eq("id", commandId)
    .eq("user_id", auth.userId)
    .single();

  if (!existing) {
    return NextResponse.json({ error: "Command not found" }, { status: 404 });
  }

  if (machine_id && existing.machine_id !== machine_id) {
    return NextResponse.json({ error: "Machine ID mismatch" }, { status: 403 });
  }

  const newStatus = success ? "applied" : "failed";

  const { error } = await supabase
    .from("agent_commands")
    .update({
      status: newStatus,
      result: { success, message },
      completed_at: new Date().toISOString(),
    })
    .eq("id", commandId)
    .eq("user_id", auth.userId);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  // If command was linked to a recommendation, update its status too
  const { data: cmd } = await supabase
    .from("agent_commands")
    .select("recommendation_id")
    .eq("id", commandId)
    .single();

  if (cmd?.recommendation_id) {
    const recStatus = success ? "applied" : "pending";
    await supabase
      .from("optimization_recommendations")
      .update({
        status: recStatus,
        ...(success && { applied_at: new Date().toISOString() }),
      })
      .eq("id", cmd.recommendation_id);

    // Audit trail
    await supabase.from("recommendation_actions").insert({
      recommendation_id: cmd.recommendation_id,
      user_id: auth.userId,
      action: success ? "applied" : "feedback_recorded",
      metadata: { success, message, command_id: commandId },
    });
  }

  return NextResponse.json({ ok: true, status: newStatus });
}
