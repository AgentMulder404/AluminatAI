import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

export async function GET(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`agent-cmd:${auth.userId}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const machineId = req.nextUrl.searchParams.get("machine_id");
  if (!machineId) {
    return NextResponse.json({ error: "machine_id required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  // Fetch pending commands for this machine
  const { data: commands, error } = await supabase
    .from("agent_commands")
    .select("id, command_type, params, recommendation_id, created_at")
    .eq("user_id", auth.userId)
    .eq("machine_id", machineId)
    .eq("status", "pending")
    .order("created_at", { ascending: true })
    .limit(10);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  // Mark fetched commands as dispatched
  if (commands && commands.length > 0) {
    const ids = commands.map((c) => c.id);
    await supabase
      .from("agent_commands")
      .update({ status: "dispatched", dispatched_at: new Date().toISOString() })
      .in("id", ids);
  }

  return NextResponse.json({ commands: commands ?? [] });
}
