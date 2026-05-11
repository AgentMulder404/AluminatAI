import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
  } = await cookieClient.auth.getUser();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`dash-activity:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const url = req.nextUrl;
  const limit = Math.min(parseInt(url.searchParams.get("limit") ?? "50"), 200);
  const offset = parseInt(url.searchParams.get("offset") ?? "0");
  const actionFilter = url.searchParams.get("action");
  const sourceFilter = url.searchParams.get("source");
  const priorityFilter = url.searchParams.get("priority");

  const supabase = createSupabaseServerClient();

  interface RecAction {
    id: string;
    action: string;
    metadata: Record<string, unknown>;
    created_at: string;
    optimization_recommendations: {
      title: string;
      hostname: string;
      machine_id: string;
      gpu_name: string | null;
      source: string;
      category: string;
      priority: string;
      estimated_savings_pct: number;
      status: string;
    } | null;
  }

  let recQuery = supabase
    .from("recommendation_actions")
    .select(
      "id, action, metadata, created_at, optimization_recommendations(title, hostname, machine_id, gpu_name, source, category, priority, estimated_savings_pct, status)",
      { count: "exact" }
    )
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (actionFilter) {
    const actions = actionFilter.split(",").map((a) => a.trim());
    const recActions = actions.filter(
      (a) => !a.startsWith("command_")
    );
    if (recActions.length > 0) {
      recQuery = recQuery.in("action", recActions);
    }
  }

  const { data: recActions, count: recCount } = (await recQuery
    .range(offset, offset + limit - 1)) as unknown as {
    data: RecAction[] | null;
    count: number | null;
  };

  interface CmdRow {
    id: string;
    command_type: string;
    machine_id: string;
    status: string;
    result: Record<string, unknown> | null;
    completed_at: string;
    recommendation_id: string | null;
    optimization_recommendations: {
      title: string;
      hostname: string;
      source: string;
      category: string;
      priority: string;
      gpu_name: string | null;
      estimated_savings_pct: number;
    } | null;
  }

  let includeCmds = true;
  if (actionFilter) {
    const actions = actionFilter.split(",").map((a) => a.trim());
    const cmdActions = actions.filter((a) => a.startsWith("command_"));
    if (actions.length > 0 && cmdActions.length === 0) {
      includeCmds = false;
    }
  }

  let cmdEvents: CmdRow[] = [];
  if (includeCmds) {
    const { data: cmds } = (await supabase
      .from("agent_commands")
      .select(
        "id, command_type, machine_id, status, result, completed_at, recommendation_id, optimization_recommendations(title, hostname, source, category, priority, gpu_name, estimated_savings_pct)"
      )
      .eq("user_id", user.id)
      .in("status", ["applied", "failed"])
      .not("completed_at", "is", null)
      .order("completed_at", { ascending: false })
      .limit(limit)) as unknown as { data: CmdRow[] | null };
    cmdEvents = cmds ?? [];
  }

  interface UnifiedEvent {
    id: string;
    event_type: string;
    action: string;
    timestamp: string;
    title: string | null;
    hostname: string | null;
    machine_id: string | null;
    gpu_name: string | null;
    source: string | null;
    category: string | null;
    priority: string | null;
    estimated_savings_pct: number | null;
    metadata: Record<string, unknown>;
  }

  const events: UnifiedEvent[] = [];

  for (const ra of recActions ?? []) {
    const rec = ra.optimization_recommendations;

    if (sourceFilter && rec?.source !== sourceFilter) continue;
    if (priorityFilter && rec?.priority !== priorityFilter) continue;

    events.push({
      id: ra.id,
      event_type: "recommendation_action",
      action: ra.action,
      timestamp: ra.created_at,
      title: rec?.title ?? null,
      hostname: rec?.hostname ?? null,
      machine_id: rec?.machine_id ?? null,
      gpu_name: rec?.gpu_name ?? null,
      source: rec?.source ?? null,
      category: rec?.category ?? null,
      priority: rec?.priority ?? null,
      estimated_savings_pct: rec?.estimated_savings_pct ?? null,
      metadata: ra.metadata ?? {},
    });
  }

  for (const cmd of cmdEvents) {
    const rec = cmd.optimization_recommendations;

    if (sourceFilter && rec?.source !== sourceFilter) continue;
    if (priorityFilter && rec?.priority !== priorityFilter) continue;

    events.push({
      id: cmd.id,
      event_type: "agent_command",
      action: cmd.status === "applied" ? "command_applied" : "command_failed",
      timestamp: cmd.completed_at,
      title: rec?.title ?? cmd.command_type,
      hostname: rec?.hostname ?? cmd.machine_id,
      machine_id: cmd.machine_id,
      gpu_name: rec?.gpu_name ?? null,
      source: rec?.source ?? null,
      category: rec?.category ?? null,
      priority: rec?.priority ?? null,
      estimated_savings_pct: rec?.estimated_savings_pct ?? null,
      metadata: cmd.result ?? {},
    });
  }

  events.sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  );

  const trimmed = events.slice(0, limit);
  const total = (recCount ?? 0) + cmdEvents.length;

  return NextResponse.json({
    events: trimmed,
    total,
    has_more: offset + limit < total,
  });
}
