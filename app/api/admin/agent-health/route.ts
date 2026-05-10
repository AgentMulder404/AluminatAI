import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "").split(",").map((e) => e.trim().toLowerCase());

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user } } = await cookieClient.auth.getUser();
  if (!user || !ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const status = req.nextUrl.searchParams.get("status");
  const cluster = req.nextUrl.searchParams.get("cluster_tag");

  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("agent_heartbeats")
    .select("*")
    .order("last_seen", { ascending: false });

  if (cluster) {
    query = query.eq("cluster_tag", cluster);
  }

  const { data: agents, error } = await query;
  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const now = Date.now();
  type AgentRow = Record<string, unknown> & { status: string };
  const enriched: AgentRow[] = (agents ?? []).map((a: Record<string, unknown>) => {
    const lastSeen = new Date(a.last_seen as string).getTime();
    const offlineThresholdMs = 10 * 60 * 1000;
    const agentStatus = now - lastSeen > offlineThresholdMs ? "offline" : "online";
    return { ...a, status: agentStatus } as AgentRow;
  });

  const filtered = status
    ? enriched.filter((a) => a.status === status)
    : enriched;

  const summary = {
    total: enriched.length,
    online: enriched.filter((a) => a.status === "online").length,
    offline: enriched.filter((a) => a.status === "offline").length,
    versions: Object.entries(
      enriched.reduce((acc: Record<string, number>, a) => {
        const v = (a.agent_version as string) || "unknown";
        acc[v] = (acc[v] || 0) + 1;
        return acc;
      }, {})
    ).map(([version, count]) => ({ version, count })),
  };

  return NextResponse.json({ summary, agents: filtered });
}
