// GET /api/admin/sla
// SLA dashboard: uptime percentage, heartbeat gaps, ingestion latency.
// Cookie auth — admin only (NEXT_PUBLIC_ADMIN_EMAIL).

import { NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requirePlan } from "@/lib/plans";

export const runtime = "edge";

const ADMIN_EMAILS = (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

export async function GET() {
  // Admin auth
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  // SLA dashboard requires Enterprise plan
  const planCheck = await requirePlan(user.id, "sla_dashboard");
  if (!planCheck.allowed) {
    return NextResponse.json(
      { error: planCheck.reason, upgrade_to: planCheck.upgrade_to },
      { status: 403 }
    );
  }

  const supabase = createSupabaseServerClient();
  const now = new Date();
  const thirtyDaysAgo = new Date(
    now.getTime() - 30 * 24 * 60 * 60 * 1000
  ).toISOString();
  const oneDayAgo = new Date(
    now.getTime() - 24 * 60 * 60 * 1000
  ).toISOString();

  // 1. Agent uptime — count heartbeat gaps > 10 minutes in last 30 days
  const { data: heartbeats } = await supabase
    .from("agent_heartbeats")
    .select("agent_id, last_seen_at")
    .gte("last_seen_at", thirtyDaysAgo)
    .order("last_seen_at", { ascending: true });

  let totalExpectedMinutes = 30 * 24 * 60; // 30 days in minutes
  let totalGapMinutes = 0;
  const gaps: Array<{
    agent_id: string;
    from: string;
    to: string;
    gap_minutes: number;
  }> = [];

  if (heartbeats && heartbeats.length > 1) {
    // Group by agent
    const byAgent = new Map<string, string[]>();
    for (const hb of heartbeats) {
      const list = byAgent.get(hb.agent_id) ?? [];
      list.push(hb.last_seen_at);
      byAgent.set(hb.agent_id, list);
    }

    for (const [agentId, timestamps] of byAgent) {
      for (let i = 1; i < timestamps.length; i++) {
        const prev = new Date(timestamps[i - 1]).getTime();
        const curr = new Date(timestamps[i]).getTime();
        const gapMin = (curr - prev) / 60_000;
        if (gapMin > 10) {
          totalGapMinutes += gapMin;
          gaps.push({
            agent_id: agentId,
            from: timestamps[i - 1],
            to: timestamps[i],
            gap_minutes: Math.round(gapMin),
          });
        }
      }
    }
  }

  const uptimePct =
    totalExpectedMinutes > 0
      ? Math.max(
          0,
          Math.round(
            ((totalExpectedMinutes - totalGapMinutes) / totalExpectedMinutes) *
              10000
          ) / 100
        )
      : 0;

  // 2. Ingestion latency — average time between metric timestamp and DB insert
  //    (check recent gpu_metrics for created_at vs timestamp delta)
  const { data: recentMetrics } = await supabase
    .from("gpu_metrics")
    .select("timestamp, created_at")
    .gte("created_at", oneDayAgo)
    .order("created_at", { ascending: false })
    .limit(100);

  let avgLatencyMs = 0;
  let p95LatencyMs = 0;
  if (recentMetrics && recentMetrics.length > 0) {
    const latencies = recentMetrics
      .map((m: any) => {
        const ts = new Date(m.timestamp).getTime();
        const created = new Date(m.created_at).getTime();
        return created - ts;
      })
      .filter((l: number) => l >= 0)
      .sort((a: number, b: number) => a - b);

    if (latencies.length > 0) {
      avgLatencyMs = Math.round(
        latencies.reduce((s: number, l: number) => s + l, 0) / latencies.length
      );
      p95LatencyMs = latencies[Math.floor(latencies.length * 0.95)] ?? 0;
    }
  }

  // 3. Active agents in last 24h
  const { count: activeAgents } = await supabase
    .from("agent_heartbeats")
    .select("id", { count: "exact", head: true })
    .gte("last_seen_at", oneDayAgo);

  // 4. Total metrics ingested in last 24h
  const { count: metricsIngested } = await supabase
    .from("gpu_metrics")
    .select("id", { count: "exact", head: true })
    .gte("created_at", oneDayAgo);

  return NextResponse.json({
    uptime_pct: uptimePct,
    gap_count: gaps.length,
    gaps: gaps.slice(0, 20), // top 20 gaps
    total_gap_minutes: Math.round(totalGapMinutes),
    ingestion: {
      avg_latency_ms: avgLatencyMs,
      p95_latency_ms: Math.round(p95LatencyMs),
      metrics_24h: metricsIngested ?? 0,
    },
    active_agents_24h: activeAgents ?? 0,
    period: "30d",
    checked_at: now.toISOString(),
  });
}
