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

  const supabase = createSupabaseServerClient();
  const tenMinAgo = new Date(Date.now() - 600_000).toISOString();
  const oneHourAgo = new Date(Date.now() - 3_600_000).toISOString();
  const oneDayAgo = new Date(Date.now() - 86_400_000).toISOString();

  // 1. Leader leases (all clusters)
  const { data: leases } = await supabase
    .from("swarm_leader_leases")
    .select("*")
    .order("acquired_at", { ascending: false });

  const now = Date.now();
  const leaderStatus = (leases ?? []).map((l) => ({
    ...l,
    is_active: new Date(l.expires_at).getTime() > now,
    expires_in_s: Math.max(0, Math.round((new Date(l.expires_at).getTime() - now) / 1000)),
  }));

  // 2. Fleet overview — active machines grouped by cluster
  const { data: heartbeats } = await supabase
    .from("agent_heartbeats")
    .select("machine_id, hostname, gpu_count, cluster_tag, last_seen_at, agent_version")
    .gte("last_seen_at", tenMinAgo);

  interface ClusterSummary {
    cluster_tag: string;
    machines: number;
    total_gpus: number;
    hostnames: string[];
  }
  const clusterMap = new Map<string, ClusterSummary>();
  for (const hb of heartbeats ?? []) {
    const tag = hb.cluster_tag || "(default)";
    const existing = clusterMap.get(tag);
    if (existing) {
      existing.machines++;
      existing.total_gpus += hb.gpu_count ?? 0;
      existing.hostnames.push(hb.hostname ?? hb.machine_id);
    } else {
      clusterMap.set(tag, {
        cluster_tag: tag,
        machines: 1,
        total_gpus: hb.gpu_count ?? 0,
        hostnames: [hb.hostname ?? hb.machine_id],
      });
    }
  }

  // 3. Recommendation stats (last 24h)
  const { data: recs } = await supabase
    .from("optimization_recommendations")
    .select("status, source, priority, estimated_savings_pct, created_at")
    .gte("created_at", oneDayAgo);

  const recsByStatus: Record<string, number> = {};
  const recsBySource: Record<string, number> = {};
  const recsByPriority: Record<string, number> = {};
  let totalSavingsPct = 0;
  let appliedCount = 0;

  for (const r of recs ?? []) {
    recsByStatus[r.status] = (recsByStatus[r.status] ?? 0) + 1;
    recsBySource[r.source] = (recsBySource[r.source] ?? 0) + 1;
    recsByPriority[r.priority] = (recsByPriority[r.priority] ?? 0) + 1;
    if (r.status === "applied") {
      totalSavingsPct += r.estimated_savings_pct ?? 0;
      appliedCount++;
    }
  }

  // Conversion funnel
  const totalRecs = (recs ?? []).length;
  const pending = recsByStatus["pending"] ?? 0;
  const approved = recsByStatus["approved"] ?? 0;
  const applied = recsByStatus["applied"] ?? 0;
  const rejected = recsByStatus["rejected"] ?? 0;
  const rolledBack = recsByStatus["rolled_back"] ?? 0;

  // 4. Command pipeline (last 24h)
  const { data: commands } = await supabase
    .from("agent_commands")
    .select("status, command_type, machine_id, created_at, dispatched_at, completed_at")
    .gte("created_at", oneDayAgo)
    .order("created_at", { ascending: false })
    .limit(200);

  const cmdsByStatus: Record<string, number> = {};
  const cmdsByType: Record<string, number> = {};
  let avgDispatchLatencyMs = 0;
  let avgExecLatencyMs = 0;
  let dispatchCount = 0;
  let execCount = 0;

  for (const c of commands ?? []) {
    cmdsByStatus[c.status] = (cmdsByStatus[c.status] ?? 0) + 1;
    cmdsByType[c.command_type] = (cmdsByType[c.command_type] ?? 0) + 1;
    if (c.dispatched_at && c.created_at) {
      const d = new Date(c.dispatched_at).getTime() - new Date(c.created_at).getTime();
      avgDispatchLatencyMs += d;
      dispatchCount++;
    }
    if (c.completed_at && c.dispatched_at) {
      const e = new Date(c.completed_at).getTime() - new Date(c.dispatched_at).getTime();
      avgExecLatencyMs += e;
      execCount++;
    }
  }

  return NextResponse.json({
    leaders: leaderStatus,
    fleet: {
      total_machines: (heartbeats ?? []).length,
      total_gpus: (heartbeats ?? []).reduce((s, h) => s + (h.gpu_count ?? 0), 0),
      clusters: [...clusterMap.values()],
    },
    recommendations: {
      total_24h: totalRecs,
      by_status: recsByStatus,
      by_source: recsBySource,
      by_priority: recsByPriority,
      avg_applied_savings_pct: appliedCount > 0 ? Math.round(totalSavingsPct / appliedCount * 10) / 10 : 0,
      funnel: { pending, approved, applied, rejected, rolled_back: rolledBack },
    },
    commands: {
      total_24h: (commands ?? []).length,
      by_status: cmdsByStatus,
      by_type: cmdsByType,
      recent: (commands ?? []).slice(0, 20).map((c) => ({
        status: c.status,
        command_type: c.command_type,
        machine_id: c.machine_id,
        created_at: c.created_at,
        completed_at: c.completed_at,
      })),
      avg_dispatch_latency_s: dispatchCount > 0 ? Math.round(avgDispatchLatencyMs / dispatchCount / 1000) : null,
      avg_exec_latency_s: execCount > 0 ? Math.round(avgExecLatencyMs / execCount / 1000) : null,
    },
  });
}
