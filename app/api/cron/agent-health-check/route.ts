import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { dispatchNotification } from "@/lib/notifications";
import { verifyCronSecret } from "@/lib/auth-helpers";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();
  const now = new Date();
  const offlineThreshold = new Date(now.getTime() - 10 * 60 * 1000).toISOString();

  // Fetch all agents
  const { data: agents, error: agentsErr } = await supabase
    .from("agent_heartbeats")
    .select("*");

  if (agentsErr || !agents) {
    return NextResponse.json({ error: agentsErr?.message ?? "No agents" }, { status: 500 });
  }

  const alerts: Array<Record<string, unknown>> = [];
  const resolvedIds: string[] = [];

  // Detect the most common version for drift detection
  const versionCounts: Record<string, number> = {};
  for (const a of agents) {
    const v = a.agent_version ?? "unknown";
    versionCounts[v] = (versionCounts[v] || 0) + 1;
  }
  const majorityVersion = Object.entries(versionCounts).sort((a, b) => b[1] - a[1])[0]?.[0];

  // Detect the most common config hash for drift detection
  const configCounts: Record<string, number> = {};
  for (const a of agents) {
    const c = a.config_hash ?? "";
    if (c) configCounts[c] = (configCounts[c] || 0) + 1;
  }
  const majorityConfig = Object.entries(configCounts).sort((a, b) => b[1] - a[1])[0]?.[0];

  for (const agent of agents) {
    const lastSeen = new Date(agent.last_seen).getTime();
    const isOffline = lastSeen < new Date(offlineThreshold).getTime();
    const machineId = agent.machine_id ?? agent.hostname;

    // Offline detection
    if (isOffline) {
      alerts.push({
        user_id: agent.user_id,
        machine_id: machineId,
        hostname: agent.hostname,
        alert_type: "agent_offline",
        severity: "critical",
        message: `Agent on ${agent.hostname} has been offline since ${agent.last_seen}`,
      });
    }

    // Version mismatch
    if (majorityVersion && agent.agent_version && agent.agent_version !== majorityVersion) {
      alerts.push({
        user_id: agent.user_id,
        machine_id: machineId,
        hostname: agent.hostname,
        alert_type: "version_mismatch",
        severity: "warning",
        message: `Agent on ${agent.hostname} is running v${agent.agent_version} (fleet majority: v${majorityVersion})`,
      });
    }

    // Config drift
    if (majorityConfig && agent.config_hash && agent.config_hash !== majorityConfig) {
      alerts.push({
        user_id: agent.user_id,
        machine_id: machineId,
        hostname: agent.hostname,
        alert_type: "config_drift",
        severity: "warning",
        message: `Agent on ${agent.hostname} has config hash ${agent.config_hash} (fleet majority: ${majorityConfig})`,
      });
    }

    // High error rate
    if ((agent.error_count_last_hour ?? 0) >= 50) {
      alerts.push({
        user_id: agent.user_id,
        machine_id: machineId,
        hostname: agent.hostname,
        alert_type: "high_error_rate",
        severity: "critical",
        message: `Agent on ${agent.hostname} has ${agent.error_count_last_hour} errors in the last hour`,
      });
    }
  }

  // ── Swarm leader health ─────────────────────────────────────────────────────
  const { data: leases } = await supabase
    .from("swarm_leader_leases")
    .select("user_id, cluster_tag, machine_id, expires_at");

  const graceMs = 5 * 60 * 1000;
  for (const lease of leases ?? []) {
    const expiresAt = new Date(lease.expires_at).getTime();
    if (expiresAt + graceMs < now.getTime()) {
      alerts.push({
        user_id: lease.user_id,
        machine_id: lease.machine_id,
        hostname: `cluster:${lease.cluster_tag || "(default)"}`,
        alert_type: "leader_lease_expired",
        severity: "critical",
        message: `Swarm leader lease for cluster "${lease.cluster_tag || "(default)"}" expired at ${lease.expires_at}. No agent has taken over.`,
      });
    }
  }

  // ── Command queue health ───────────────────────────────────────────────────
  const stuckThresholdMs = 30 * 60 * 1000;
  const stuckCutoff = new Date(now.getTime() - stuckThresholdMs).toISOString();

  const { data: stuckCommands } = await supabase
    .from("agent_commands")
    .select("user_id, machine_id, status, created_at")
    .in("status", ["pending", "dispatched"])
    .lt("created_at", stuckCutoff);

  const stuckByMachine = new Map<string, { user_id: string; machine_id: string; count: number }>();
  for (const cmd of stuckCommands ?? []) {
    const key = `${cmd.user_id}:${cmd.machine_id}`;
    const existing = stuckByMachine.get(key);
    if (existing) {
      existing.count++;
    } else {
      stuckByMachine.set(key, { user_id: cmd.user_id, machine_id: cmd.machine_id, count: 1 });
    }
  }

  for (const { user_id, machine_id, count } of stuckByMachine.values()) {
    alerts.push({
      user_id,
      machine_id,
      hostname: machine_id,
      alert_type: "command_queue_stuck",
      severity: count >= 5 ? "critical" : "warning",
      message: `${count} command(s) stuck in pending/dispatched for >30 min on ${machine_id}`,
    });
  }

  // ── Recommendation backlog ─────────────────────────────────────────────────
  const oneDayAgo = new Date(now.getTime() - 86_400_000).toISOString();

  const { data: pendingRecs } = await supabase
    .from("optimization_recommendations")
    .select("user_id, machine_id, created_at")
    .eq("status", "pending")
    .lt("created_at", oneDayAgo);

  const backlogByUser = new Map<string, { user_id: string; count: number; oldest: string }>();
  for (const rec of pendingRecs ?? []) {
    const entry = backlogByUser.get(rec.user_id);
    if (entry) {
      entry.count++;
      if (rec.created_at < entry.oldest) entry.oldest = rec.created_at;
    } else {
      backlogByUser.set(rec.user_id, { user_id: rec.user_id, count: 1, oldest: rec.created_at });
    }
  }

  for (const { user_id, count, oldest } of backlogByUser.values()) {
    if (count >= 10) {
      alerts.push({
        user_id,
        machine_id: "all",
        hostname: "recommendation-backlog",
        alert_type: "recommendation_backlog",
        severity: count >= 25 ? "critical" : "warning",
        message: `${count} recommendations pending for >24h (oldest: ${oldest}). Review or auto-expire them.`,
      });
    }
  }

  // Auto-resolve alerts where the condition has cleared
  const { data: activeAlerts } = await supabase
    .from("agent_health_alerts")
    .select("id, machine_id, alert_type")
    .eq("resolved", false);

  if (activeAlerts) {
    const newAlertKeys = new Set(
      alerts.map((a) => `${a.machine_id}:${a.alert_type}`)
    );
    for (const existing of activeAlerts) {
      const key = `${existing.machine_id}:${existing.alert_type}`;
      if (!newAlertKeys.has(key)) {
        resolvedIds.push(existing.id);
      }
    }
  }

  // Resolve cleared alerts
  if (resolvedIds.length > 0) {
    await supabase
      .from("agent_health_alerts")
      .update({ resolved: true, resolved_at: now.toISOString() })
      .in("id", resolvedIds);
  }

  // Upsert new alerts (dedup by machine_id + alert_type within unresolved)
  let insertedCount = 0;
  for (const alert of alerts) {
    const { data: existing } = await supabase
      .from("agent_health_alerts")
      .select("id")
      .eq("machine_id", alert.machine_id as string)
      .eq("alert_type", alert.alert_type as string)
      .eq("resolved", false)
      .limit(1);

    if (!existing || existing.length === 0) {
      const { error: insertErr } = await supabase.from("agent_health_alerts").insert(alert);
      if (!insertErr) {
        insertedCount++;
        if (alert.severity === "critical") {
          const notifType = (alert.alert_type === "agent_offline" || alert.alert_type === "high_error_rate")
            ? "agent_offline" as const
            : "system" as const;
          void dispatchNotification(
            alert.user_id as string,
            notifType,
            alert.alert_type as string,
            alert.message as string,
            { metadata: { machine_id: alert.machine_id, alert_type: alert.alert_type } }
          );
        }
      }
    }
  }

  return NextResponse.json({
    ok: true,
    agents_checked: agents.length,
    alerts_created: insertedCount,
    alerts_resolved: resolvedIds.length,
  });
}
