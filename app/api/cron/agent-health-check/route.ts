import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const authHeader = req.headers.get("authorization");
  if (authHeader !== `Bearer ${process.env.CRON_SECRET}`) {
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
      await supabase.from("agent_health_alerts").insert(alert);
      insertedCount++;
    }
  }

  return NextResponse.json({
    ok: true,
    agents_checked: agents.length,
    alerts_created: insertedCount,
    alerts_resolved: resolvedIds.length,
  });
}
