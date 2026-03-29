// GET /api/cron/waste-detect
// Schedule: 0 */6 * * * (every 6 hours)
// Scans gpu_metrics for idle/underutilized GPUs and creates waste_events.
// Generates scheduling recommendations based on usage patterns.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { dispatchWebhook } from "@/lib/webhooks";

export const runtime = "edge";

const LOW_UTIL_THRESHOLD = 10; // % — below this is considered waste
const IDLE_THRESHOLD = 1; // % — below this is considered idle
const MIN_DURATION_HOURS = 1; // minimum duration to flag

export async function GET(req: NextRequest) {
  const auth = req.headers.get("authorization") ?? "";
  const secret = process.env.CRON_SECRET;
  if (!secret || auth !== `Bearer ${secret}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Find all users with recent agent activity (heartbeat in last 24h)
  const oneDayAgo = new Date(Date.now() - 24 * 3600000).toISOString();

  const { data: activeUsers } = await supabase
    .from("agent_heartbeats")
    .select("user_id")
    .gte("last_seen", oneDayAgo);

  const userIds = [...new Set((activeUsers ?? []).map((r) => r.user_id as string))];

  let totalEventsCreated = 0;
  let totalRecsCreated = 0;

  for (const userId of userIds) {
    const kwhRate = await getUserKwhRate(userId);

    // Get last 24h of metrics grouped by gpu_uuid
    const { data: metrics } = await supabase
      .from("gpu_metrics")
      .select(
        "gpu_uuid, gpu_name, job_id, team_id, cluster_tag, " +
          "utilization_gpu_pct, power_draw_w, energy_delta_j, gpu_fraction, time"
      )
      .eq("user_id", userId)
      .gte("time", oneDayAgo)
      .order("time", { ascending: true })
      .limit(50000);

    if (!metrics || metrics.length === 0) continue;

    // Group metrics by gpu_uuid
    const byGpu = new Map<string, typeof metrics>();
    for (const m of metrics) {
      const uuid = m.gpu_uuid as string;
      if (!uuid) continue;
      let arr = byGpu.get(uuid);
      if (!arr) {
        arr = [];
        byGpu.set(uuid, arr);
      }
      arr.push(m);
    }

    const wasteEvents: Array<Record<string, unknown>> = [];
    const recommendations: Array<Record<string, unknown>> = [];

    for (const [gpuUuid, gpuMetrics] of byGpu) {
      if (gpuMetrics.length < 10) continue; // not enough data

      const avgUtil =
        gpuMetrics.reduce((s, m) => s + ((m.utilization_gpu_pct as number) ?? 0), 0) /
        gpuMetrics.length;

      const totalJ = gpuMetrics.reduce(
        (s, m) => s + (((m.energy_delta_j as number) ?? 0) * ((m.gpu_fraction as number) ?? 1)),
        0
      );
      const totalKwh = totalJ / 3_600_000;

      // Duration in hours based on time range
      const firstTime = new Date(gpuMetrics[0].time as string).getTime();
      const lastTime = new Date(gpuMetrics[gpuMetrics.length - 1].time as string).getTime();
      const durationHours = (lastTime - firstTime) / 3_600_000;

      if (durationHours < MIN_DURATION_HOURS) continue;

      const gpuName = (gpuMetrics[0].gpu_name as string) ?? "Unknown GPU";
      const jobId = gpuMetrics[0].job_id as string | null;
      const teamId = gpuMetrics[0].team_id as string | null;
      const clusterTag = gpuMetrics[0].cluster_tag as string | null;
      const wasteCost = totalKwh * kwhRate;

      // Idle GPU: avg utilization < 1%
      if (avgUtil < IDLE_THRESHOLD && durationHours >= MIN_DURATION_HOURS) {
        wasteEvents.push({
          user_id: userId,
          gpu_uuid: gpuUuid,
          gpu_name: gpuName,
          job_id: jobId,
          team_id: teamId,
          cluster_tag: clusterTag,
          waste_type: "idle_gpu",
          avg_utilization_pct: Math.round(avgUtil * 100) / 100,
          duration_hours: Math.round(durationHours * 100) / 100,
          estimated_waste_usd: Math.round(wasteCost * 100) / 100,
        });
      }
      // Low utilization: avg < 10%
      else if (avgUtil < LOW_UTIL_THRESHOLD && durationHours >= 2) {
        wasteEvents.push({
          user_id: userId,
          gpu_uuid: gpuUuid,
          gpu_name: gpuName,
          job_id: jobId,
          team_id: teamId,
          cluster_tag: clusterTag,
          waste_type: "low_utilization",
          avg_utilization_pct: Math.round(avgUtil * 100) / 100,
          duration_hours: Math.round(durationHours * 100) / 100,
          estimated_waste_usd: Math.round(wasteCost * 100) / 100,
        });

        // Generate recommendation: consolidate or downsize
        recommendations.push({
          user_id: userId,
          recommendation_type: "gpu_downsize",
          title: `${gpuName} underutilized at ${Math.round(avgUtil)}%`,
          description: `GPU ${gpuUuid.slice(0, 8)} averaged ${Math.round(avgUtil)}% utilization over ${Math.round(durationHours)}h, costing $${wasteCost.toFixed(2)}. Consider a smaller GPU or consolidating workloads.`,
          estimated_savings_usd: Math.round(wasteCost * 0.5 * 100) / 100,
          estimated_savings_pct: 50,
          context: { gpu_uuid: gpuUuid, gpu_name: gpuName, avg_util: avgUtil },
        });
      }

      // Time-shift recommendation: check if most activity is during peak hours (9am-5pm UTC)
      const peakMetrics = gpuMetrics.filter((m) => {
        const hour = new Date(m.time as string).getUTCHours();
        return hour >= 9 && hour < 17;
      });
      const offPeakMetrics = gpuMetrics.filter((m) => {
        const hour = new Date(m.time as string).getUTCHours();
        return hour < 9 || hour >= 17;
      });

      if (peakMetrics.length > gpuMetrics.length * 0.7 && avgUtil > 30) {
        const peakAvgUtil =
          peakMetrics.reduce((s, m) => s + ((m.utilization_gpu_pct as number) ?? 0), 0) /
          peakMetrics.length;
        const offPeakAvgUtil = offPeakMetrics.length > 0
          ? offPeakMetrics.reduce((s, m) => s + ((m.utilization_gpu_pct as number) ?? 0), 0) /
            offPeakMetrics.length
          : 0;

        if (peakAvgUtil > offPeakAvgUtil * 2) {
          recommendations.push({
            user_id: userId,
            recommendation_type: "time_shift",
            title: `Shift ${gpuName} workloads to off-peak hours`,
            description: `Peak-hour utilization (${Math.round(peakAvgUtil)}%) is much higher than off-peak (${Math.round(offPeakAvgUtil)}%). Scheduling batch jobs after 5pm UTC could reduce carbon intensity by 15-30% in most grid zones.`,
            estimated_savings_pct: 20,
            context: { gpu_uuid: gpuUuid, peak_util: peakAvgUtil, offpeak_util: offPeakAvgUtil },
          });
        }
      }
    }

    // Insert waste events and dispatch webhooks
    if (wasteEvents.length > 0) {
      const { error } = await supabase.from("waste_events").insert(wasteEvents);
      if (!error) {
        totalEventsCreated += wasteEvents.length;
        const totalWasteUsd = wasteEvents.reduce(
          (s: number, e: any) => s + (e.estimated_waste_usd ?? 0),
          0
        );
        await dispatchWebhook("waste.detected", userId, {
          event_count: wasteEvents.length,
          total_waste_usd: Math.round(totalWasteUsd * 100) / 100,
          events: wasteEvents.slice(0, 10).map((e: any) => ({
            gpu_name: e.gpu_name,
            waste_type: e.waste_type,
            avg_utilization_pct: e.avg_utilization_pct,
            duration_hours: e.duration_hours,
            estimated_waste_usd: e.estimated_waste_usd,
          })),
        });
      }
    }

    // Insert recommendations
    if (recommendations.length > 0) {
      const { error } = await supabase.from("scheduling_recommendations").insert(recommendations);
      if (!error) totalRecsCreated += recommendations.length;
    }
  }

  return NextResponse.json({
    users_scanned: userIds.length,
    waste_events_created: totalEventsCreated,
    recommendations_created: totalRecsCreated,
  });
}
