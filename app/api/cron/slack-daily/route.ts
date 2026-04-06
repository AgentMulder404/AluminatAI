// GET /api/cron/slack-daily
// Schedule: 0 9 * * * (daily at 9 AM UTC)
// Sends daily GPU cost/energy summary to configured Slack channels.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";
import { verifyCronSecret } from "@/lib/auth-helpers";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Get all active Slack installations with a channel configured
  const { data: installs } = await supabase
    .from("slack_installations")
    .select("user_id, bot_token, channel_id, slack_team_name")
    .eq("is_active", true)
    .not("channel_id", "is", null);

  if (!installs || installs.length === 0) {
    return NextResponse.json({ sent: 0, message: "No Slack channels configured" });
  }

  const yesterday = new Date();
  yesterday.setUTCDate(yesterday.getUTCDate() - 1);
  yesterday.setUTCHours(0, 0, 0, 0);
  const today = new Date();
  today.setUTCHours(0, 0, 0, 0);

  let sent = 0;

  for (const install of installs) {
    // Get yesterday's metrics for this user
    const { data: metrics } = await supabase
      .from("gpu_metrics")
      .select("energy_delta_j, gpu_fraction, carbon_g_per_kwh, gpu_name")
      .eq("user_id", install.user_id)
      .gte("time", yesterday.toISOString())
      .lt("time", today.toISOString());

    if (!metrics || metrics.length === 0) continue;

    let totalJ = 0;
    let totalCo2eG: number | null = null;
    const gpuNames = new Set<string>();

    for (const m of metrics) {
      const frac = (m.gpu_fraction ?? 1) as number;
      const energyJ = ((m.energy_delta_j ?? 0) as number) * frac;
      totalJ += energyJ;

      if (m.carbon_g_per_kwh != null) {
        totalCo2eG = (totalCo2eG ?? 0) + (energyJ / 3_600_000) * (m.carbon_g_per_kwh as number);
      }
      if (m.gpu_name) gpuNames.add(m.gpu_name as string);
    }

    const kwh = totalJ / 3_600_000;
    const rate = await getUserKwhRate(install.user_id);
    const cost = kwh * rate;

    // Get waste events from yesterday
    const { count: wasteCount } = await supabase
      .from("waste_events")
      .select("id", { count: "exact", head: true })
      .eq("user_id", install.user_id)
      .gte("detected_at", yesterday.toISOString())
      .lt("detected_at", today.toISOString());

    // Build Slack message
    const dateStr = yesterday.toISOString().slice(0, 10);
    let text = `*AluminatAI Daily Summary — ${dateStr}*\n\n`;
    text += `Energy: *${kwh.toFixed(3)} kWh*\n`;
    text += `Cost: *$${cost.toFixed(4)}*\n`;
    if (totalCo2eG != null) {
      const co2Display =
        totalCo2eG < 1000
          ? `${totalCo2eG.toFixed(1)}g`
          : `${(totalCo2eG / 1000).toFixed(2)}kg`;
      text += `CO₂e: *${co2Display}*\n`;
    }
    text += `GPUs: ${gpuNames.size > 0 ? [...gpuNames].join(", ") : "—"}\n`;
    text += `Samples: ${metrics.length.toLocaleString()}\n`;

    if (wasteCount && wasteCount > 0) {
      text += `\n⚠️ ${wasteCount} new waste event${wasteCount > 1 ? "s" : ""} detected\n`;
    }

    text += `\n<https://www.aluminatai.com/dashboard|View Dashboard →>`;

    // Send via Slack API
    try {
      const res = await fetch("https://slack.com/api/chat.postMessage", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${install.bot_token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          channel: install.channel_id,
          text,
          unfurl_links: false,
        }),
      });

      const result = await res.json();
      if (result.ok) sent++;
    } catch {
      // Best-effort; don't fail the entire cron
    }
  }

  return NextResponse.json({
    sent,
    total_installs: installs.length,
    date: yesterday.toISOString().slice(0, 10),
  });
}
