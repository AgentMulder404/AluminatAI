// POST /api/integrations/slack/events
// Handles Slack slash commands: /aluminatai cost | waste | budget | status
// Also handles Slack URL verification challenge.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";

export const runtime = "edge";

const SLACK_SIGNING_SECRET = process.env.SLACK_SIGNING_SECRET ?? "";

/**
 * Verify Slack request signature (HMAC-SHA256).
 */
async function verifySlackSignature(
  req: NextRequest,
  rawBody: string
): Promise<boolean> {
  if (!SLACK_SIGNING_SECRET) return false;

  const timestamp = req.headers.get("x-slack-request-timestamp") ?? "";
  const slackSig = req.headers.get("x-slack-signature") ?? "";

  // Reject requests older than 5 minutes
  if (Math.abs(Date.now() / 1000 - Number(timestamp)) > 300) return false;

  const baseString = `v0:${timestamp}:${rawBody}`;
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(SLACK_SIGNING_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(baseString));
  const hex =
    "v0=" +
    Array.from(new Uint8Array(sig))
      .map((b) => b.toString(16).padStart(2, "0"))
      .join("");

  return hex === slackSig;
}

/**
 * Look up the AluminatAI user associated with a Slack team.
 */
async function findUserBySlackTeam(
  slackTeamId: string
): Promise<string | null> {
  const supabase = createSupabaseServerClient();
  const { data } = await supabase
    .from("slack_installations")
    .select("user_id")
    .eq("slack_team_id", slackTeamId)
    .eq("is_active", true)
    .limit(1)
    .maybeSingle();

  return data?.user_id ?? null;
}

/**
 * Handle /aluminatai cost
 */
async function handleCost(userId: string): Promise<string> {
  const supabase = createSupabaseServerClient();
  const startOfDay = new Date();
  startOfDay.setUTCHours(0, 0, 0, 0);

  const { data } = await supabase
    .from("gpu_metrics")
    .select("energy_delta_j, gpu_fraction")
    .eq("user_id", userId)
    .gte("time", startOfDay.toISOString());

  if (!data || data.length === 0) {
    return "No GPU metrics recorded today yet.";
  }

  let totalJ = 0;
  for (const rawR of data) {
    const r = rawR as unknown as Record<string, unknown>;
    totalJ += ((r.energy_delta_j as number) ?? 0) * ((r.gpu_fraction as number) ?? 1);
  }

  const kwh = totalJ / 3_600_000;
  const rate = await getUserKwhRate(userId);
  const cost = kwh * rate;

  return (
    `*Today's GPU Cost*\n` +
    `Energy: ${kwh.toFixed(3)} kWh\n` +
    `Cost: $${cost.toFixed(4)}\n` +
    `Rate: $${rate}/kWh`
  );
}

/**
 * Handle /aluminatai waste
 */
async function handleWaste(userId: string): Promise<string> {
  const supabase = createSupabaseServerClient();

  const { data } = await supabase
    .from("waste_events")
    .select("gpu_name, waste_type, avg_utilization_pct, estimated_waste_usd, duration_hours")
    .eq("user_id", userId)
    .eq("dismissed", false)
    .order("detected_at", { ascending: false })
    .limit(5);

  if (!data || data.length === 0) {
    return "No active waste events detected. Your GPUs are running efficiently.";
  }

  const rows = data as unknown as Record<string, unknown>[];
  const totalWaste = rows.reduce(
    (s, e) => s + Number(e.estimated_waste_usd),
    0
  );

  let msg = `*${rows.length} Active Waste Event${rows.length > 1 ? "s" : ""}* — $${totalWaste.toFixed(2)} estimated waste\n\n`;

  for (const e of rows) {
    const typeLabel =
      e.waste_type === "idle_gpu" ? "Idle" : "Low utilization";
    msg += `• ${e.gpu_name}: ${typeLabel} (${e.avg_utilization_pct}% util, ${e.duration_hours}h) — $${Number(e.estimated_waste_usd as number).toFixed(2)}\n`;
  }

  return msg;
}

/**
 * Handle /aluminatai budget
 */
async function handleBudget(userId: string): Promise<string> {
  const supabase = createSupabaseServerClient();

  const { data } = await supabase
    .from("budgets")
    .select("name, scope_type, period, limit_usd, is_active")
    .eq("user_id", userId)
    .eq("is_active", true)
    .order("created_at", { ascending: false })
    .limit(10);

  if (!data || data.length === 0) {
    return "No active budgets configured. Set one up at aluminatai.com/dashboard/settings";
  }

  const budgets = data as unknown as Record<string, unknown>[];
  let msg = `*${budgets.length} Active Budget${budgets.length > 1 ? "s" : ""}*\n\n`;
  for (const b of budgets) {
    msg += `• *${b.name}*: $${b.limit_usd}/${b.period} (${b.scope_type})\n`;
  }

  return msg;
}

/**
 * Handle /aluminatai status
 */
async function handleStatus(userId: string): Promise<string> {
  const supabase = createSupabaseServerClient();
  const tenMinAgo = new Date(Date.now() - 10 * 60_000).toISOString();

  const { count: onlineAgents } = await supabase
    .from("agent_heartbeats")
    .select("id", { count: "exact", head: true })
    .eq("user_id", userId)
    .gte("last_seen_at", tenMinAgo);

  const { count: totalGpus } = await supabase
    .from("agent_heartbeats")
    .select("gpu_count")
    .eq("user_id", userId)
    .gte("last_seen_at", tenMinAgo);

  return (
    `*Agent Status*\n` +
    `Online agents: ${onlineAgents ?? 0}\n` +
    `View dashboard: https://www.aluminatai.com/dashboard`
  );
}

export async function POST(req: NextRequest) {
  const rawBody = await req.text();

  // Parse form-encoded body (Slack sends slash commands as form data)
  const params = new URLSearchParams(rawBody);

  // Handle URL verification challenge (JSON body)
  if (!params.has("command")) {
    try {
      const json = JSON.parse(rawBody);
      if (json.type === "url_verification") {
        return NextResponse.json({ challenge: json.challenge });
      }
    } catch {
      // Not JSON, continue
    }
  }

  // Verify Slack signature — fail closed if secret is not configured
  if (!SLACK_SIGNING_SECRET) {
    return NextResponse.json(
      { error: "Slack integration not configured" },
      { status: 503 }
    );
  }
  const valid = await verifySlackSignature(req, rawBody);
  if (!valid) {
    return NextResponse.json({ error: "Invalid signature" }, { status: 401 });
  }

  const command = params.get("command") ?? "";
  const text = (params.get("text") ?? "").trim().toLowerCase();
  const slackTeamId = params.get("team_id") ?? "";

  // Only handle /aluminatai command
  if (!command.includes("aluminatai")) {
    return NextResponse.json({
      response_type: "ephemeral",
      text: "Unknown command.",
    });
  }

  // Find the AluminatAI user for this Slack workspace
  const userId = await findUserBySlackTeam(slackTeamId);
  if (!userId) {
    return NextResponse.json({
      response_type: "ephemeral",
      text: "This Slack workspace isn't connected to AluminatAI. Visit aluminatai.com/dashboard/settings to connect.",
    });
  }

  let responseText: string;

  switch (text.split(" ")[0]) {
    case "cost":
      responseText = await handleCost(userId);
      break;
    case "waste":
      responseText = await handleWaste(userId);
      break;
    case "budget":
    case "budgets":
      responseText = await handleBudget(userId);
      break;
    case "status":
      responseText = await handleStatus(userId);
      break;
    default:
      responseText =
        "*AluminatAI Commands*\n" +
        "• `/aluminatai cost` — Today's GPU cost & energy\n" +
        "• `/aluminatai waste` — Active waste events\n" +
        "• `/aluminatai budget` — Budget status\n" +
        "• `/aluminatai status` — Agent connectivity";
      break;
  }

  return NextResponse.json({
    response_type: "ephemeral",
    text: responseText,
  });
}
