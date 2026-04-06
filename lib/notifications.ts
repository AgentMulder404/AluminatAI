// Notification dispatch library for budget alerts, waste detection, and in-app notifications.
// Supports Slack webhooks, email (via Resend), PagerDuty, OpsGenie, and in-app.

import { createSupabaseServerClient } from "./supabase-client";

interface SlackMessage {
  text: string;
  blocks?: unknown[];
}

interface BudgetAlertContext {
  budget_name: string;
  alert_type: "warn" | "exceeded";
  spend_usd: number;
  limit_usd: number;
  period: string;
  scope: string;
}

interface NotifyChannel {
  type: "email" | "slack" | "pagerduty" | "opsgenie";
  target: string; // email address, Slack webhook URL, PagerDuty routing key, or OpsGenie API key
}

/**
 * Send a message to a Slack webhook URL.
 */
export async function sendSlackWebhook(
  webhookUrl: string,
  message: SlackMessage
): Promise<boolean> {
  try {
    const res = await fetch(webhookUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(message),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Send an email via Resend API.
 */
export async function sendEmail(
  to: string,
  subject: string,
  body: string
): Promise<boolean> {
  const apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) return false;

  try {
    const res = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "AluminatAI <alerts@aluminatai.com>",
        to: [to],
        subject,
        text: body,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Send a PagerDuty alert via Events API v2.
 * target = PagerDuty routing (integration) key.
 */
export async function sendPagerDutyAlert(
  routingKey: string,
  summary: string,
  severity: "critical" | "error" | "warning" | "info" = "warning",
  details?: Record<string, unknown>
): Promise<boolean> {
  try {
    const res = await fetch("https://events.pagerduty.com/v2/enqueue", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        routing_key: routingKey,
        event_action: "trigger",
        payload: {
          summary,
          severity,
          source: "AluminatAI",
          component: "gpu-cost-monitoring",
          custom_details: details,
        },
        links: [
          {
            href: "https://www.aluminatai.com/dashboard",
            text: "AluminatAI Dashboard",
          },
        ],
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Send an OpsGenie alert via Alert API.
 * target = OpsGenie API key (GenieKey).
 */
export async function sendOpsGenieAlert(
  apiKey: string,
  message: string,
  priority: "P1" | "P2" | "P3" | "P4" | "P5" = "P3",
  details?: Record<string, string>
): Promise<boolean> {
  try {
    const res = await fetch("https://api.opsgenie.com/v2/alerts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `GenieKey ${apiKey}`,
      },
      body: JSON.stringify({
        message,
        priority,
        source: "AluminatAI",
        tags: ["aluminatai", "gpu", "cost"],
        details,
      }),
    });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Dispatch a budget alert to all configured notification channels.
 */
export async function dispatchBudgetAlert(
  channels: NotifyChannel[],
  context: BudgetAlertContext
): Promise<string[]> {
  const sent: string[] = [];

  const pctUsed = Math.round((context.spend_usd / context.limit_usd) * 100);
  const emoji = context.alert_type === "exceeded" ? "🚨" : "⚠️";
  const statusText =
    context.alert_type === "exceeded" ? "EXCEEDED" : "WARNING";

  for (const channel of channels) {
    if (channel.type === "slack") {
      const text =
        `${emoji} *Budget ${statusText}*: ${context.budget_name}\n` +
        `Spend: $${context.spend_usd.toFixed(2)} / $${context.limit_usd.toFixed(2)} (${pctUsed}%)\n` +
        `Period: ${context.period} | Scope: ${context.scope}`;

      const ok = await sendSlackWebhook(channel.target, { text });
      if (ok) sent.push(`slack:${channel.target.slice(0, 30)}...`);
    }

    if (channel.type === "email") {
      const subject = `[AluminatAI] Budget ${statusText}: ${context.budget_name}`;
      const body =
        `Your budget "${context.budget_name}" has ${context.alert_type === "exceeded" ? "been exceeded" : "reached the warning threshold"}.\n\n` +
        `Current spend: $${context.spend_usd.toFixed(2)}\n` +
        `Budget limit: $${context.limit_usd.toFixed(2)}\n` +
        `Usage: ${pctUsed}%\n` +
        `Period: ${context.period}\n` +
        `Scope: ${context.scope}\n\n` +
        `View your dashboard: https://www.aluminatai.com/dashboard`;

      const ok = await sendEmail(channel.target, subject, body);
      if (ok) sent.push(`email:${channel.target}`);
    }

    if (channel.type === "pagerduty") {
      const severity = context.alert_type === "exceeded" ? "error" : "warning";
      const summary = `Budget ${statusText}: ${context.budget_name} — $${context.spend_usd.toFixed(2)}/$${context.limit_usd.toFixed(2)} (${pctUsed}%)`;
      const ok = await sendPagerDutyAlert(channel.target, summary, severity, {
        budget_name: context.budget_name,
        spend_usd: context.spend_usd,
        limit_usd: context.limit_usd,
        period: context.period,
        scope: context.scope,
      });
      if (ok) sent.push(`pagerduty:${channel.target.slice(0, 8)}...`);
    }

    if (channel.type === "opsgenie") {
      const priority = context.alert_type === "exceeded" ? "P2" : "P3";
      const message = `Budget ${statusText}: ${context.budget_name} ($${context.spend_usd.toFixed(2)}/$${context.limit_usd.toFixed(2)})`;
      const ok = await sendOpsGenieAlert(channel.target, message, priority, {
        budget_name: context.budget_name,
        spend_usd: String(context.spend_usd.toFixed(2)),
        limit_usd: String(context.limit_usd.toFixed(2)),
        period: context.period,
        scope: context.scope,
      });
      if (ok) sent.push(`opsgenie:${channel.target.slice(0, 8)}...`);
    }
  }

  return sent;
}

/**
 * Create an in-app notification for a user.
 * Fire-and-forget — errors are logged but don't throw.
 */
export async function createInAppNotification(
  userId: string,
  type: "budget_alert" | "waste_detected" | "agent_offline" | "system" | "team",
  title: string,
  message: string,
  metadata?: Record<string, unknown>
): Promise<boolean> {
  try {
    const supabase = createSupabaseServerClient();

    // Check if user has in-app notifications enabled
    const { data: prefs } = await supabase
      .from("notification_preferences")
      .select("in_app")
      .eq("user_id", userId)
      .maybeSingle();

    // Default to enabled if no preferences row exists
    if (prefs && prefs.in_app === false) return false;

    const { error } = await supabase.from("notifications").insert({
      user_id: userId,
      type,
      title,
      message,
      metadata: metadata ?? {},
    });

    return !error;
  } catch {
    return false;
  }
}

/**
 * Unified notification dispatcher.
 * Checks user's channel preferences, then dispatches to all enabled channels.
 * Use this instead of calling individual channel functions directly.
 */
export async function dispatchNotification(
  userId: string,
  type: "budget_alert" | "waste_detected" | "agent_offline" | "system" | "team",
  title: string,
  message: string,
  opts?: {
    channels?: NotifyChannel[];
    metadata?: Record<string, unknown>;
    budgetContext?: BudgetAlertContext;
  }
): Promise<{ sent: string[] }> {
  const sent: string[] = [];

  try {
    const supabase = createSupabaseServerClient();
    const { data: prefs } = await supabase
      .from("notification_preferences")
      .select("in_app, email, slack, pagerduty, opsgenie")
      .eq("user_id", userId)
      .maybeSingle();

    // Defaults: all channels enabled
    const p = {
      in_app: prefs?.in_app ?? true,
      email: prefs?.email ?? true,
      slack: prefs?.slack ?? true,
      pagerduty: prefs?.pagerduty ?? true,
      opsgenie: prefs?.opsgenie ?? true,
    };

    // In-app notification
    if (p.in_app) {
      const ok = await createInAppNotification(userId, type, title, message, opts?.metadata);
      if (ok) sent.push("in_app");
    }

    // External channels
    if (opts?.channels) {
      for (const ch of opts.channels) {
        if (ch.type === "email" && !p.email) continue;
        if (ch.type === "slack" && !p.slack) continue;
        if (ch.type === "pagerduty" && !p.pagerduty) continue;
        if (ch.type === "opsgenie" && !p.opsgenie) continue;

        if (opts.budgetContext) {
          // Budget alerts use the specialized dispatcher
          const results = await dispatchBudgetAlert([ch], opts.budgetContext);
          sent.push(...results);
        } else {
          // Generic notification to external channels
          if (ch.type === "slack") {
            const ok = await sendSlackWebhook(ch.target, { text: `*${title}*\n${message}` });
            if (ok) sent.push("slack");
          } else if (ch.type === "email") {
            const ok = await sendEmail(ch.target, `[AluminatAI] ${title}`, message);
            if (ok) sent.push("email");
          } else if (ch.type === "pagerduty") {
            const ok = await sendPagerDutyAlert(ch.target, `${title}: ${message}`);
            if (ok) sent.push("pagerduty");
          } else if (ch.type === "opsgenie") {
            const ok = await sendOpsGenieAlert(ch.target, `${title}: ${message}`);
            if (ok) sent.push("opsgenie");
          }
        }
      }
    }
  } catch {
    // Notification dispatch should never block operations
  }

  return { sent };
}
