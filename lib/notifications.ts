// Notification dispatch library for budget alerts and waste detection.
// Supports Slack webhooks and email (via Resend).

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
