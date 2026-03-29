// Webhook dispatcher with HMAC-SHA256 signing and delivery tracking.
// Called from cron jobs (budget-alerts, waste-detect) when events fire.

import { createSupabaseServerClient } from "@/lib/supabase-client";

interface WebhookRow {
  id: string;
  url: string;
  secret: string;
  event_types: string[];
}

/**
 * Sign a payload with HMAC-SHA256 using the webhook's secret.
 * Returns hex-encoded signature.
 */
async function signPayload(
  secret: string,
  body: string
): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(body));
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Deliver a single webhook and record the result.
 */
async function deliverWebhook(
  webhook: WebhookRow,
  eventType: string,
  payload: Record<string, unknown>
): Promise<boolean> {
  const supabase = createSupabaseServerClient();
  const body = JSON.stringify({
    event: eventType,
    timestamp: new Date().toISOString(),
    data: payload,
  });

  const signature = await signPayload(webhook.secret, body);
  const startMs = Date.now();
  let responseStatus = 0;
  let responseBody = "";
  let success = false;

  try {
    const res = await fetch(webhook.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-AluminatAI-Signature": `sha256=${signature}`,
        "X-AluminatAI-Event": eventType,
      },
      body,
      signal: AbortSignal.timeout(10_000),
    });
    responseStatus = res.status;
    responseBody = (await res.text()).slice(0, 1000);
    success = res.ok;
  } catch (err: any) {
    responseBody = err?.message ?? "Delivery failed";
  }

  const durationMs = Date.now() - startMs;

  // Record delivery attempt
  await supabase.from("webhook_deliveries").insert({
    webhook_id: webhook.id,
    event_type: eventType,
    payload,
    response_status: responseStatus || null,
    response_body: responseBody || null,
    duration_ms: durationMs,
    success,
  });

  return success;
}

/**
 * Dispatch an event to all matching active webhooks for a user.
 * Returns count of successful deliveries.
 */
export async function dispatchWebhook(
  eventType: string,
  userId: string,
  payload: Record<string, unknown>
): Promise<number> {
  const supabase = createSupabaseServerClient();

  const { data: webhooks, error } = await supabase
    .from("webhooks")
    .select("id, url, secret, event_types")
    .eq("user_id", userId)
    .eq("is_active", true);

  if (error || !webhooks || webhooks.length === 0) {
    return 0;
  }

  // Filter to webhooks subscribed to this event type
  const matching = webhooks.filter(
    (w: WebhookRow) =>
      w.event_types.length === 0 || w.event_types.includes(eventType)
  );

  let successCount = 0;
  for (const webhook of matching) {
    const ok = await deliverWebhook(webhook, eventType, payload);
    if (ok) successCount++;
  }

  return successCount;
}

/**
 * Dispatch a team-scoped event — finds all webhooks for the team's owner
 * and any team-scoped webhooks.
 */
export async function dispatchTeamWebhook(
  eventType: string,
  teamId: string,
  payload: Record<string, unknown>
): Promise<number> {
  const supabase = createSupabaseServerClient();

  const { data: webhooks, error } = await supabase
    .from("webhooks")
    .select("id, url, secret, event_types")
    .eq("team_id", teamId)
    .eq("is_active", true);

  if (error || !webhooks || webhooks.length === 0) {
    return 0;
  }

  const matching = webhooks.filter(
    (w: WebhookRow) =>
      w.event_types.length === 0 || w.event_types.includes(eventType)
  );

  let successCount = 0;
  for (const webhook of matching) {
    const ok = await deliverWebhook(webhook, eventType, payload);
    if (ok) successCount++;
  }

  return successCount;
}
