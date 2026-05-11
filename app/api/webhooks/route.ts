// GET/POST/PATCH/DELETE /api/webhooks
// CRUD for user-configured event webhooks with HMAC signing.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { checkCountLimit } from "@/lib/plans";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

const VALID_EVENT_TYPES = [
  "budget.warning",
  "budget.exceeded",
  "waste.detected",
  "agent.offline",
  "job.completed",
  "retention.purged",
];

async function authenticate() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) return null;
  return user;
}

function generateSecret(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// GET — list webhooks
export async function GET() {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`webhooks:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("webhooks")
    .select(
      "id, url, description, event_types, is_active, created_at, updated_at"
    )
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ webhooks: data ?? [] });
}

// POST — create a webhook
export async function POST(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`webhooks:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const body = await req.json();
  const { url, description, event_types, team_id } = body;

  if (!url || typeof url !== "string") {
    return NextResponse.json({ error: "url is required" }, { status: 400 });
  }

  // Plan limit check
  const supabaseCount = createSupabaseServerClient();
  const { count: webhookCount } = await supabaseCount
    .from("webhooks")
    .select("id", { count: "exact", head: true })
    .eq("user_id", user.id);
  const limitCheck = await checkCountLimit(user.id, "max_webhooks", webhookCount ?? 0);
  if (!limitCheck.allowed) {
    return NextResponse.json({ error: limitCheck.reason, limit: limitCheck.limit }, { status: 403 });
  }

  // Validate URL — HTTPS only, no private/internal addresses
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "https:") {
      throw new Error("Only HTTPS URLs are allowed");
    }
    const hostname = parsed.hostname;
    const blocked = ["127.0.0.1", "::1", "localhost", "169.254.169.254", "metadata.google.internal"];
    if (blocked.includes(hostname)) {
      throw new Error("Internal addresses are not allowed");
    }
    if (/^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)/.test(hostname)) {
      throw new Error("Private IP addresses are not allowed");
    }
  } catch (e) {
    return NextResponse.json(
      { error: e instanceof Error ? e.message : "url must be a valid HTTPS URL" },
      { status: 400 }
    );
  }

  // Validate event types if provided
  if (event_types && Array.isArray(event_types)) {
    const invalid = event_types.filter(
      (t: string) => !VALID_EVENT_TYPES.includes(t)
    );
    if (invalid.length > 0) {
      return NextResponse.json(
        { error: `Invalid event types: ${invalid.join(", ")}` },
        { status: 400 }
      );
    }
  }

  const secret = generateSecret();
  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("webhooks")
    .insert({
      user_id: user.id,
      team_id: team_id ?? null,
      url,
      secret,
      description: description ?? "",
      event_types: event_types ?? [],
    })
    .select("id, url, secret, description, event_types, is_active, created_at")
    .single();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  // Return secret only on creation — never returned again
  return NextResponse.json(data, { status: 201 });
}

// PATCH — update a webhook
export async function PATCH(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`webhooks:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const body = await req.json();
  const { id, ...updates } = body;

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const allowed = ["url", "description", "event_types", "is_active"];
  const safeUpdates: Record<string, unknown> = {
    updated_at: new Date().toISOString(),
  };
  for (const key of allowed) {
    if (updates[key] !== undefined) {
      safeUpdates[key] = updates[key];
    }
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("webhooks")
    .update(safeUpdates)
    .eq("id", id)
    .eq("user_id", user.id)
    .select("id, url, description, event_types, is_active, updated_at")
    .single();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json(data);
}

// DELETE — delete a webhook
export async function DELETE(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`webhooks:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const { id } = await req.json();

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  const { error } = await supabase
    .from("webhooks")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ deleted: true });
}
