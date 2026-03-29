// GET/POST/PATCH/DELETE /api/webhooks
// CRUD for user-configured event webhooks with HMAC signing.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

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

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("webhooks")
    .select(
      "id, url, description, event_types, is_active, created_at, updated_at"
    )
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ webhooks: data ?? [] });
}

// POST — create a webhook
export async function POST(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { url, description, event_types, team_id } = body;

  if (!url || typeof url !== "string") {
    return NextResponse.json({ error: "url is required" }, { status: 400 });
  }

  // Validate URL
  try {
    const parsed = new URL(url);
    if (!["https:", "http:"].includes(parsed.protocol)) {
      throw new Error("Invalid protocol");
    }
  } catch {
    return NextResponse.json(
      { error: "url must be a valid HTTP(S) URL" },
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
    return NextResponse.json({ error: error.message }, { status: 500 });
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
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}

// DELETE — delete a webhook
export async function DELETE(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
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
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ deleted: true });
}
