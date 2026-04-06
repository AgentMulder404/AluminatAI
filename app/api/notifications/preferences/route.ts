// GET/PATCH /api/notifications/preferences — notification channel preferences

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

const VALID_CHANNELS = ["in_app", "email", "slack", "pagerduty", "opsgenie"] as const;

export async function GET() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();
  const { data } = await supabase
    .from("notification_preferences")
    .select("in_app, email, slack, pagerduty, opsgenie")
    .eq("user_id", user.id)
    .maybeSingle();

  // Return defaults if no row exists
  return NextResponse.json(
    data ?? { in_app: true, email: true, slack: true, pagerduty: false, opsgenie: false }
  );
}

export async function PATCH(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  // Only allow known channel keys with boolean values
  const updates: Record<string, boolean> = {};
  for (const ch of VALID_CHANNELS) {
    if (typeof body[ch] === "boolean") {
      updates[ch] = body[ch] as boolean;
    }
  }

  if (Object.keys(updates).length === 0) {
    return NextResponse.json({ error: "No valid fields" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  const { error } = await supabase
    .from("notification_preferences")
    .upsert(
      { user_id: user.id, ...updates, updated_at: new Date().toISOString() },
      { onConflict: "user_id" }
    );

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ updated: true });
}
