import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { logAudit } from "@/lib/audit";

export const runtime = "edge";

// PATCH /api/user/profile
// Accepts: { benchmark_opt_in?: boolean }
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

  const updates: Record<string, unknown> = {};

  if (typeof body.benchmark_opt_in === "boolean") {
    updates.benchmark_opt_in = body.benchmark_opt_in;
  }

  if (Object.keys(updates).length === 0) {
    return NextResponse.json({ error: "No valid fields to update" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();
  const { error } = await supabase
    .from("users")
    .update(updates)
    .eq("id", user.id);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}

// GET /api/user/profile — fetch current user profile fields
// API key is returned MASKED — use POST with action:"reveal_key" for full key
export async function GET(_req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase
    .from("users")
    .select("benchmark_opt_in, api_key, plan, plan_period_end, plan_cancel_at_period_end")
    .eq("id", user.id)
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  // Mask the API key — never return full key in GET
  const masked = data?.api_key
    ? data.api_key.slice(0, 9) + "••••••••••••••••" + data.api_key.slice(-4)
    : null;

  return NextResponse.json({
    benchmark_opt_in: data?.benchmark_opt_in ?? false,
    api_key_masked: masked,
    plan: data?.plan ?? "free",
    plan_period_end: data?.plan_period_end ?? null,
    plan_cancel_at_period_end: data?.plan_cancel_at_period_end ?? false,
  });
}

// POST /api/user/profile — actions (reveal_key)
export async function POST(req: NextRequest) {
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

  if (body.action === "reveal_key") {
    const supabase = createSupabaseServerClient();
    const { data, error } = await supabase
      .from("users")
      .select("api_key")
      .eq("id", user.id)
      .single();

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    void logAudit({
      userId: user.id,
      action: "api_key.reveal",
      resourceType: "api_key",
      ip: req.headers.get("x-forwarded-for")?.split(",")[0]?.trim(),
      userAgent: req.headers.get("user-agent") ?? undefined,
    });

    return NextResponse.json({ api_key: data?.api_key ?? null });
  }

  return NextResponse.json({ error: "Unknown action" }, { status: 400 });
}
