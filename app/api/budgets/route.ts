// GET/POST/PATCH/DELETE /api/budgets
// CRUD for budget thresholds with notification channels.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { checkCountLimit } from "@/lib/plans";

export const runtime = "edge";

async function authenticate() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) return null;
  return user;
}

export async function GET() {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("budgets")
    .select(
      "id, name, scope_type, scope_value, period, limit_usd, warn_pct, " +
        "notify_channels, is_active, created_at, updated_at"
    )
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ budgets: data ?? [] });
}

export async function POST(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { name, scope_type, scope_value, period, limit_usd, warn_pct, notify_channels } = body;

  if (!name || !scope_type || !period || limit_usd == null) {
    return NextResponse.json(
      { error: "name, scope_type, period, and limit_usd are required" },
      { status: 400 }
    );
  }

  if (!["global", "team", "project", "cluster", "gpu_model"].includes(scope_type)) {
    return NextResponse.json(
      { error: "Invalid scope_type" },
      { status: 400 }
    );
  }

  if (!["daily", "weekly", "monthly"].includes(period)) {
    return NextResponse.json({ error: "Invalid period" }, { status: 400 });
  }

  if (typeof limit_usd !== "number" || limit_usd <= 0) {
    return NextResponse.json(
      { error: "limit_usd must be a positive number" },
      { status: 400 }
    );
  }

  // Plan limit check
  const supabaseCount = createSupabaseServerClient();
  const { count: budgetCount } = await supabaseCount
    .from("budgets")
    .select("id", { count: "exact", head: true })
    .eq("user_id", user.id);
  const limitCheck = await checkCountLimit(user.id, "max_budgets", budgetCount ?? 0);
  if (!limitCheck.allowed) {
    return NextResponse.json({ error: limitCheck.reason, limit: limitCheck.limit }, { status: 403 });
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("budgets")
    .insert({
      user_id: user.id,
      name,
      scope_type,
      scope_value: scope_value ?? null,
      period,
      limit_usd,
      warn_pct: warn_pct ?? 80,
      notify_channels: notify_channels ?? [],
    })
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data, { status: 201 });
}

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

  // Only allow safe fields to be updated
  const allowed = ["name", "limit_usd", "warn_pct", "notify_channels", "is_active"];
  const safeUpdates: Record<string, unknown> = { updated_at: new Date().toISOString() };
  for (const key of allowed) {
    if (updates[key] !== undefined) {
      safeUpdates[key] = updates[key];
    }
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("budgets")
    .update(safeUpdates)
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}

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
    .from("budgets")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ deleted: true });
}
