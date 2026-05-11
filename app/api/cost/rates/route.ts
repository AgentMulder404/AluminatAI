// GET/POST/PATCH/DELETE /api/cost/rates
// CRUD for user electricity rates and cloud GPU pricing.
// Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

async function authenticate(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) return null;
  return user;
}

export async function GET(req: NextRequest) {
  const user = await authenticate(req);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`cost-rates:${user.id}`, 30);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const supabase = createSupabaseServerClient();
  const { data, error } = await supabase
    .from("cost_rates")
    .select("id, rate_type, rate_usd, gpu_model, provider, label, is_default, effective_from, created_at")
    .eq("user_id", user.id)
    .order("created_at", { ascending: false });

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  // Find default electricity rate for convenience
  const defaultRate = data?.find(
    (r) => r.rate_type === "electricity" && r.is_default
  );

  return NextResponse.json({
    rates: data ?? [],
    default_kwh_rate: defaultRate ? Number(defaultRate.rate_usd) : 0.12,
  });
}

export async function POST(req: NextRequest) {
  const user = await authenticate(req);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`cost-rates:${user.id}`, 30);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const body = await req.json();
  const { rate_type, rate_usd, gpu_model, provider, label, is_default } = body;

  if (!rate_type || rate_usd == null) {
    return NextResponse.json(
      { error: "rate_type and rate_usd are required" },
      { status: 400 }
    );
  }

  if (!["electricity", "cloud_gpu"].includes(rate_type)) {
    return NextResponse.json(
      { error: "rate_type must be 'electricity' or 'cloud_gpu'" },
      { status: 400 }
    );
  }

  if (typeof rate_usd !== "number" || rate_usd < 0) {
    return NextResponse.json(
      { error: "rate_usd must be a non-negative number" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  // If setting as default, clear any existing default for this rate_type
  if (is_default) {
    await supabase
      .from("cost_rates")
      .update({ is_default: false })
      .eq("user_id", user.id)
      .eq("rate_type", rate_type)
      .eq("is_default", true);
  }

  const { data, error } = await supabase
    .from("cost_rates")
    .insert({
      user_id: user.id,
      rate_type,
      rate_usd,
      gpu_model: gpu_model ?? null,
      provider: provider ?? null,
      label: label ?? null,
      is_default: is_default ?? false,
    })
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json(data, { status: 201 });
}

export async function PATCH(req: NextRequest) {
  const user = await authenticate(req);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { id, rate_usd, label, is_default } = body;

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  // If promoting to default, demote existing default first
  if (is_default) {
    // Look up the rate to get its type
    const { data: existing } = await supabase
      .from("cost_rates")
      .select("rate_type")
      .eq("id", id)
      .eq("user_id", user.id)
      .single();

    if (existing) {
      await supabase
        .from("cost_rates")
        .update({ is_default: false })
        .eq("user_id", user.id)
        .eq("rate_type", existing.rate_type)
        .eq("is_default", true);
    }
  }

  const updates: Record<string, unknown> = {};
  if (rate_usd != null) updates.rate_usd = rate_usd;
  if (label != null) updates.label = label;
  if (is_default != null) updates.is_default = is_default;

  const { data, error } = await supabase
    .from("cost_rates")
    .update(updates)
    .eq("id", id)
    .eq("user_id", user.id)
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json(data);
}

export async function DELETE(req: NextRequest) {
  const user = await authenticate(req);
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { id } = await req.json();

  if (!id) {
    return NextResponse.json({ error: "id is required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  const { error } = await supabase
    .from("cost_rates")
    .delete()
    .eq("id", id)
    .eq("user_id", user.id);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ deleted: true });
}
