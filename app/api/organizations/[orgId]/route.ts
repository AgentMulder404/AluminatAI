// GET/PATCH/DELETE /api/organizations/[orgId]

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requireOrgRole } from "@/lib/org-auth";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

interface RouteParams {
  params: Promise<{ orgId: string }>;
}

export async function GET(req: NextRequest, { params }: RouteParams) {
  const { orgId } = await params;
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`org:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const { allowed } = await requireOrgRole(user.id, orgId, "member");
  if (!allowed) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const supabase = createSupabaseServerClient();
  const { data: org, error } = await supabase
    .from("organizations")
    .select("id, name, slug, plan, plan_period_end, plan_cancel_at_period_end, created_at")
    .eq("id", orgId)
    .single();

  if (error || !org) {
    return NextResponse.json({ error: "Not found" }, { status: 404 });
  }

  return NextResponse.json(org);
}

export async function PATCH(req: NextRequest, { params }: RouteParams) {
  const { orgId } = await params;
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`org:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const { allowed } = await requireOrgRole(user.id, orgId, "admin");
  if (!allowed) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  let body: { name?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const updates: Record<string, unknown> = {};
  if (body.name && body.name.trim().length >= 2) {
    updates.name = body.name.trim();
  }
  updates.updated_at = new Date().toISOString();

  const supabase = createSupabaseServerClient();
  const { error } = await supabase
    .from("organizations")
    .update(updates)
    .eq("id", orgId);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ updated: true });
}

export async function DELETE(req: NextRequest, { params }: RouteParams) {
  const { orgId } = await params;
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`org:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const { allowed } = await requireOrgRole(user.id, orgId, "owner");
  if (!allowed) {
    return NextResponse.json({ error: "Only owner can delete org" }, { status: 403 });
  }

  const supabase = createSupabaseServerClient();
  const { error } = await supabase
    .from("organizations")
    .delete()
    .eq("id", orgId);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ deleted: true });
}
