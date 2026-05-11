// PATCH/DELETE /api/organizations/[orgId]/members/[userId]

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requireOrgRole } from "@/lib/org-auth";
import { logAudit } from "@/lib/audit";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

interface RouteParams {
  params: Promise<{ orgId: string; userId: string }>;
}

export async function PATCH(req: NextRequest, { params }: RouteParams) {
  const { orgId, userId: targetUserId } = await params;
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`org-member:${user.id}`, 60);
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

  let body: { role: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!["admin", "member"].includes(body.role)) {
    return NextResponse.json({ error: "Invalid role" }, { status: 400 });
  }

  // Cannot change owner role
  const supabase = createSupabaseServerClient();
  const { data: existing } = await supabase
    .from("org_members")
    .select("role")
    .eq("org_id", orgId)
    .eq("user_id", targetUserId)
    .single();

  if (existing?.role === "owner") {
    return NextResponse.json({ error: "Cannot change owner role" }, { status: 403 });
  }

  const { error } = await supabase
    .from("org_members")
    .update({ role: body.role })
    .eq("org_id", orgId)
    .eq("user_id", targetUserId);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "org.member.role_change",
    resourceType: "organization",
    resourceId: orgId,
    metadata: { target_user_id: targetUserId, old_role: existing?.role, new_role: body.role },
  });

  return NextResponse.json({ updated: true });
}

export async function DELETE(req: NextRequest, { params }: RouteParams) {
  const { orgId, userId: targetUserId } = await params;
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`org-member:${user.id}`, 60);
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

  // Cannot remove owner
  const supabase = createSupabaseServerClient();
  const { data: existing } = await supabase
    .from("org_members")
    .select("role")
    .eq("org_id", orgId)
    .eq("user_id", targetUserId)
    .single();

  if (existing?.role === "owner") {
    return NextResponse.json({ error: "Cannot remove org owner" }, { status: 403 });
  }

  const { error } = await supabase
    .from("org_members")
    .delete()
    .eq("org_id", orgId)
    .eq("user_id", targetUserId);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "org.member.remove",
    resourceType: "organization",
    resourceId: orgId,
    metadata: { target_user_id: targetUserId },
  });

  return NextResponse.json({ removed: true });
}
