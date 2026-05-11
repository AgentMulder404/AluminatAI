// GET /api/organizations/[orgId]/members — list org members
// POST /api/organizations/[orgId]/members — invite a member

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requireOrgRole } from "@/lib/org-auth";
import { logAudit } from "@/lib/audit";
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

  const rl = await rateLimit(`org-members:${user.id}`, 60);
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
  const { data: members, error } = await supabase
    .from("org_members")
    .select("user_id, role, joined_at")
    .eq("org_id", orgId)
    .order("joined_at", { ascending: true });

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ members: members ?? [] });
}

export async function POST(req: NextRequest, { params }: RouteParams) {
  const { orgId } = await params;
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`org-members:${user.id}`, 60);
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

  let body: { user_id: string; role?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!body.user_id) {
    return NextResponse.json({ error: "user_id required" }, { status: 400 });
  }

  const role = body.role === "admin" ? "admin" : "member";

  const supabase = createSupabaseServerClient();
  const { error } = await supabase.from("org_members").insert({
    org_id: orgId,
    user_id: body.user_id,
    role,
  });

  if (error) {
    if (error.message.includes("duplicate")) {
      return NextResponse.json({ error: "User already a member" }, { status: 409 });
    }
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "org.member.add",
    resourceType: "organization",
    resourceId: orgId,
    metadata: { target_user_id: body.user_id, role },
  });

  return NextResponse.json({ added: true }, { status: 201 });
}
