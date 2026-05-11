// GET /api/organizations — list user's orgs
// POST /api/organizations — create a new org

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserOrgs } from "@/lib/org-auth";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

export const runtime = "edge";

export async function GET() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`orgs:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const orgs = await getUserOrgs(user.id);
  return NextResponse.json({ organizations: orgs });
}

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`orgs:${user.id}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  let body: { name: string; slug?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  if (!body.name || body.name.trim().length < 2) {
    return NextResponse.json({ error: "Name must be at least 2 characters" }, { status: 400 });
  }

  // Generate slug from name if not provided
  const slug =
    body.slug?.toLowerCase().replace(/[^a-z0-9-]/g, "-").replace(/-+/g, "-") ??
    body.name.toLowerCase().replace(/[^a-z0-9]/g, "-").replace(/-+/g, "-");

  const supabase = createSupabaseServerClient();

  // Create org
  const { data: org, error: orgErr } = await supabase
    .from("organizations")
    .insert({
      name: body.name.trim(),
      slug,
      owner_id: user.id,
    })
    .select("id, name, slug")
    .single();

  if (orgErr) {
    if (orgErr.message.includes("duplicate")) {
      return NextResponse.json({ error: "Slug already taken" }, { status: 409 });
    }
    return NextResponse.json({ error: orgErr.message }, { status: 500 });
  }

  // Add creator as owner member
  await supabase.from("org_members").insert({
    org_id: org.id,
    user_id: user.id,
    role: "owner",
  });

  return NextResponse.json(org, { status: 201 });
}
