// GET/POST /api/teams
// Create and list teams. Cookie auth (dashboard session).

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getTeamScope } from "@/lib/rbac";
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

  const { teams } = await getTeamScope(user.id);
  return NextResponse.json({ teams });
}

export async function POST(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { name } = body;

  if (!name || typeof name !== "string" || name.trim().length < 2) {
    return NextResponse.json(
      { error: "name is required (min 2 characters)" },
      { status: 400 }
    );
  }

  // Plan limit check
  const { teams: existingTeams } = await getTeamScope(user.id);
  const ownedCount = existingTeams.filter((t: { owner_id: string }) => t.owner_id === user.id).length;
  const limitCheck = await checkCountLimit(user.id, "max_teams", ownedCount);
  if (!limitCheck.allowed) {
    return NextResponse.json({ error: limitCheck.reason, limit: limitCheck.limit }, { status: 403 });
  }

  // Generate slug from name
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 48);

  const supabase = createSupabaseServerClient();

  // Check slug uniqueness
  const { data: existing } = await supabase
    .from("teams")
    .select("id")
    .eq("slug", slug)
    .maybeSingle();

  if (existing) {
    return NextResponse.json(
      { error: "A team with this name already exists" },
      { status: 409 }
    );
  }

  const { data: team, error } = await supabase
    .from("teams")
    .insert({
      name: name.trim(),
      slug,
      owner_id: user.id,
    })
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(team, { status: 201 });
}
