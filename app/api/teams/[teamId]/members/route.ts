// GET/POST/DELETE /api/teams/[teamId]/members
// List members, invite new members, remove members.
// Requires admin+ role for invite/remove, viewer+ for listing.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requireRole, type TeamRole } from "@/lib/rbac";
import { logAudit } from "@/lib/audit";

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

// GET — list team members
export async function GET(
  _req: NextRequest,
  { params }: { params: { teamId: string } }
) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { teamId } = params;
  const { allowed } = await requireRole(user.id, teamId, "viewer");
  if (!allowed) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("team_members")
    .select("id, user_id, role, invited_at, accepted_at")
    .eq("team_id", teamId)
    .order("created_at", { ascending: true });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ members: data ?? [] });
}

// POST — invite a member by email
export async function POST(
  req: NextRequest,
  { params }: { params: { teamId: string } }
) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { teamId } = params;
  const { allowed } = await requireRole(user.id, teamId, "admin");
  if (!allowed) {
    return NextResponse.json(
      { error: "Admin role required to invite members" },
      { status: 403 }
    );
  }

  const body = await req.json();
  const { email, role } = body as { email?: string; role?: TeamRole };

  if (!email || typeof email !== "string") {
    return NextResponse.json({ error: "email is required" }, { status: 400 });
  }

  const memberRole: TeamRole = role && ["admin", "viewer", "billing"].includes(role)
    ? role
    : "viewer";

  // Cannot invite as owner
  if (role === "owner") {
    return NextResponse.json(
      { error: "Cannot invite as owner" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  // Check team member limit
  const { data: team } = await supabase
    .from("teams")
    .select("max_members")
    .eq("id", teamId)
    .single();

  const { count } = await supabase
    .from("team_members")
    .select("id", { count: "exact", head: true })
    .eq("team_id", teamId);

  if (team && count != null && count >= team.max_members) {
    return NextResponse.json(
      { error: `Team is at max capacity (${team.max_members} members)` },
      { status: 400 }
    );
  }

  // Resolve user by email
  const { data: targetUser } = await supabase
    .from("users")
    .select("id")
    .eq("email", email.toLowerCase())
    .maybeSingle();

  if (!targetUser) {
    return NextResponse.json(
      { error: "No user found with that email" },
      { status: 404 }
    );
  }

  // Check if already a member
  const { data: existing } = await supabase
    .from("team_members")
    .select("id")
    .eq("team_id", teamId)
    .eq("user_id", targetUser.id)
    .maybeSingle();

  if (existing) {
    return NextResponse.json(
      { error: "User is already a member of this team" },
      { status: 409 }
    );
  }

  const { data: member, error } = await supabase
    .from("team_members")
    .insert({
      team_id: teamId,
      user_id: targetUser.id,
      role: memberRole,
      invited_by: user.id,
      accepted_at: new Date().toISOString(), // auto-accept for now
    })
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "team.member.add",
    resourceType: "team",
    resourceId: teamId,
    metadata: { target_user_id: targetUser.id, role: memberRole },
  });

  return NextResponse.json(member, { status: 201 });
}

// DELETE — remove a member
export async function DELETE(
  req: NextRequest,
  { params }: { params: { teamId: string } }
) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { teamId } = params;
  const { allowed } = await requireRole(user.id, teamId, "admin");
  if (!allowed) {
    return NextResponse.json(
      { error: "Admin role required to remove members" },
      { status: 403 }
    );
  }

  const body = await req.json();
  const { user_id } = body;

  if (!user_id) {
    return NextResponse.json({ error: "user_id is required" }, { status: 400 });
  }

  // Cannot remove the team owner
  const supabase = createSupabaseServerClient();

  const { data: targetMember } = await supabase
    .from("team_members")
    .select("role")
    .eq("team_id", teamId)
    .eq("user_id", user_id)
    .maybeSingle();

  if (!targetMember) {
    return NextResponse.json({ error: "Member not found" }, { status: 404 });
  }

  if (targetMember.role === "owner") {
    return NextResponse.json(
      { error: "Cannot remove the team owner" },
      { status: 400 }
    );
  }

  const { error } = await supabase
    .from("team_members")
    .delete()
    .eq("team_id", teamId)
    .eq("user_id", user_id);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "team.member.remove",
    resourceType: "team",
    resourceId: teamId,
    metadata: { target_user_id: user_id },
  });

  return NextResponse.json({ removed: true });
}
