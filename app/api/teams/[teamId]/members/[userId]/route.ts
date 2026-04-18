// PATCH /api/teams/[teamId]/members/[userId]
// Change a team member's role. Requires admin+ role.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { requireRole, isTeamOwner, type TeamRole } from "@/lib/rbac";
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

export async function PATCH(
  req: NextRequest,
  { params }: { params: Promise<{ teamId: string; userId: string }> }
) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { teamId, userId: targetUserId } = await params;

  // Only admins+ can change roles
  const { allowed, role: callerRole } = await requireRole(user.id, teamId, "admin");
  if (!allowed) {
    return NextResponse.json(
      { error: "Admin role required to change member roles" },
      { status: 403 }
    );
  }

  const body = await req.json();
  const { role: newRole } = body as { role?: TeamRole };

  if (!newRole || !["admin", "viewer", "billing"].includes(newRole)) {
    return NextResponse.json(
      { error: "role must be one of: admin, viewer, billing" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  // Check target exists
  const { data: targetMember } = await supabase
    .from("team_members")
    .select("role")
    .eq("team_id", teamId)
    .eq("user_id", targetUserId)
    .maybeSingle();

  if (!targetMember) {
    return NextResponse.json({ error: "Member not found" }, { status: 404 });
  }

  // Cannot change the owner's role (must transfer ownership instead)
  if (targetMember.role === "owner") {
    return NextResponse.json(
      { error: "Cannot change the owner's role. Transfer ownership first." },
      { status: 400 }
    );
  }

  // Only the owner can promote to admin
  if (newRole === "admin" && callerRole !== "owner") {
    const ownerCheck = await isTeamOwner(user.id, teamId);
    if (!ownerCheck) {
      return NextResponse.json(
        { error: "Only the team owner can promote members to admin" },
        { status: 403 }
      );
    }
  }

  const { data, error } = await supabase
    .from("team_members")
    .update({ role: newRole })
    .eq("team_id", teamId)
    .eq("user_id", targetUserId)
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  void logAudit({
    userId: user.id,
    action: "team.member.role_change",
    resourceType: "team",
    resourceId: teamId,
    metadata: { target_user_id: targetUserId, old_role: targetMember.role, new_role: newRole },
  });

  return NextResponse.json(data);
}
