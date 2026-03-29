// Role-based access control for multi-tenant teams.
// Role hierarchy: owner > admin > billing > viewer

import { createSupabaseServerClient } from "@/lib/supabase-client";

export type TeamRole = "owner" | "admin" | "billing" | "viewer";

const ROLE_RANK: Record<TeamRole, number> = {
  owner: 4,
  admin: 3,
  billing: 2,
  viewer: 1,
};

/**
 * Check if a user has at least the required role in a team.
 * Returns { allowed: true, role } or { allowed: false }.
 */
export async function requireRole(
  userId: string,
  teamId: string,
  minRole: TeamRole
): Promise<{ allowed: boolean; role?: TeamRole }> {
  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("team_members")
    .select("role")
    .eq("team_id", teamId)
    .eq("user_id", userId)
    .not("accepted_at", "is", null)
    .single();

  if (error || !data) {
    return { allowed: false };
  }

  const userRole = data.role as TeamRole;
  const allowed = ROLE_RANK[userRole] >= ROLE_RANK[minRole];
  return { allowed, role: userRole };
}

/**
 * Get all team IDs a user belongs to (accepted memberships only).
 * Returns { teamIds, teams } where teams includes id, name, slug, role.
 */
export async function getTeamScope(userId: string): Promise<{
  teamIds: string[];
  teams: Array<{ id: string; name: string; slug: string; role: TeamRole }>;
}> {
  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("team_members")
    .select("team_id, role, teams(id, name, slug)")
    .eq("user_id", userId)
    .not("accepted_at", "is", null);

  if (error || !data) {
    return { teamIds: [], teams: [] };
  }

  const teams = data.map((row: any) => ({
    id: row.teams.id as string,
    name: row.teams.name as string,
    slug: row.teams.slug as string,
    role: row.role as TeamRole,
  }));

  return {
    teamIds: teams.map((t) => t.id),
    teams,
  };
}

/**
 * Check if a user can manage (admin+) a specific team.
 * Shorthand for requireRole with minRole = "admin".
 */
export async function canManageTeam(
  userId: string,
  teamId: string
): Promise<boolean> {
  const { allowed } = await requireRole(userId, teamId, "admin");
  return allowed;
}

/**
 * Check if a user is the owner of a team.
 */
export async function isTeamOwner(
  userId: string,
  teamId: string
): Promise<boolean> {
  const { allowed } = await requireRole(userId, teamId, "owner");
  return allowed;
}
