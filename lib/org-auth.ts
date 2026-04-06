// Organization authorization helpers

import { createSupabaseServerClient } from "./supabase-client";

type OrgRole = "owner" | "admin" | "member";

const ROLE_HIERARCHY: Record<OrgRole, number> = {
  owner: 3,
  admin: 2,
  member: 1,
};

/**
 * Check if a user has at least the minimum required role in an organization.
 */
export async function requireOrgRole(
  userId: string,
  orgId: string,
  minRole: OrgRole
): Promise<{ allowed: boolean; role: OrgRole | null }> {
  const supabase = createSupabaseServerClient();
  const { data } = await supabase
    .from("org_members")
    .select("role")
    .eq("org_id", orgId)
    .eq("user_id", userId)
    .single();

  if (!data) return { allowed: false, role: null };

  const userLevel = ROLE_HIERARCHY[data.role as OrgRole] ?? 0;
  const requiredLevel = ROLE_HIERARCHY[minRole] ?? 0;

  return {
    allowed: userLevel >= requiredLevel,
    role: data.role as OrgRole,
  };
}

/**
 * Get all organizations a user belongs to.
 */
export async function getUserOrgs(userId: string) {
  const supabase = createSupabaseServerClient();
  const { data } = await supabase
    .from("org_members")
    .select("org_id, role, organizations(id, name, slug, plan)")
    .eq("user_id", userId);

  return (data ?? []).map((m: any) => ({
    org_id: m.org_id,
    role: m.role,
    ...m.organizations,
  }));
}

/**
 * Get user's active org. For now, returns the first org they belong to.
 * In the future, this could read from a cookie or user preference.
 */
export async function getUserActiveOrg(userId: string) {
  const orgs = await getUserOrgs(userId);
  return orgs[0] ?? null;
}
