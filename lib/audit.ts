// Audit logging — fire-and-forget insert into audit_log table

import { createSupabaseServerClient } from "./supabase-client";

interface AuditEntry {
  userId: string;
  orgId?: string;
  action: string; // e.g. "budget.create", "team.member.add", "api_key.rotate"
  resourceType: string; // e.g. "budget", "team", "api_key", "export_config"
  resourceId?: string;
  metadata?: Record<string, unknown>;
  ip?: string;
  userAgent?: string;
}

/**
 * Log an audit entry. Fire-and-forget — errors are swallowed.
 */
export async function logAudit(entry: AuditEntry): Promise<void> {
  try {
    const supabase = createSupabaseServerClient();
    await supabase.from("audit_log").insert({
      user_id: entry.userId,
      org_id: entry.orgId ?? null,
      action: entry.action,
      resource_type: entry.resourceType,
      resource_id: entry.resourceId ?? null,
      metadata: entry.metadata ?? {},
      ip_address: entry.ip ?? null,
      user_agent: entry.userAgent ?? null,
    });
  } catch {
    // Swallow errors — audit logging should never block operations
  }
}
