// GET /api/cron/data-retention
// Schedule: 0 3 * * * (daily at 3 AM UTC)
// Deletes expired rows per user's data_retention_policies.
// Requires CRON_SECRET bearer token.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { verifyCronSecret } from "@/lib/auth-helpers";

export const runtime = "edge";

const DEFAULT_RETENTION_DAYS = 90;

// Map table_name → the timestamp column to filter on
const TABLE_TS_COLUMN: Record<string, string> = {
  gpu_metrics: "timestamp",
  waste_events: "detected_at",
  webhook_deliveries: "created_at",
  budget_alerts: "created_at",
};

export async function GET(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  // Fetch all explicit retention policies
  const { data: policies, error: pErr } = await supabase
    .from("data_retention_policies")
    .select("user_id, table_name, retention_days");

  if (pErr) {
    return NextResponse.json({ error: pErr.message }, { status: 500 });
  }

  // Also get all users who have data but no explicit policy (use default)
  const { data: allUsers } = await supabase
    .from("users")
    .select("id");

  // Build per-user-per-table retention map
  const retentionMap = new Map<string, number>(); // "userId:tableName" → days
  for (const p of policies ?? []) {
    retentionMap.set(`${p.user_id}:${p.table_name}`, p.retention_days);
  }

  let totalDeleted = 0;
  const errors: string[] = [];

  // Process each table
  for (const [tableName, tsColumn] of Object.entries(TABLE_TS_COLUMN)) {
    // Find the minimum retention across all users for this table
    // We process by getting distinct user_ids from the table and applying their policy
    const usersWithPolicies = (policies ?? [])
      .filter((p: any) => p.table_name === tableName)
      .map((p: any) => ({ user_id: p.user_id, days: p.retention_days }));

    // For users with explicit policies
    for (const { user_id, days } of usersWithPolicies) {
      const cutoff = new Date(
        Date.now() - days * 24 * 60 * 60 * 1000
      ).toISOString();

      const userIdColumn = tableName === "gpu_metrics" ? "user_id" : "user_id";

      const { count, error: delErr } = await supabase
        .from(tableName)
        .delete({ count: "exact" })
        .eq(userIdColumn, user_id)
        .lt(tsColumn, cutoff);

      if (delErr) {
        errors.push(`${tableName}/${user_id}: ${delErr.message}`);
      } else {
        totalDeleted += count ?? 0;
      }
    }

    // For users without explicit policies, apply default
    const policyUserIds = new Set(usersWithPolicies.map((p: any) => p.user_id));
    const defaultCutoff = new Date(
      Date.now() - DEFAULT_RETENTION_DAYS * 24 * 60 * 60 * 1000
    ).toISOString();

    // Delete old rows for users without a custom policy
    // This is a bulk delete — any row older than 90 days from a user without a policy
    // Build the NOT IN filter safely — Supabase PostgREST expects (val1,val2) without quotes for UUIDs
    const idArray = Array.from(policyUserIds);
    let query = supabase
      .from(tableName)
      .delete({ count: "exact" })
      .lt(tsColumn, defaultCutoff);

    if (idArray.length > 0) {
      query = query.not("user_id", "in", `(${idArray.join(",")})`);
    }

    const { count: defaultCount, error: defErr } = await query;

    if (defErr) {
      errors.push(`${tableName}/default: ${defErr.message}`);
    } else {
      totalDeleted += defaultCount ?? 0;
    }
  }

  // Also prune old webhook deliveries globally (30 days hard cap)
  const deliveryCutoff = new Date(
    Date.now() - 30 * 24 * 60 * 60 * 1000
  ).toISOString();
  const { count: deliveryPruned } = await supabase
    .from("webhook_deliveries")
    .delete({ count: "exact" })
    .lt("created_at", deliveryCutoff);

  totalDeleted += deliveryPruned ?? 0;

  return NextResponse.json({
    deleted: totalDeleted,
    errors: errors.length > 0 ? errors : undefined,
    ran_at: new Date().toISOString(),
  });
}
