// GET/POST/PATCH /api/settings/retention
// Manage data retention policies. Cookie auth.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

const VALID_TABLES = [
  "gpu_metrics",
  "waste_events",
  "webhook_deliveries",
  "budget_alerts",
];
const MIN_RETENTION_DAYS = 7;
const DEFAULT_RETENTION_DAYS = 90;

async function authenticate() {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) return null;
  return user;
}

// GET — list retention policies
export async function GET() {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("data_retention_policies")
    .select("id, table_name, retention_days, updated_at")
    .eq("user_id", user.id);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  // Fill in defaults for tables without explicit policies
  const policies = VALID_TABLES.map((table) => {
    const existing = (data ?? []).find(
      (p: any) => p.table_name === table
    );
    return existing ?? {
      id: null,
      table_name: table,
      retention_days: DEFAULT_RETENTION_DAYS,
      updated_at: null,
    };
  });

  return NextResponse.json({ policies });
}

// POST — create or update a retention policy (upsert)
export async function POST(req: NextRequest) {
  const user = await authenticate();
  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json();
  const { table_name, retention_days } = body;

  if (!table_name || !VALID_TABLES.includes(table_name)) {
    return NextResponse.json(
      { error: `table_name must be one of: ${VALID_TABLES.join(", ")}` },
      { status: 400 }
    );
  }

  if (
    typeof retention_days !== "number" ||
    retention_days < MIN_RETENTION_DAYS
  ) {
    return NextResponse.json(
      { error: `retention_days must be at least ${MIN_RETENTION_DAYS}` },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();

  const { data, error } = await supabase
    .from("data_retention_policies")
    .upsert(
      {
        user_id: user.id,
        table_name,
        retention_days,
        updated_at: new Date().toISOString(),
      },
      { onConflict: "user_id,table_name" }
    )
    .select()
    .single();

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json(data);
}
