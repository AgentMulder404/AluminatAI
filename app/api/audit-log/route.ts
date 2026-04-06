// GET /api/audit-log — paginated, filterable audit log

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const params = req.nextUrl.searchParams;
  const action = params.get("action");
  const resourceType = params.get("resource_type");
  const from = params.get("from");
  const to = params.get("to");
  const limit = Math.min(Number(params.get("limit") ?? 50), 200);
  const offset = Number(params.get("offset") ?? 0);

  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("audit_log")
    .select("id, user_id, org_id, action, resource_type, resource_id, metadata, ip_address, created_at", {
      count: "exact",
    })
    .eq("user_id", user.id)
    .order("created_at", { ascending: false })
    .range(offset, offset + limit - 1);

  if (action) query = query.eq("action", action);
  if (resourceType) query = query.eq("resource_type", resourceType);
  if (from) query = query.gte("created_at", from);
  if (to) query = query.lte("created_at", to);

  const { data: entries, count, error } = await query;

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({
    entries: entries ?? [],
    total: count ?? 0,
  });
}
