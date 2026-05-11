import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "").split(",").map((e) => e.trim().toLowerCase()).filter(Boolean);

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user } } = await cookieClient.auth.getUser();
  if (!user || !ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  const rl = await rateLimit(`admin:${user.id}`, 30);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const machineId = req.nextUrl.searchParams.get("machine_id");
  const errorType = req.nextUrl.searchParams.get("error_type");
  const limit = Math.min(parseInt(req.nextUrl.searchParams.get("limit") ?? "100"), 500);
  const offset = parseInt(req.nextUrl.searchParams.get("offset") ?? "0");

  const supabase = createSupabaseServerClient();

  let query = supabase
    .from("agent_error_log")
    .select("*", { count: "exact" })
    .order("created_at", { ascending: false })
    .range(offset, offset + limit - 1);

  if (machineId) {
    query = query.eq("machine_id", machineId);
  }
  if (errorType) {
    query = query.eq("error_type", errorType);
  }

  const { data: errors, error, count } = await query;
  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ errors: errors ?? [], total: count ?? 0 });
}
