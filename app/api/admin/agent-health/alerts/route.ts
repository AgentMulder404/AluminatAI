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

  const supabase = createSupabaseServerClient();
  const { data: alerts, error } = await supabase
    .from("agent_health_alerts")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(200);

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ alerts: alerts ?? [] });
}
