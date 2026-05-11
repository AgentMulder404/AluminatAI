import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { getRunStatus } from "@/lib/apify";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user }, error } = await cookieClient.auth.getUser();
  if (error || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? ""))
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const rl = await rateLimit(`admin:${user.id}`, 30);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  const runId = req.nextUrl.searchParams.get("runId");
  if (!runId) return NextResponse.json({ error: "runId required" }, { status: 400 });

  try {
    const status = await getRunStatus(runId);
    return NextResponse.json(status);
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Status check failed", status: "ERROR" },
      { status: 502 }
    );
  }
}
