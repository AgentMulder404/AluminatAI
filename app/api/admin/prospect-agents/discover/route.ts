import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { startActorRun } from "@/lib/apify";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

const ADMIN_EMAILS = (process.env.ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

export async function POST(req: NextRequest) {
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

  const body = await req.json();
  const keywords: string[] = body.keywords ?? [
    "GPU cloud",
    "AI infrastructure",
    "machine learning",
    "deep learning infrastructure",
    "AI startup GPU",
  ];
  const companySize: string[] = body.companySize ?? ["11-50", "51-200", "201-500", "501-1000"];
  const locations: string[] = body.locations ?? [];
  const maxItems: number = Math.min(body.maxItems ?? 20, 50);

  const errors: string[] = [];

  // Start all discovery runs in parallel
  const results = await Promise.allSettled(
    keywords.map((keyword) => {
      const input: Record<string, unknown> = {
        searchQuery: keyword,
        scraperMode: "full",
        maxItems,
        companySize,
      };
      if (locations.length > 0) input.locations = locations;
      return startActorRun("harvestapi~linkedin-company-search", input).then((run) => ({
        keyword,
        ...run,
      }));
    })
  );

  const runs = [];
  for (const r of results) {
    if (r.status === "fulfilled") {
      runs.push(r.value);
    } else {
      errors.push(r.reason?.message ?? "Unknown error");
    }
  }

  if (runs.length === 0) {
    return NextResponse.json(
      { error: `All discovery runs failed: ${errors.join("; ")}`, runs: [], total: 0 },
      { status: 502 }
    );
  }

  return NextResponse.json({ runs, total: runs.length, errors });
}
