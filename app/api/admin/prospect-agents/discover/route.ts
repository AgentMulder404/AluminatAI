import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { startActorRun } from "@/lib/apify";

const ADMIN_EMAILS = (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user }, error } = await cookieClient.auth.getUser();
  if (error || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? ""))
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });

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

  const runs = [];
  for (const keyword of keywords) {
    const input: Record<string, unknown> = {
      searchQuery: keyword,
      scraperMode: "full",
      maxItems,
      companySize,
    };
    if (locations.length > 0) input.locations = locations;

    const run = await startActorRun("harvestapi~linkedin-company-search", input);
    runs.push({ keyword, ...run });
  }

  return NextResponse.json({ runs, total: runs.length });
}
