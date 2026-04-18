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
  const emails: string[] = body.emails ?? [];

  if (emails.length === 0)
    return NextResponse.json({ error: "No emails to verify" }, { status: 400 });

  const run = await startActorRun("snipercoder~email-validator", {
    emails,
  });

  return NextResponse.json({ ...run, emailCount: emails.length });
}
