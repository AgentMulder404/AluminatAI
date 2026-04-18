import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";

const ADMIN_EMAILS = (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

interface ProspectRow {
  company_name: string;
  company_url?: string;
  linkedin_url?: string;
  domain?: string;
  industry?: string;
  company_size?: string;
  employee_count?: number;
  location?: string;
  description?: string;
  contact_name?: string;
  contact_email?: string;
  contact_title?: string;
  contact_phone?: string;
  contact_linkedin?: string;
  email_verified?: boolean;
  email_status?: string;
  source_query?: string;
  category?: string;
}

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user }, error } = await cookieClient.auth.getUser();
  if (error || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? ""))
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const body = await req.json();
  const prospects: ProspectRow[] = body.prospects ?? [];

  if (prospects.length === 0)
    return NextResponse.json({ error: "No prospects to load" }, { status: 400 });

  const db = createSupabaseServerClient();

  const rows = prospects.map((p) => ({
    ...p,
    source: "apify",
    status: "new",
    created_by: user.id,
  }));

  const { data, error: dbError } = await db
    .from("prospects")
    .upsert(rows, { onConflict: "domain,contact_email", ignoreDuplicates: true })
    .select("id");

  if (dbError) {
    console.error("Prospect load error:", dbError);
    return NextResponse.json({ error: dbError.message }, { status: 500 });
  }

  return NextResponse.json({ inserted: data?.length ?? 0, total: prospects.length });
}

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user }, error } = await cookieClient.auth.getUser();
  if (error || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? ""))
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const db = createSupabaseServerClient();
  const limit = parseInt(req.nextUrl.searchParams.get("limit") ?? "200");
  const offset = parseInt(req.nextUrl.searchParams.get("offset") ?? "0");
  const status = req.nextUrl.searchParams.get("status");

  let query = db
    .from("prospects")
    .select("*", { count: "exact" })
    .order("discovered_at", { ascending: false })
    .range(offset, offset + limit - 1);

  if (status) query = query.eq("status", status);

  const { data, error: dbError, count } = await query;

  if (dbError) return NextResponse.json({ error: dbError.message }, { status: 500 });

  return NextResponse.json({ prospects: data, count });
}
