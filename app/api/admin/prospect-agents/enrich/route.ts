import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { getDatasetItems, startActorRun } from "@/lib/apify";

const ADMIN_EMAILS = (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

interface LinkedInCompany {
  name: string;
  linkedinUrl?: string;
  website?: string;
  industry?: string;
  employeeCount?: number;
  employeeCountRange?: { start?: number; end?: number };
  description?: string;
  location?: { linkedinText?: string };
  locations?: Array<{ city?: string; country?: string }>;
}

interface EnrichedContact {
  "01_Name": string | null;
  "04_Email": string | null;
  "05_Phone_number": string | null;
  "06_Linkedin_url": string | null;
  "07_Title": string | null;
  "16_Company_name": string | null;
}

function extractDomain(url: string | undefined): string | null {
  if (!url) return null;
  try {
    const hostname = new URL(url.startsWith("http") ? url : `https://${url}`).hostname;
    return hostname.replace(/^www\./, "");
  } catch {
    return null;
  }
}

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user }, error } = await cookieClient.auth.getUser();
  if (error || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? ""))
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const body = await req.json();
  const datasetIds: string[] = body.datasetIds ?? [];
  const maxLeads: number = body.maxLeads ?? 3;

  const companies: Array<LinkedInCompany & { domain: string | null }> = [];
  const seenDomains = new Set<string>();

  for (const dsId of datasetIds) {
    try {
      const items = await getDatasetItems<LinkedInCompany>(dsId, 50);
      for (const item of items) {
        const domain = extractDomain(item.website);
        if (!domain || seenDomains.has(domain)) continue;
        seenDomains.add(domain);
        companies.push({ ...item, domain });
      }
    } catch (err) {
      console.error(`Failed to fetch dataset ${dsId}:`, err);
    }
  }

  const enrichRuns = [];
  for (const company of companies) {
    if (!company.domain) continue;
    try {
      const run = await startActorRun("snipercoder~decision-maker-email-finder", {
        domain: company.domain,
        decision_maker_category: "ceo_founder_owner",
        max_leads_to_find: maxLeads,
      });
      enrichRuns.push({
        ...run,
        domain: company.domain,
        companyName: company.name,
        linkedinUrl: company.linkedinUrl,
        website: company.website,
        industry: company.industry,
        employeeCount: company.employeeCount,
        companyDescription: company.description?.slice(0, 500),
        location: company.locations?.[0]?.country || company.location?.linkedinText,
      });
    } catch (err) {
      console.error(`Enrich failed for ${company.domain}:`, err);
    }
  }

  return NextResponse.json({
    companiesFound: companies.length,
    enrichRuns,
  });
}

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const { data: { user }, error } = await cookieClient.auth.getUser();
  if (error || !user) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? ""))
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });

  const datasetId = req.nextUrl.searchParams.get("datasetId");
  if (!datasetId) return NextResponse.json({ error: "datasetId required" }, { status: 400 });

  try {
    const contacts = await getDatasetItems<EnrichedContact>(datasetId, 50);
    return NextResponse.json({ contacts });
  } catch (err) {
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Failed to fetch contacts", contacts: [] },
      { status: 502 }
    );
  }
}
