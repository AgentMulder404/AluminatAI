// POST /api/admin/prospect-scraper
// Searches for potential AluminatAI clients using Apify RAG Web Browser.
// Cookie auth — admin only.

import { NextRequest, NextResponse } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";

const ADMIN_EMAILS = (process.env.NEXT_PUBLIC_ADMIN_EMAIL ?? "")
  .split(",")
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

const APIFY_TOKEN = process.env.APIFY_API_TOKEN;

const SEARCH_TEMPLATES: Record<string, string[]> = {
  "ai-startups": [
    "AI startup GPU infrastructure cost optimization 2026",
    "machine learning company GPU cloud spending",
    "AI company looking to reduce GPU costs",
  ],
  "ml-teams": [
    "machine learning team GPU utilization monitoring",
    "ML engineering GPU cost attribution",
    "deep learning training GPU energy efficiency",
  ],
  "data-centers": [
    "data center GPU energy monitoring sustainability",
    "GPU data center power usage effectiveness PUE",
    "colocation GPU workload cost management",
  ],
  "research-labs": [
    "university research lab GPU computing costs",
    "academic HPC GPU energy consumption",
    "research computing GPU cost per experiment",
  ],
  "cloud-users": [
    "company migrating from cloud GPU to on-premise",
    "AWS GPU instance cost optimization",
    "cloud GPU spending too much alternative",
  ],
};

interface ApifyResult {
  crawl?: { httpStatusCode?: number };
  searchResult?: { title?: string; description?: string; url?: string };
  metadata?: { title?: string; description?: string };
  markdown?: string;
}

interface Prospect {
  title: string;
  url: string;
  description: string;
  snippet: string;
  category: string;
  query: string;
}

export async function POST(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error,
  } = await cookieClient.auth.getUser();
  if (error || !user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }
  if (!ADMIN_EMAILS.includes(user.email?.toLowerCase() ?? "")) {
    return NextResponse.json({ error: "Forbidden" }, { status: 403 });
  }

  if (!APIFY_TOKEN) {
    return NextResponse.json(
      { error: "APIFY_API_TOKEN not configured" },
      { status: 500 }
    );
  }

  const body = await req.json();
  const category: string = body.category || "ai-startups";
  const customQuery: string | undefined = body.query;
  const maxResults: number = Math.min(body.maxResults || 5, 10);

  const queries = customQuery
    ? [customQuery]
    : SEARCH_TEMPLATES[category] || SEARCH_TEMPLATES["ai-startups"];

  const allProspects: Prospect[] = [];

  for (const query of queries) {
    try {
      const url = new URL("https://rag-web-browser.apify.actor/search");
      url.searchParams.set("query", query);
      url.searchParams.set("maxResults", String(maxResults));

      const res = await fetch(url.toString(), {
        headers: {
          Authorization: `Bearer ${APIFY_TOKEN}`,
        },
      });

      if (!res.ok) {
        console.error(`Apify error for "${query}": ${res.status}`);
        continue;
      }

      const results: ApifyResult[] = await res.json();

      for (const r of results) {
        if (!r.searchResult?.url) continue;

        const snippet = (r.markdown || "").slice(0, 500);

        allProspects.push({
          title: r.searchResult.title || r.metadata?.title || "Untitled",
          url: r.searchResult.url,
          description:
            r.searchResult.description ||
            r.metadata?.description ||
            "",
          snippet,
          category,
          query,
        });
      }
    } catch (err) {
      console.error(`Apify fetch failed for "${query}":`, err);
    }
  }

  const seen = new Set<string>();
  const deduped = allProspects.filter((p) => {
    const domain = new URL(p.url).hostname;
    if (seen.has(domain)) return false;
    seen.add(domain);
    return true;
  });

  return NextResponse.json({ prospects: deduped, count: deduped.length });
}
