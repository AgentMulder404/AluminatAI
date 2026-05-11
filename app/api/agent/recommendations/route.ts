import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit } from "@/lib/rate-limiter";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`agent-recs:${auth.userId}`, 60);
  if (!rl.success) {
    return NextResponse.json({ error: "Rate limited" }, { status: 429 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { recommendations } = body as {
    recommendations?: Array<{
      machine_id: string;
      hostname?: string;
      gpu_index?: number;
      gpu_name?: string;
      source: string;
      category: string;
      priority?: string;
      title: string;
      description?: string;
      action?: string;
      estimated_savings_pct?: number;
      effort_score?: number;
      action_payload?: Record<string, unknown>;
    }>;
  };

  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    return NextResponse.json({ error: "Non-empty recommendations array required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();
  const oneHourAgo = new Date(Date.now() - 3600_000).toISOString();

  let inserted = 0;
  for (const rec of recommendations.slice(0, 50)) {
    // Dedup: skip if same (machine_id, category, gpu_index) exists within 1 hour
    const { data: existing } = await supabase
      .from("optimization_recommendations")
      .select("id")
      .eq("user_id", auth.userId)
      .eq("machine_id", rec.machine_id)
      .eq("category", rec.category)
      .eq("gpu_index", rec.gpu_index ?? -1)
      .gte("created_at", oneHourAgo)
      .limit(1);

    if (existing && existing.length > 0) continue;

    const { data: newRec, error } = await supabase
      .from("optimization_recommendations")
      .insert({
        user_id: auth.userId,
        machine_id: rec.machine_id,
        hostname: rec.hostname ?? "",
        gpu_index: rec.gpu_index ?? null,
        gpu_name: rec.gpu_name ?? null,
        source: rec.source,
        category: rec.category,
        priority: rec.priority ?? "P2",
        title: (rec.title ?? "").slice(0, 200),
        description: (rec.description ?? "").slice(0, 2000),
        action: (rec.action ?? "").slice(0, 500),
        estimated_savings_pct: rec.estimated_savings_pct ?? 0,
        effort_score: Math.min(5, Math.max(1, rec.effort_score ?? 3)),
        action_payload: rec.action_payload ?? {},
        expires_at: new Date(Date.now() + 24 * 3600_000).toISOString(),
      })
      .select("id")
      .single();

    if (!error && newRec) {
      inserted++;
      void supabase.from("recommendation_actions").insert({
        recommendation_id: newRec.id,
        user_id: auth.userId,
        action: "created",
        metadata: { source: rec.source, category: rec.category },
      });
    }
  }

  return NextResponse.json({ ok: true, inserted });
}
