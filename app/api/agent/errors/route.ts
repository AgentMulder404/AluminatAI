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

  const rl = await rateLimit(`agent-errors:${auth.userId}`, 100);
  if (!rl.success) {
    return NextResponse.json({ error: "Rate limited" }, { status: 429 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { machine_id, hostname, errors } = body as {
    machine_id?: string;
    hostname?: string;
    errors?: Array<{
      timestamp: number;
      error_type: string;
      message: string;
      gpu_index?: number;
      stack_trace?: string;
    }>;
  };

  if (!machine_id || !Array.isArray(errors) || errors.length === 0) {
    return NextResponse.json({ error: "machine_id and non-empty errors array required" }, { status: 400 });
  }

  const maxErrors = errors.slice(0, 100);

  const rows = maxErrors.map((e) => ({
    user_id: auth.userId,
    machine_id,
    hostname: hostname ?? "",
    error_type: (e.error_type ?? "unknown").slice(0, 100),
    error_message: (e.message ?? "").slice(0, 2000),
    stack_trace: e.stack_trace ? e.stack_trace.slice(0, 8000) : null,
    gpu_index: e.gpu_index ?? null,
    created_at: e.timestamp
      ? new Date(e.timestamp * 1000).toISOString()
      : new Date().toISOString(),
  }));

  const supabase = createSupabaseServerClient();
  const { error } = await supabase.from("agent_error_log").insert(rows);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true, inserted: rows.length });
}
