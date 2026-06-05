// POST /api/benchmarks/hardware
// X-API-Key auth (agent-facing). Accepts a benchmark measurement from the
// `nemulai benchmark --upload` CLI subcommand and upserts into
// benchmark_submissions using the HMAC-based user_hash.

import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

import { safeError } from "@/lib/safe-error";
export const runtime = "edge";

async function computeUserHash(userId: string): Promise<string> {
  const salt = process.env.BENCHMARK_SALT ?? "";
  if (!salt) throw new Error("BENCHMARK_SALT not configured");

  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(salt),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const buf = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(userId)
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

interface HardwarePayload {
  gpu_arch: string;
  model_tag: string;
  avg_power_w: number;
  energy_j_per_gpu_hour: number;
  duration_seconds: number;
  gpu_count?: number;
  precision_tag?: string;
  batch_size_hint?: number;
  tokens_per_second?: number;
  framework_tag?: string;
  kwh_per_1m_tokens?: number;
}

export async function POST(req: NextRequest) {
  if (!process.env.BENCHMARK_SALT) {
    return NextResponse.json(
      { error: "Benchmarking not configured (missing BENCHMARK_SALT)" },
      { status: 500 }
    );
  }

  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`bench-hw:${auth.userId}`, 60);
  if (!rl.success) {
    return NextResponse.json(
      { error: "Rate limit exceeded" },
      { status: 429, headers: getRateLimitHeaders(rl) }
    );
  }

  let body: HardwarePayload;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const {
    gpu_arch,
    model_tag,
    avg_power_w,
    energy_j_per_gpu_hour,
    duration_seconds,
    gpu_count = 1,
    precision_tag = "unknown",
    batch_size_hint,
    tokens_per_second,
    framework_tag = "unknown",
    kwh_per_1m_tokens,
  } = body;

  // Validate required fields
  if (!gpu_arch || !model_tag) {
    return NextResponse.json(
      { error: "gpu_arch and model_tag are required" },
      { status: 400 }
    );
  }
  if (
    typeof avg_power_w !== "number" ||
    typeof energy_j_per_gpu_hour !== "number" ||
    typeof duration_seconds !== "number"
  ) {
    return NextResponse.json(
      { error: "avg_power_w, energy_j_per_gpu_hour, duration_seconds must be numbers" },
      { status: 400 }
    );
  }
  if (energy_j_per_gpu_hour > 1e9 || energy_j_per_gpu_hour <= 0) {
    return NextResponse.json(
      { error: "energy_j_per_gpu_hour out of plausible range" },
      { status: 422 }
    );
  }
  if (tokens_per_second !== undefined && tokens_per_second <= 0) {
    return NextResponse.json(
      { error: "tokens_per_second must be > 0" },
      { status: 422 }
    );
  }
  if (framework_tag && framework_tag.length > 32) {
    return NextResponse.json(
      { error: "framework_tag max 32 characters" },
      { status: 422 }
    );
  }

  let userHash: string;
  try {
    userHash = await computeUserHash(auth.userId);
  } catch {
    return NextResponse.json(
      { error: "Benchmarking not configured" },
      { status: 500 }
    );
  }

  const now = new Date().toISOString();
  const supabase = createSupabaseServerClient();

  const { error } = await supabase.from("benchmark_submissions").upsert(
    {
      user_hash: userHash,
      gpu_arch,
      model_tag,
      precision_tag,
      batch_size_hint: batch_size_hint ?? null,
      avg_power_w,
      energy_j_per_gpu_hour,
      duration_seconds,
      gpu_count,
      tokens_per_second: tokens_per_second ?? null,
      framework_tag,
      kwh_per_1m_tokens: kwh_per_1m_tokens ?? null,
      opt_in_at: now,
      updated_at: now,
    },
    { onConflict: "user_hash,gpu_arch,model_tag" }
  );

  if (error) {
    return NextResponse.json({ error: safeError(error) }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
