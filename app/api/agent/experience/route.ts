import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { rateLimit } from "@/lib/rate-limiter";

export const runtime = "edge";

interface ExperiencePayload {
  id?: string;
  machine_id: string;
  gpu_index: number;
  gpu_name: string;
  context?: {
    gpu_arch?: string;
    workload_class?: string;
    utilization_gpu_pct?: number;
    utilization_memory_pct?: number;
    memory_pressure?: number;
    power_draw_w?: number;
    power_limit_w?: number;
    temperature_c?: number;
  };
  action?: {
    action_type: string;
    source: string;
    recommended_value?: number;
    current_value?: number;
    estimated_savings_pct?: number;
  };
  outcome?: {
    energy_delta_j_before?: number;
    energy_delta_j_after?: number;
    throughput_before?: number;
    throughput_after?: number;
    actual_savings_pct?: number;
    recommendation_status?: string;
    observation_window_s?: number;
  };
  reward?: number;
}

export async function POST(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const rl = await rateLimit(`agent-exp:${auth.userId}`, 100);
  if (!rl.success) {
    return NextResponse.json({ error: "Rate limited" }, { status: 429 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const { experiences } = body as { experiences?: ExperiencePayload[] };
  if (!Array.isArray(experiences) || experiences.length === 0) {
    return NextResponse.json({ error: "Non-empty experiences array required" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();
  const rows = experiences.slice(0, 500).map((exp) => {
    const ctx = exp.context ?? {};
    const act = exp.action ?? { action_type: "unknown", source: "unknown" };
    const out = exp.outcome;

    return {
      user_id: auth.userId,
      machine_id: (exp.machine_id ?? "").slice(0, 64),
      gpu_index: exp.gpu_index ?? 0,
      gpu_name: (exp.gpu_name ?? "").slice(0, 100),
      gpu_arch: (ctx.gpu_arch ?? "").slice(0, 64),
      workload_class: (ctx.workload_class ?? "unknown").slice(0, 64),
      utilization_gpu: Math.max(0, Math.min(100, ctx.utilization_gpu_pct ?? 0)),
      utilization_mem: Math.max(0, Math.min(100, ctx.utilization_memory_pct ?? 0)),
      memory_pressure: Math.max(0, Math.min(1, ctx.memory_pressure ?? 0)),
      power_draw_w: Math.max(0, Math.min(1500, ctx.power_draw_w ?? 0)),
      power_limit_w: Math.max(0, Math.min(1500, ctx.power_limit_w ?? 0)),
      temperature_c: Math.max(0, Math.min(120, ctx.temperature_c ?? 0)),
      action_type: act.action_type.slice(0, 32),
      action_source: act.source.slice(0, 32),
      recommended_value: act.recommended_value ?? 0,
      current_value: act.current_value ?? 0,
      estimated_savings: act.estimated_savings_pct ?? 0,
      energy_before_j: out?.energy_delta_j_before ?? null,
      energy_after_j: out?.energy_delta_j_after ?? null,
      throughput_before: out?.throughput_before ?? null,
      throughput_after: out?.throughput_after ?? null,
      actual_savings_pct: out?.actual_savings_pct ?? null,
      rec_status: out?.recommendation_status ?? null,
      observation_s: out?.observation_window_s ?? null,
      reward: exp.reward ?? null,
    };
  });

  const { error, count } = await supabase
    .from("experience_log")
    .insert(rows);

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true, inserted: rows.length });
}
