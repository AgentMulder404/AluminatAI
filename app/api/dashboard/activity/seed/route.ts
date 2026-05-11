import { NextRequest, NextResponse } from "next/server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { verifyCronSecret } from "@/lib/auth-helpers";

export async function POST(req: NextRequest) {
  const isAuthed = await verifyCronSecret(req.headers.get("authorization"));
  if (!isAuthed) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const body = await req.json().catch(() => ({}));
  const userId = (body as Record<string, string>).user_id;
  if (!userId) {
    return NextResponse.json(
      { error: "user_id required in request body" },
      { status: 400 }
    );
  }

  const supabase = createSupabaseServerClient();
  const now = Date.now();

  const SOURCES = ["auto_tuner", "workload_analyzer", "carbon_scheduler", "swarm_policy"];
  const CATEGORIES = ["power_cap", "precision", "utilization", "idle", "carbon_schedule"];
  const PRIORITIES = ["P1", "P2", "P3"];
  const MACHINES = ["gpu-host-1", "gpu-host-2", "gpu-host-3", "gpu-host-4"];
  const GPU_NAMES = ["H100-SXM5-80GB", "A100-SXM4-40GB", "RTX 4090", "MI300X"];

  const OFFSETS_MS = [
    0,
    30_000,
    60_000,
    90_000,
    5 * 60_000,
    15 * 60_000,
    45 * 60_000,
    2 * 3600_000,
    4 * 3600_000,
    6 * 3600_000,
    12 * 3600_000,
    18 * 3600_000,
    24 * 3600_000,
    30 * 3600_000,
    36 * 3600_000,
    40 * 3600_000,
    44 * 3600_000,
    47 * 3600_000,
  ];

  interface SeedRec {
    status: string;
    actionHistory: string[];
    savings: number;
  }

  const SEED_RECS: SeedRec[] = [
    { status: "applied", actionHistory: ["created", "approved", "applied"], savings: 25 },
    { status: "applied", actionHistory: ["created", "approved", "applied"], savings: 40 },
    { status: "applied", actionHistory: ["created", "approved", "applied"], savings: 18 },
    { status: "approved", actionHistory: ["created", "approved"], savings: 30 },
    { status: "approved", actionHistory: ["created", "approved"], savings: 22 },
    { status: "pending", actionHistory: ["created"], savings: 15 },
    { status: "pending", actionHistory: ["created"], savings: 35 },
    { status: "pending", actionHistory: ["created"], savings: 12 },
    { status: "rejected", actionHistory: ["created", "rejected"], savings: 20 },
    { status: "rejected", actionHistory: ["created", "rejected"], savings: 8 },
    { status: "expired", actionHistory: ["created", "expired"], savings: 10 },
    { status: "expired", actionHistory: ["created", "expired"], savings: 5 },
    { status: "rolled_back", actionHistory: ["created", "approved", "applied", "rolled_back"], savings: 28 },
    { status: "applied", actionHistory: ["created", "approved", "applied"], savings: 45 },
    { status: "pending", actionHistory: ["created"], savings: 33 },
    { status: "applied", actionHistory: ["created", "approved", "applied"], savings: 22 },
    { status: "rejected", actionHistory: ["created", "rejected"], savings: 7 },
    { status: "pending", actionHistory: ["created"], savings: 19 },
  ];

  const TITLES = [
    "Cap GPU 0 to 200W — save ~25% power",
    "Switch GPU 2 to BF16 mixed precision",
    "Reduce idle power on GPU 3",
    "Defer training job to low-carbon window",
    "Right-size GPU 1 — 80% idle",
    "Thermal throttle GPU 0 at 83°C",
    "Cap fleet idle GPUs to 150W",
    "Scale down GPU 5 memory clock",
    "Enable power management on GPU 2",
    "Migrate workload to efficient node",
    "Cap GPU 4 to 250W during idle",
    "Switch FP32 to TF32 for training",
    "Schedule batch jobs for 2am UTC",
    "Reduce GPU 6 power — low utilization",
    "Enable dynamic power scaling fleet-wide",
    "Cap overnight idle GPUs to 100W",
    "Optimize VRAM allocation on GPU 3",
    "Consolidate workloads to fewer GPUs",
  ];

  let recsCreated = 0;
  let actionsCreated = 0;
  let cmdsCreated = 0;

  for (let i = 0; i < SEED_RECS.length; i++) {
    const seed = SEED_RECS[i];
    const baseTime = now - OFFSETS_MS[i];
    const machine = MACHINES[i % MACHINES.length];
    const gpuIndex = i % 8;

    const { data: rec, error } = await supabase
      .from("optimization_recommendations")
      .insert({
        user_id: userId,
        machine_id: machine,
        hostname: machine,
        gpu_index: gpuIndex,
        gpu_name: GPU_NAMES[i % GPU_NAMES.length],
        source: SOURCES[i % SOURCES.length],
        category: CATEGORIES[i % CATEGORIES.length],
        priority: PRIORITIES[i % PRIORITIES.length],
        title: TITLES[i],
        description: `Seed recommendation ${i + 1} for testing the activity feed visualization.`,
        action: `Apply optimization to GPU ${gpuIndex}`,
        estimated_savings_pct: seed.savings,
        effort_score: (i % 5) + 1,
        action_payload: { command: "apply_power_cap", gpu_index: gpuIndex, power_limit_w: 200 },
        status: seed.status,
        expires_at: new Date(baseTime + 24 * 3600_000).toISOString(),
        ...(seed.status === "applied" && { applied_at: new Date(baseTime + 120_000).toISOString() }),
        ...(seed.status === "approved" && { approved_at: new Date(baseTime + 60_000).toISOString() }),
        ...(seed.status === "rolled_back" && {
          applied_at: new Date(baseTime + 120_000).toISOString(),
          rolled_back_at: new Date(baseTime + 300_000).toISOString(),
        }),
      })
      .select("id")
      .single();

    if (error || !rec) continue;
    recsCreated++;

    for (let j = 0; j < seed.actionHistory.length; j++) {
      const actionTime = baseTime + j * 60_000;
      await supabase.from("recommendation_actions").insert({
        recommendation_id: rec.id,
        user_id: userId,
        action: seed.actionHistory[j],
        metadata: { seeded: true, step: j },
        created_at: new Date(actionTime).toISOString(),
      });
      actionsCreated++;
    }

    if (seed.status === "applied" || seed.status === "rolled_back") {
      const cmdStatus = seed.status === "rolled_back" ? "applied" : "applied";
      await supabase.from("agent_commands").insert({
        user_id: userId,
        machine_id: machine,
        recommendation_id: rec.id,
        command_type: "apply_power_cap",
        params: { gpu_index: gpuIndex, power_limit_w: 200 },
        status: cmdStatus,
        result: { success: true, message: `Power cap set to 200W on GPU ${gpuIndex}` },
        created_at: new Date(baseTime + 60_000).toISOString(),
        dispatched_at: new Date(baseTime + 90_000).toISOString(),
        completed_at: new Date(baseTime + 120_000).toISOString(),
      });
      cmdsCreated++;
    }
  }

  // Add 2 failed commands for variety
  for (let i = 0; i < 2; i++) {
    await supabase.from("agent_commands").insert({
      user_id: userId,
      machine_id: MACHINES[i],
      command_type: "set_precision",
      params: { precision: "bf16" },
      status: "failed",
      result: { success: false, message: "GPU does not support BF16" },
      created_at: new Date(now - (3 + i) * 3600_000).toISOString(),
      dispatched_at: new Date(now - (3 + i) * 3600_000 + 30_000).toISOString(),
      completed_at: new Date(now - (3 + i) * 3600_000 + 60_000).toISOString(),
    });
    cmdsCreated++;
  }

  return NextResponse.json({
    ok: true,
    recommendations_created: recsCreated,
    actions_created: actionsCreated,
    commands_created: cmdsCreated,
  });
}
