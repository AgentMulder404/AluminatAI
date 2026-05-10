import { NextRequest, NextResponse } from "next/server";
import { validateApiKey } from "@/lib/api-auth";
import { createSupabaseServerClient } from "@/lib/supabase-client";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  const apiKey = req.headers.get("x-api-key") ?? "";
  const auth = await validateApiKey(apiKey);
  if (!auth.valid) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const {
    agent_version,
    hostname,
    gpu_count,
    gpu_uuids,
    scheduler,
    uptime_sec,
    config_hash,
    machine_id,
    cluster_tag,
    location_hint,
    gpu_names,
    error_count_total,
    error_count_last_hour,
    last_error_message,
    last_error_at,
    os_info,
    python_version,
    gpu_backend,
    agent_mode,
  } = body as {
    agent_version?: string;
    hostname?: string;
    gpu_count?: number;
    gpu_uuids?: string[];
    scheduler?: string;
    uptime_sec?: number;
    config_hash?: string;
    machine_id?: string;
    cluster_tag?: string;
    location_hint?: string;
    gpu_names?: string[];
    error_count_total?: number;
    error_count_last_hour?: number;
    last_error_message?: string;
    last_error_at?: number;
    os_info?: string;
    python_version?: string;
    gpu_backend?: string;
    agent_mode?: string;
  };

  const supabase = createSupabaseServerClient();
  const { error } = await supabase.from("agent_heartbeats").upsert(
    {
      user_id: auth.userId,
      hostname: hostname ?? "",
      agent_version: agent_version ?? "",
      gpu_count: gpu_count ?? 0,
      gpu_uuids: gpu_uuids ?? [],
      scheduler: scheduler ?? "none",
      uptime_sec: uptime_sec ?? 0,
      config_hash: config_hash ?? "",
      machine_id: machine_id ?? null,
      cluster_tag: cluster_tag ?? "",
      location_hint: location_hint ?? "",
      gpu_names: gpu_names ?? [],
      last_seen: new Date().toISOString(),
      ...(error_count_total !== undefined && { error_count_total }),
      ...(error_count_last_hour !== undefined && { error_count_last_hour }),
      ...(last_error_message !== undefined && { last_error_message }),
      ...(last_error_at !== undefined && {
        last_error_at: new Date(last_error_at * 1000).toISOString(),
      }),
      ...(os_info !== undefined && { os_info }),
      ...(python_version !== undefined && { python_version }),
      ...(gpu_backend !== undefined && { gpu_backend }),
      ...(agent_mode !== undefined && { agent_mode }),
    },
    { onConflict: "user_id,hostname" }
  );

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
