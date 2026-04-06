// GET /api/dashboard/stream — Server-Sent Events for real-time dashboard updates
// Pushes live GPU metrics summary, notification count, and agent status.

import { NextRequest } from "next/server";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { getUserKwhRate } from "@/lib/cost";

export const runtime = "edge";

export async function GET(req: NextRequest) {
  const cookieClient = await createSupabaseCookieClient();
  const {
    data: { user },
    error: authError,
  } = await cookieClient.auth.getUser();

  if (authError || !user) {
    return new Response("Unauthorized", { status: 401 });
  }

  const userId = user.id;
  const encoder = new TextEncoder();
  let closed = false;

  const stream = new ReadableStream({
    async start(controller) {
      const supabase = createSupabaseServerClient();

      const send = (event: string, data: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(
            encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
          );
        } catch {
          closed = true;
        }
      };

      const poll = async () => {
        if (closed) return;

        try {
          // Today's metrics summary
          const todayStart = new Date();
          todayStart.setHours(0, 0, 0, 0);

          const { data: metrics } = await supabase
            .from("gpu_metrics")
            .select("energy_delta_j, gpu_fraction, power_draw_w")
            .eq("user_id", userId)
            .gte("time", todayStart.toISOString())
            .limit(50000);

          let totalJ = 0;
          let gpuCount = 0;
          for (const m of metrics ?? []) {
            const frac = (m.gpu_fraction ?? 1) as number;
            totalJ += ((m.energy_delta_j ?? 0) as number) * frac;
            gpuCount++;
          }

          const kwhRate = await getUserKwhRate(userId);
          const costUsd = (totalJ / 3_600_000) * kwhRate;

          send("metrics", {
            cost_usd: Math.round(costUsd * 100) / 100,
            sample_count: gpuCount,
            timestamp: new Date().toISOString(),
          });

          // Unread notification count
          const { count } = await supabase
            .from("notifications")
            .select("id", { count: "exact", head: true })
            .eq("user_id", userId)
            .eq("read", false);

          send("notification", { unread_count: count ?? 0 });

          // Agent status
          const { data: agents } = await supabase
            .from("agent_heartbeats")
            .select("agent_id, last_seen_at, machine_id")
            .eq("user_id", userId)
            .order("last_seen_at", { ascending: false })
            .limit(20);

          const now = Date.now();
          const agentStatus = (agents ?? []).map((a: any) => ({
            agent_id: a.agent_id,
            machine_id: a.machine_id,
            online:
              now - new Date(a.last_seen_at).getTime() < 10 * 60 * 1000, // 10 min threshold
          }));

          send("agent-status", { agents: agentStatus });
        } catch {
          // Silently skip failed polls
        }
      };

      // Initial data
      await poll();

      // Poll every 5 seconds
      const interval = setInterval(poll, 5000);

      // Heartbeat every 15 seconds to keep connection alive
      const heartbeat = setInterval(() => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(": heartbeat\n\n"));
        } catch {
          closed = true;
        }
      }, 15000);

      // Clean up on abort
      req.signal.addEventListener("abort", () => {
        closed = true;
        clearInterval(interval);
        clearInterval(heartbeat);
        try {
          controller.close();
        } catch {
          // Already closed
        }
      });
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
