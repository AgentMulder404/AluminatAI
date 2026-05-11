import { describe, it, expect, vi, beforeEach } from "vitest";
import { createGetRequest } from "../helpers/mock-request";
import { setMockData, resetMocks, mockSupabaseClient } from "../helpers/mock-supabase";

vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

const mockDispatchNotification = vi.fn().mockResolvedValue({ sent: [] });
vi.mock("@/lib/notifications", () => ({
  dispatchNotification: (...args: unknown[]) => mockDispatchNotification(...args),
}));

const CRON_SECRET = "test-cron-secret";

function cronRequest() {
  return createGetRequest("/api/cron/agent-health-check", {
    authorization: `Bearer ${CRON_SECRET}`,
  });
}

const now = Date.now();

describe("GET /api/cron/agent-health-check", () => {
  beforeEach(() => {
    resetMocks();
    mockDispatchNotification.mockClear();
    vi.resetModules();
  });

  it("returns 401 without cron secret", async () => {
    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const req = createGetRequest("/api/cron/agent-health-check");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("reports no alerts when everything is healthy", async () => {
    setMockData("agent_heartbeats", [
      {
        user_id: "u1",
        machine_id: "m1",
        hostname: "gpu-host-1",
        agent_version: "0.3.0",
        config_hash: "abc123",
        last_seen: new Date(now - 60_000).toISOString(),
        error_count_last_hour: 0,
      },
    ]);
    setMockData("swarm_leader_leases", []);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", []);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.alerts_created).toBe(0);
  });

  it("detects offline agent", async () => {
    setMockData("agent_heartbeats", [
      {
        user_id: "u1",
        machine_id: "m1",
        hostname: "gpu-host-1",
        agent_version: "0.3.0",
        config_hash: "abc123",
        last_seen: new Date(now - 15 * 60_000).toISOString(),
        error_count_last_hour: 0,
      },
    ]);
    setMockData("swarm_leader_leases", []);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", []);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();
    expect(body.alerts_created).toBeGreaterThanOrEqual(1);
  });

  it("detects expired leader lease", async () => {
    setMockData("agent_heartbeats", []);
    setMockData("swarm_leader_leases", [
      {
        user_id: "u1",
        cluster_tag: "prod",
        machine_id: "m1",
        expires_at: new Date(now - 10 * 60_000).toISOString(),
      },
    ]);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", []);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();
    expect(body.alerts_created).toBeGreaterThanOrEqual(1);
  });

  it("does not alert on active leader lease", async () => {
    setMockData("agent_heartbeats", []);
    setMockData("swarm_leader_leases", [
      {
        user_id: "u1",
        cluster_tag: "prod",
        machine_id: "m1",
        expires_at: new Date(now + 5 * 60_000).toISOString(),
      },
    ]);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", []);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();
    expect(body.alerts_created).toBe(0);
  });

  it("detects stuck commands", async () => {
    setMockData("agent_heartbeats", []);
    setMockData("swarm_leader_leases", []);
    setMockData("agent_commands", [
      {
        user_id: "u1",
        machine_id: "m1",
        status: "pending",
        created_at: new Date(now - 60 * 60_000).toISOString(),
      },
      {
        user_id: "u1",
        machine_id: "m1",
        status: "dispatched",
        created_at: new Date(now - 45 * 60_000).toISOString(),
      },
    ]);
    setMockData("optimization_recommendations", []);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();
    expect(body.alerts_created).toBeGreaterThanOrEqual(1);
  });

  it("detects recommendation backlog (10+ pending >24h)", async () => {
    const oldRecs = Array.from({ length: 12 }, (_, i) => ({
      user_id: "u1",
      machine_id: `m${i}`,
      created_at: new Date(now - 48 * 3600_000).toISOString(),
    }));

    setMockData("agent_heartbeats", []);
    setMockData("swarm_leader_leases", []);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", oldRecs);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();
    expect(body.alerts_created).toBeGreaterThanOrEqual(1);
  });

  it("does not alert on small recommendation backlog (<10)", async () => {
    const fewRecs = Array.from({ length: 5 }, (_, i) => ({
      user_id: "u1",
      machine_id: `m${i}`,
      created_at: new Date(now - 48 * 3600_000).toISOString(),
    }));

    setMockData("agent_heartbeats", []);
    setMockData("swarm_leader_leases", []);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", fewRecs);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();
    expect(body.alerts_created).toBe(0);
  });

  it("dispatches notification for critical alerts", async () => {
    setMockData("agent_heartbeats", [
      {
        user_id: "u1",
        machine_id: "m1",
        hostname: "gpu-host-1",
        agent_version: "0.3.0",
        config_hash: "abc123",
        last_seen: new Date(now - 15 * 60_000).toISOString(),
        error_count_last_hour: 0,
      },
    ]);
    setMockData("swarm_leader_leases", []);
    setMockData("agent_commands", []);
    setMockData("optimization_recommendations", []);
    setMockData("agent_health_alerts", []);

    const { GET } = await import("@/app/api/cron/agent-health-check/route");
    await GET(cronRequest() as any);

    expect(mockDispatchNotification).toHaveBeenCalled();
    const call = mockDispatchNotification.mock.calls[0];
    expect(call[0]).toBe("u1");
    expect(call[1]).toBe("agent_offline");
  });
});
