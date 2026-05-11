import { describe, it, expect, vi, beforeEach } from "vitest";
import { createGetRequest } from "../helpers/mock-request";
import { setMockData, resetMocks, mockSupabaseClient } from "../helpers/mock-supabase";

vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

let mockUser: { id: string; email: string } | null = null;
vi.mock("@/lib/supabase-server", () => ({
  createSupabaseCookieClient: () =>
    Promise.resolve({
      auth: {
        getUser: () =>
          Promise.resolve({
            data: { user: mockUser },
            error: mockUser ? null : { message: "Not authenticated" },
          }),
      },
    }),
}));

describe("GET /api/dashboard/activity", () => {
  beforeEach(() => {
    resetMocks();
    mockUser = null;
    vi.resetModules();
  });

  it("returns 401 when unauthenticated", async () => {
    const { GET } = await import("@/app/api/dashboard/activity/route");
    const req = createGetRequest("/api/dashboard/activity");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("returns enriched events from recommendation_actions", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };

    setMockData("recommendation_actions", [
      {
        id: "ra-1",
        action: "approved",
        metadata: {},
        created_at: new Date().toISOString(),
        optimization_recommendations: {
          title: "Cap GPU 0 to 200W",
          hostname: "gpu-host-1",
          machine_id: "m1",
          gpu_name: "H100",
          source: "auto_tuner",
          category: "power_cap",
          priority: "P1",
          estimated_savings_pct: 25,
          status: "approved",
        },
      },
    ]);
    setMockData("agent_commands", []);

    const { GET } = await import("@/app/api/dashboard/activity/route");
    const req = createGetRequest("/api/dashboard/activity?limit=10");
    const res = await GET(req as any);
    expect(res.status).toBe(200);

    const body = await res.json();
    expect(body.events).toHaveLength(1);
    expect(body.events[0].action).toBe("approved");
    expect(body.events[0].title).toBe("Cap GPU 0 to 200W");
    expect(body.events[0].source).toBe("auto_tuner");
    expect(body.events[0].estimated_savings_pct).toBe(25);
  });

  it("includes agent command events", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };

    setMockData("recommendation_actions", []);
    setMockData("agent_commands", [
      {
        id: "cmd-1",
        command_type: "apply_power_cap",
        machine_id: "m1",
        status: "applied",
        result: { success: true },
        completed_at: new Date().toISOString(),
        recommendation_id: "rec-1",
        optimization_recommendations: {
          title: "Cap GPU 0",
          hostname: "gpu-host-1",
          source: "auto_tuner",
          category: "power_cap",
          priority: "P2",
          gpu_name: "H100",
          estimated_savings_pct: 20,
        },
      },
    ]);

    const { GET } = await import("@/app/api/dashboard/activity/route");
    const req = createGetRequest("/api/dashboard/activity");
    const res = await GET(req as any);
    const body = await res.json();

    expect(body.events).toHaveLength(1);
    expect(body.events[0].action).toBe("command_applied");
    expect(body.events[0].event_type).toBe("agent_command");
  });

  it("returns empty events when no activity exists", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("recommendation_actions", []);
    setMockData("agent_commands", []);

    const { GET } = await import("@/app/api/dashboard/activity/route");
    const req = createGetRequest("/api/dashboard/activity");
    const res = await GET(req as any);
    const body = await res.json();

    expect(body.events).toHaveLength(0);
    expect(body.has_more).toBe(false);
  });

  it("respects limit parameter", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };

    const actions = Array.from({ length: 5 }, (_, i) => ({
      id: `ra-${i}`,
      action: "created",
      metadata: {},
      created_at: new Date(Date.now() - i * 60_000).toISOString(),
      optimization_recommendations: {
        title: `Rec ${i}`,
        hostname: "h1",
        machine_id: "m1",
        gpu_name: null,
        source: "auto_tuner",
        category: "power_cap",
        priority: "P2",
        estimated_savings_pct: 10,
        status: "pending",
      },
    }));

    setMockData("recommendation_actions", actions);
    setMockData("agent_commands", []);

    const { GET } = await import("@/app/api/dashboard/activity/route");
    const req = createGetRequest("/api/dashboard/activity?limit=3");
    const res = await GET(req as any);
    const body = await res.json();

    expect(body.events.length).toBeLessThanOrEqual(3);
  });
});
