import { describe, it, expect, vi, beforeEach } from "vitest";
import { createJsonRequest, createGetRequest } from "../helpers/mock-request";
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

let mockApiKeyValid = false;
let mockApiKeyUserId = "user-1";
vi.mock("@/lib/api-auth", () => ({
  validateApiKey: () =>
    Promise.resolve(
      mockApiKeyValid
        ? { valid: true, userId: mockApiKeyUserId }
        : { valid: false }
    ),
}));

vi.mock("@/lib/rate-limiter", () => ({
  rateLimit: () => Promise.resolve({ success: true, remaining: 99 }),
}));

vi.mock("@/lib/notifications", () => ({
  dispatchNotification: vi.fn().mockResolvedValue({ sent: [] }),
  createInAppNotification: vi.fn().mockResolvedValue(true),
}));

describe("Recommendation pipeline (integration)", () => {
  beforeEach(() => {
    resetMocks();
    mockUser = null;
    mockApiKeyValid = false;
    mockApiKeyUserId = "user-1";
    vi.resetModules();
  });

  it("step 1: agent uploads a recommendation", async () => {
    mockApiKeyValid = true;
    setMockData("optimization_recommendations", []);

    const { POST } = await import("@/app/api/agent/recommendations/route");
    const req = createJsonRequest("/api/agent/recommendations", {
      recommendations: [
        {
          machine_id: "m1",
          source: "auto_tuner",
          category: "power_cap",
          title: "Cap GPU 0 to 200W",
          gpu_index: 0,
          estimated_savings_pct: 25,
          action_payload: { command: "apply_power_cap", gpu_index: 0, power_limit_w: 200 },
        },
      ],
    }, { "x-api-key": "alum_test_key" });

    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.inserted).toBe(1);
  });

  it("step 2: dashboard user approves the recommendation", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("optimization_recommendations", {
      id: "rec-1",
      user_id: "user-1",
      machine_id: "m1",
      status: "pending",
      action_payload: { command: "apply_power_cap", gpu_index: 0, power_limit_w: 200 },
    });
    setMockData("recommendation_actions", null);
    setMockData("agent_commands", null);

    const { POST } = await import("@/app/api/recommendations/[id]/approve/route");
    const req = createJsonRequest("/api/recommendations/rec-1/approve", {});
    const res = await POST(req as any, { params: Promise.resolve({ id: "rec-1" }) });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.status).toBe("approved");
  });

  it("step 3: agent polls and receives the command", async () => {
    mockApiKeyValid = true;
    setMockData("agent_commands", [
      {
        id: "cmd-1",
        command_type: "apply_power_cap",
        params: { command: "apply_power_cap", gpu_index: 0, power_limit_w: 200 },
        recommendation_id: "rec-1",
        created_at: new Date().toISOString(),
      },
    ]);

    const { GET } = await import("@/app/api/agent/commands/route");
    const req = createGetRequest("/api/agent/commands?machine_id=m1", {
      "x-api-key": "alum_test_key",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.commands).toHaveLength(1);
    expect(body.commands[0].command_type).toBe("apply_power_cap");
  });

  it("step 4: agent reports successful execution", async () => {
    mockApiKeyValid = true;
    setMockData("agent_commands", { id: "cmd-1", machine_id: "m1", recommendation_id: "rec-1" });
    setMockData("optimization_recommendations", null);
    setMockData("recommendation_actions", null);

    const { POST } = await import("@/app/api/agent/commands/[commandId]/result/route");
    const req = createJsonRequest("/api/agent/commands/cmd-1/result", {
      success: true,
      message: "Power cap applied to 200W on GPU 0",
      machine_id: "m1",
    }, { "x-api-key": "alum_test_key" });

    const res = await POST(req as any, { params: Promise.resolve({ commandId: "cmd-1" }) });
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.status).toBe("applied");
  });

  it("rejects upload without API key", async () => {
    mockApiKeyValid = false;

    const { POST } = await import("@/app/api/agent/recommendations/route");
    const req = createJsonRequest("/api/agent/recommendations", {
      recommendations: [{ machine_id: "m1", source: "auto_tuner", category: "test", title: "t" }],
    });
    const res = await POST(req as any);
    expect(res.status).toBe(401);
  });

  it("rejects approval of non-pending recommendation", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("optimization_recommendations", {
      id: "rec-2",
      user_id: "user-1",
      machine_id: "m1",
      status: "applied",
      action_payload: {},
    });

    const { POST } = await import("@/app/api/recommendations/[id]/approve/route");
    const req = createJsonRequest("/api/recommendations/rec-2/approve", {});
    const res = await POST(req as any, { params: Promise.resolve({ id: "rec-2" }) });
    expect(res.status).toBe(400);
  });

  it("rejects command result with mismatched machine_id", async () => {
    mockApiKeyValid = true;
    setMockData("agent_commands", { id: "cmd-1", machine_id: "m1" });

    const { POST } = await import("@/app/api/agent/commands/[commandId]/result/route");
    const req = createJsonRequest("/api/agent/commands/cmd-1/result", {
      success: true,
      message: "done",
      machine_id: "wrong-machine",
    }, { "x-api-key": "alum_test_key" });

    const res = await POST(req as any, { params: Promise.resolve({ commandId: "cmd-1" }) });
    expect(res.status).toBe(403);
  });
});
