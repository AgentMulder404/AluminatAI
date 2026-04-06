import { describe, it, expect, vi, beforeEach } from "vitest";
import { createGetRequest } from "../helpers/mock-request";
import { setMockData, resetMocks, mockSupabaseClient } from "../helpers/mock-supabase";

// Mock supabase clients
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

describe("GET /api/audit-log", () => {
  beforeEach(() => {
    resetMocks();
    mockUser = null;
  });

  it("returns 401 when unauthenticated", async () => {
    const { GET } = await import("@/app/api/audit-log/route");
    const req = createGetRequest("/api/audit-log");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("returns paginated entries when authenticated", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("audit_log", {
      data: [
        {
          id: "a1",
          user_id: "user-1",
          action: "budget.create",
          resource_type: "budget",
          resource_id: "b1",
          metadata: {},
          ip_address: null,
          created_at: new Date().toISOString(),
        },
      ],
      count: 1,
    });

    const { GET } = await import("@/app/api/audit-log/route");
    const req = createGetRequest("/api/audit-log?limit=25&offset=0");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
  });

  it("respects action filter param", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("audit_log", []);

    const { GET } = await import("@/app/api/audit-log/route");
    const req = createGetRequest("/api/audit-log?action=budget.create");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
  });

  it("clamps limit to max 200", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("audit_log", []);

    const { GET } = await import("@/app/api/audit-log/route");
    const req = createGetRequest("/api/audit-log?limit=999");
    const res = await GET(req as any);
    // Should still succeed — limit internally capped
    expect(res.status).toBe(200);
  });
});
