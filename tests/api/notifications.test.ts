import { describe, it, expect, vi, beforeEach } from "vitest";
import { createGetRequest, createRequest } from "../helpers/mock-request";
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

describe("GET /api/notifications", () => {
  beforeEach(() => {
    resetMocks();
    mockUser = null;
  });

  it("returns 401 when unauthenticated", async () => {
    const { GET } = await import("@/app/api/notifications/route");
    const req = createGetRequest("/api/notifications");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("returns notifications and unread count when authenticated", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("notifications", [
      { id: "n1", type: "budget_alert", title: "Budget exceeded", message: "Over limit", read: false, created_at: new Date().toISOString() },
      { id: "n2", type: "system", title: "Welcome", message: "Hello", read: true, created_at: new Date().toISOString() },
    ]);

    const { GET } = await import("@/app/api/notifications/route");
    const req = createGetRequest("/api/notifications?limit=10");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.notifications).toBeDefined();
  });
});

describe("PATCH /api/notifications", () => {
  beforeEach(() => {
    resetMocks();
    mockUser = null;
  });

  it("returns 401 when unauthenticated", async () => {
    const { PATCH } = await import("@/app/api/notifications/route");
    const req = createRequest("/api/notifications", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: { all: true },
    });
    const res = await PATCH(req as any);
    expect(res.status).toBe(401);
  });

  it("marks all as read when authenticated", async () => {
    mockUser = { id: "user-1", email: "test@test.com" };
    setMockData("notifications", null);

    const { PATCH } = await import("@/app/api/notifications/route");
    const req = createRequest("/api/notifications", {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: { all: true },
    });
    const res = await PATCH(req as any);
    expect(res.status).toBe(200);
  });
});
