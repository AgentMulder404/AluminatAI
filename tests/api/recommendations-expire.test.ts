import { describe, it, expect, vi, beforeEach } from "vitest";
import { createGetRequest } from "../helpers/mock-request";
import { setMockData, resetMocks, mockSupabaseClient } from "../helpers/mock-supabase";

vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

const CRON_SECRET = "test-cron-secret";

function cronRequest() {
  return createGetRequest("/api/cron/recommendations-expire", {
    authorization: `Bearer ${CRON_SECRET}`,
  });
}

describe("GET /api/cron/recommendations-expire", () => {
  beforeEach(() => {
    resetMocks();
    vi.resetModules();
  });

  it("returns 401 without cron secret", async () => {
    const { GET } = await import("@/app/api/cron/recommendations-expire/route");
    const req = createGetRequest("/api/cron/recommendations-expire");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("expires pending recommendations past their expiry date", async () => {
    setMockData("optimization_recommendations", [
      { id: "rec-1", user_id: "u1" },
      { id: "rec-2", user_id: "u1" },
    ]);
    setMockData("recommendation_actions", null);

    const { GET } = await import("@/app/api/cron/recommendations-expire/route");
    const res = await GET(cronRequest() as any);
    expect(res.status).toBe(200);

    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.expired_count).toBe(2);
  });

  it("handles no expired recommendations gracefully", async () => {
    setMockData("optimization_recommendations", []);

    const { GET } = await import("@/app/api/cron/recommendations-expire/route");
    const res = await GET(cronRequest() as any);
    const body = await res.json();

    expect(body.ok).toBe(true);
    expect(body.expired_count).toBe(0);
  });
});
