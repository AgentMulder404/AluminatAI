import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setMockData,
  resetMocks,
  mockSupabaseClient,
} from "../helpers/mock-supabase";
import { createGetRequest } from "../helpers/mock-request";

// Mock supabase
vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

// Mock cookie-based auth
const mockGetUser = vi.fn();
vi.mock("@/lib/supabase-server", () => ({
  createSupabaseCookieClient: () =>
    Promise.resolve({
      auth: { getUser: mockGetUser },
    }),
}));

// Mock cost helper
vi.mock("@/lib/cost", () => ({
  getUserKwhRate: vi.fn().mockResolvedValue(0.12),
}));

// Dynamic import after mocks
let GET: (req: any) => Promise<Response>;

describe("GET /api/dashboard/today-cost", () => {
  beforeEach(async () => {
    resetMocks();
    vi.clearAllMocks();
    mockGetUser.mockResolvedValue({
      data: { user: { id: "user-123" } },
      error: null,
    });
    // Re-import to pick up fresh mocks
    const mod = await import("@/app/api/dashboard/today-cost/route");
    GET = mod.GET;
  });

  it("rejects unauthenticated request", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: null },
      error: { message: "Not authenticated" },
    });

    const req = createGetRequest("/api/dashboard/today-cost");
    const res = await GET(req);
    expect(res.status).toBe(401);
  });

  it("returns cost data for authenticated user", async () => {
    setMockData("gpu_metrics", [
      { energy_delta_j: 3600, gpu_fraction: 1, power_draw_w: 300 },
    ]);

    const req = createGetRequest("/api/dashboard/today-cost");
    const res = await GET(req);
    expect(res.status).toBe(200);
    const body = await res.json();
    // The response should have cost_usd field
    expect(body).toHaveProperty("cost_usd");
  });
});
