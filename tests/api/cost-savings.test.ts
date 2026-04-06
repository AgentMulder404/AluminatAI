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

// Mock RBAC
vi.mock("@/lib/rbac", () => ({
  requireRole: vi.fn().mockResolvedValue({ allowed: true }),
}));

const { GET } = await import("@/app/api/cost/savings/route");

describe("GET /api/cost/savings", () => {
  beforeEach(() => {
    resetMocks();
    vi.clearAllMocks();
    mockGetUser.mockResolvedValue({
      data: { user: { id: "user-123" } },
      error: null,
    });
  });

  it("rejects unauthenticated requests", async () => {
    mockGetUser.mockResolvedValue({
      data: { user: null },
      error: { message: "Not authenticated" },
    });

    const req = createGetRequest("/api/cost/savings");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("returns zero savings when no metrics exist", async () => {
    setMockData("gpu_metrics", []);
    setMockData("gpu_reference_pricing", []);

    const req = createGetRequest("/api/cost/savings?days=30");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.total_savings_usd).toBe(0);
    expect(body.total_actual_usd).toBe(0);
    expect(body.by_gpu).toEqual([]);
  });

  it("calculates savings when metrics and reference pricing exist", async () => {
    // 100 samples of A100 usage
    setMockData("gpu_metrics", Array.from({ length: 100 }, () => ({
      gpu_name: "NVIDIA A100-SXM4-80GB",
      energy_delta_j: 1800, // 1800J per 5-second sample ≈ 360W
      gpu_fraction: 1,
      time: new Date().toISOString(),
    })));

    setMockData("gpu_reference_pricing", [
      {
        gpu_model: "A100-SXM4-80GB",
        provider: "AWS",
        rate_usd_per_gpu_hour: 3.0,
      },
    ]);

    const req = createGetRequest("/api/cost/savings?days=30");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.period_days).toBe(30);
    expect(body.total_actual_usd).toBeGreaterThanOrEqual(0);
    expect(body.total_cloud_equivalent_usd).toBeGreaterThan(0);
    expect(body.by_gpu.length).toBe(1);
    expect(body.by_gpu[0].gpu_name).toBe("NVIDIA A100-SXM4-80GB");
    expect(body.by_gpu[0].reference_provider).toBe("AWS");
  });

  it("caps days parameter at 365", async () => {
    setMockData("gpu_metrics", []);
    setMockData("gpu_reference_pricing", []);

    const req = createGetRequest("/api/cost/savings?days=9999");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.period_days).toBe(365);
  });

  it("skips GPUs with no reference pricing", async () => {
    setMockData("gpu_metrics", [
      {
        gpu_name: "Unknown-GPU-Model",
        energy_delta_j: 1000,
        gpu_fraction: 1,
        time: new Date().toISOString(),
      },
    ]);
    setMockData("gpu_reference_pricing", [
      {
        gpu_model: "A100-SXM4-80GB",
        provider: "AWS",
        rate_usd_per_gpu_hour: 3.0,
      },
    ]);

    const req = createGetRequest("/api/cost/savings?days=30");
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    // Unknown GPU has no reference pricing, so it's skipped
    expect(body.by_gpu).toEqual([]);
    expect(body.total_savings_usd).toBe(0);
  });
});
