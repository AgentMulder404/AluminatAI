import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setMockData,
  resetMocks,
  mockSupabaseClient,
} from "../helpers/mock-supabase";
import { createJsonRequest } from "../helpers/mock-request";

// Mock supabase client
vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

// Mock api-auth
const mockValidateApiKey = vi.fn();
vi.mock("@/lib/api-auth", () => ({
  validateApiKey: (...args: unknown[]) => mockValidateApiKey(...args),
}));

// Mock rate limiter
const mockRateLimit = vi.fn();
vi.mock("@/lib/rate-limiter", () => ({
  rateLimit: (...args: unknown[]) => mockRateLimit(...args),
  getRateLimitHeaders: () => ({
    "X-RateLimit-Remaining": "99",
    "X-RateLimit-Reset": "1700000000",
  }),
}));

// Import the route handler AFTER mocks are set up
const { POST } = await import("@/app/api/metrics/ingest/route");

describe("POST /api/metrics/ingest", () => {
  beforeEach(() => {
    resetMocks();
    vi.clearAllMocks();
    mockValidateApiKey.mockResolvedValue({ valid: true, userId: "user-123" });
    mockRateLimit.mockResolvedValue({
      success: true,
      remaining: 99,
      resetTime: Date.now() + 60000,
    });
    // Default: insert succeeds
    setMockData("gpu_metrics", null);
    setMockData("carbon_intensities", null);
  });

  it("rejects missing API key", async () => {
    mockValidateApiKey.mockResolvedValue({ valid: false, userId: "" });

    const req = createJsonRequest("/api/metrics/ingest", [
      { timestamp: new Date().toISOString(), power_draw_w: 250 },
    ]);
    const res = await POST(req as any);
    expect(res.status).toBe(401);
    const body = await res.json();
    expect(body.error).toBe("Unauthorized");
  });

  it("rejects invalid API key prefix", async () => {
    mockValidateApiKey.mockResolvedValue({ valid: false, userId: "" });

    const req = createJsonRequest(
      "/api/metrics/ingest",
      [{ timestamp: new Date().toISOString() }],
      { "x-api-key": "bad_key_here" }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(401);
  });

  it("returns 429 when rate limited", async () => {
    mockRateLimit.mockResolvedValue({
      success: false,
      remaining: 0,
      resetTime: Date.now() + 60000,
    });

    const req = createJsonRequest(
      "/api/metrics/ingest",
      [{ timestamp: new Date().toISOString() }],
      { "x-api-key": "alum_test123" }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(429);
  });

  it("accepts valid single metric", async () => {
    const req = createJsonRequest(
      "/api/metrics/ingest",
      {
        timestamp: new Date().toISOString(),
        gpu_uuid: "GPU-abc-123",
        gpu_name: "NVIDIA A100-SXM4-80GB",
        power_draw_w: 250,
        temperature_c: 65,
        utilization_gpu_pct: 85,
        memory_used_mb: 40000,
      },
      { "x-api-key": "alum_test123" }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.inserted).toBe(1);
  });

  it("accepts valid batch of metrics", async () => {
    const metrics = Array.from({ length: 5 }, (_, i) => ({
      timestamp: new Date(Date.now() - i * 5000).toISOString(),
      gpu_uuid: "GPU-abc-123",
      power_draw_w: 200 + i * 10,
      utilization_gpu_pct: 80,
    }));

    const req = createJsonRequest("/api/metrics/ingest", metrics, {
      "x-api-key": "alum_test123",
    });
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.inserted).toBe(5);
  });

  it("rejects batch larger than 1000", async () => {
    const metrics = Array.from({ length: 1001 }, () => ({
      timestamp: new Date().toISOString(),
      power_draw_w: 100,
    }));

    const req = createJsonRequest("/api/metrics/ingest", metrics, {
      "x-api-key": "alum_test123",
    });
    const res = await POST(req as any);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toContain("Batch too large");
  });

  it("filters out metrics with power_draw_w > 1500", async () => {
    const req = createJsonRequest(
      "/api/metrics/ingest",
      [
        {
          timestamp: new Date().toISOString(),
          power_draw_w: 1600,
          utilization_gpu_pct: 50,
        },
      ],
      { "x-api-key": "alum_test123" }
    );
    const res = await POST(req as any);
    const body = await res.json();
    expect(body.inserted).toBe(0);
  });

  it("filters out metrics with utilization_gpu_pct > 100", async () => {
    const req = createJsonRequest(
      "/api/metrics/ingest",
      [
        {
          timestamp: new Date().toISOString(),
          power_draw_w: 250,
          utilization_gpu_pct: 150,
        },
      ],
      { "x-api-key": "alum_test123" }
    );
    const res = await POST(req as any);
    const body = await res.json();
    expect(body.inserted).toBe(0);
  });

  it("filters out metrics with memory_used_mb > 1000000", async () => {
    const req = createJsonRequest(
      "/api/metrics/ingest",
      [
        {
          timestamp: new Date().toISOString(),
          memory_used_mb: 2_000_000,
        },
      ],
      { "x-api-key": "alum_test123" }
    );
    const res = await POST(req as any);
    const body = await res.json();
    expect(body.inserted).toBe(0);
  });

  it("filters out metrics with gpu_fraction > 1", async () => {
    const req = createJsonRequest(
      "/api/metrics/ingest",
      [
        {
          timestamp: new Date().toISOString(),
          gpu_fraction: 1.5,
        },
      ],
      { "x-api-key": "alum_test123" }
    );
    const res = await POST(req as any);
    const body = await res.json();
    expect(body.inserted).toBe(0);
  });

  it("returns { inserted: 0 } for empty array", async () => {
    const req = createJsonRequest("/api/metrics/ingest", [], {
      "x-api-key": "alum_test123",
    });
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.inserted).toBe(0);
  });

  it("returns 400 for invalid JSON body", async () => {
    const req = new Request(
      "https://www.aluminatai.com/api/metrics/ingest",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-api-key": "alum_test123",
        },
        body: "not json{{{",
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("Invalid JSON");
  });
});
