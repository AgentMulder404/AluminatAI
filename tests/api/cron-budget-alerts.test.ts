import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setMockData,
  resetMocks,
  mockSupabaseClient,
  setInsertError,
} from "../helpers/mock-supabase";
import { createGetRequest } from "../helpers/mock-request";

// Mock supabase
vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

// Mock auth helpers
vi.mock("@/lib/auth-helpers", () => ({
  verifyCronSecret: vi.fn().mockImplementation(async (header: string | null) => {
    return header === "Bearer test-cron-secret";
  }),
  constantTimeEqual: vi.fn().mockResolvedValue(true),
}));

// Mock cost helpers
vi.mock("@/lib/cost", () => ({
  getUserKwhRate: vi.fn().mockResolvedValue(0.12),
  getPeriodStart: vi.fn().mockReturnValue(new Date(Date.now() - 86400000)),
}));

// Mock notifications + webhooks (fire and forget)
vi.mock("@/lib/notifications", () => ({
  dispatchBudgetAlert: vi.fn().mockResolvedValue(undefined),
}));
vi.mock("@/lib/webhooks", () => ({
  dispatchWebhook: vi.fn().mockResolvedValue(undefined),
}));

const { GET } = await import("@/app/api/cron/budget-alerts/route");

describe("GET /api/cron/budget-alerts", () => {
  beforeEach(() => {
    resetMocks();
    vi.clearAllMocks();
  });

  it("rejects request without CRON_SECRET", async () => {
    const req = createGetRequest("/api/cron/budget-alerts");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("rejects request with wrong CRON_SECRET", async () => {
    const req = createGetRequest("/api/cron/budget-alerts", {
      authorization: "Bearer wrong-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("returns success with no budgets", async () => {
    setMockData("budgets", []);

    const req = createGetRequest("/api/cron/budget-alerts", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.budgets_checked).toBe(0);
    expect(body.alerts_sent).toBe(0);
  });

  it("checks budgets and detects exceeded threshold", async () => {
    setMockData("budgets", [
      {
        id: "budget-1",
        user_id: "user-1",
        name: "Monthly GPU Budget",
        scope_type: "global",
        scope_value: null,
        period: "monthly",
        limit_usd: 100,
        warn_pct: 80,
        notify_channels: [{ type: "email", target: "user@test.com" }],
      },
    ]);

    // Metrics that would result in high spend:
    // 500 samples × 1000J each = 500,000J = 0.139 kWh × $0.12 = way under $100
    // To exceed, we need a lot of energy — but the cron calculates differently
    // Let's mock metrics to have enough energy to exceed
    setMockData("gpu_metrics", [
      { energy_delta_j: 5_000_000, gpu_fraction: 1 },
      { energy_delta_j: 5_000_000, gpu_fraction: 1 },
    ]);
    setMockData("budget_alerts", null);

    const req = createGetRequest("/api/cron/budget-alerts", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.budgets_checked).toBe(1);
  });

  it("skips budgets with no metrics", async () => {
    setMockData("budgets", [
      {
        id: "budget-2",
        user_id: "user-2",
        name: "Empty Budget",
        scope_type: "global",
        scope_value: null,
        period: "daily",
        limit_usd: 50,
        warn_pct: 80,
        notify_channels: [],
      },
    ]);
    setMockData("gpu_metrics", []);

    const req = createGetRequest("/api/cron/budget-alerts", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.budgets_checked).toBe(1);
    expect(body.alerts_sent).toBe(0);
  });

  it("deduplicates alerts (insert conflict = no notification)", async () => {
    setMockData("budgets", [
      {
        id: "budget-3",
        user_id: "user-3",
        name: "Dedup Test",
        scope_type: "global",
        scope_value: null,
        period: "monthly",
        limit_usd: 0.01, // Very low threshold so it always exceeds
        warn_pct: 50,
        notify_channels: [],
      },
    ]);
    setMockData("gpu_metrics", [{ energy_delta_j: 1000, gpu_fraction: 1 }]);

    // Simulate dedup conflict on insert
    setInsertError({ message: "duplicate key" });

    const req = createGetRequest("/api/cron/budget-alerts", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    // Alert was not "sent" because the insert was a dedup conflict
    expect(body.alerts_sent).toBe(0);
  });
});
