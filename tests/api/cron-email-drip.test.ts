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

// Mock auth helpers
vi.mock("@/lib/auth-helpers", () => ({
  verifyCronSecret: vi.fn().mockImplementation(async (header: string | null) => {
    return header === "Bearer test-cron-secret";
  }),
}));

// Mock notifications
const mockSendEmail = vi.fn().mockResolvedValue(true);
vi.mock("@/lib/notifications", () => ({
  sendEmail: (...args: unknown[]) => mockSendEmail(...args),
}));

const { GET } = await import("@/app/api/cron/email-drip/route");

describe("GET /api/cron/email-drip", () => {
  beforeEach(() => {
    resetMocks();
    vi.clearAllMocks();
    mockSendEmail.mockResolvedValue(true);
  });

  it("rejects without CRON_SECRET", async () => {
    const req = createGetRequest("/api/cron/email-drip");
    const res = await GET(req as any);
    expect(res.status).toBe(401);
  });

  it("returns success with no users needing drip", async () => {
    setMockData("users", []);

    const req = createGetRequest("/api/cron/email-drip", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.users_processed).toBe(0);
    expect(body.emails_sent).toBe(0);
  });

  it("sends welcome email to new user (day 0)", async () => {
    setMockData("users", [
      {
        id: "user-1",
        email: "new@test.com",
        created_at: new Date().toISOString(), // Today
        onboarding_drip_sent: 0,
      },
    ]);

    const req = createGetRequest("/api/cron/email-drip", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.users_processed).toBe(1);

    // Verify sendEmail was called with welcome subject
    expect(mockSendEmail).toHaveBeenCalledWith(
      "new@test.com",
      expect.stringContaining("Welcome"),
      expect.any(String)
    );
  });

  it("skips user who signed up today but needs day-1 email", async () => {
    setMockData("users", [
      {
        id: "user-2",
        email: "wait@test.com",
        created_at: new Date().toISOString(),
        onboarding_drip_sent: 1, // Needs day 1 email, but signed up today
      },
    ]);

    const req = createGetRequest("/api/cron/email-drip", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.users_processed).toBe(0);
    expect(mockSendEmail).not.toHaveBeenCalled();
  });

  it("sends day-3 email to user who signed up 4 days ago", async () => {
    const fourDaysAgo = new Date(Date.now() - 4 * 86_400_000).toISOString();
    setMockData("users", [
      {
        id: "user-3",
        email: "returning@test.com",
        created_at: fourDaysAgo,
        onboarding_drip_sent: 2, // Next is drip index 2 = day 3
      },
    ]);

    const req = createGetRequest("/api/cron/email-drip", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.users_processed).toBe(1);

    expect(mockSendEmail).toHaveBeenCalledWith(
      "returning@test.com",
      expect.stringContaining("dashboard"),
      expect.any(String)
    );
  });

  it("handles email send failure gracefully", async () => {
    mockSendEmail.mockResolvedValue(false);

    setMockData("users", [
      {
        id: "user-4",
        email: "fail@test.com",
        created_at: new Date().toISOString(),
        onboarding_drip_sent: 0,
      },
    ]);

    const req = createGetRequest("/api/cron/email-drip", {
      authorization: "Bearer test-cron-secret",
    });
    const res = await GET(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.users_processed).toBe(1);
    expect(body.emails_sent).toBe(0); // Failed to send
  });
});
