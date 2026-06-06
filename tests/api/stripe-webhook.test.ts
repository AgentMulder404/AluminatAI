import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  setMockData,
  resetMocks,
  mockSupabaseClient,
} from "../helpers/mock-supabase";

// Mock supabase client
vi.mock("@/lib/supabase-client", () => ({
  createSupabaseServerClient: () => mockSupabaseClient,
}));

// Mock Stripe SDK
const mockConstructEvent = vi.fn();
const mockRetrieveSubscription = vi.fn();

vi.mock("stripe", () => {
  return {
    default: class StripeMock {
      webhooks = { constructEvent: mockConstructEvent };
      subscriptions = { retrieve: mockRetrieveSubscription };
    },
  };
});

const { POST } = await import("@/app/api/stripe/webhook/route");

describe("POST /api/stripe/webhook", () => {
  beforeEach(() => {
    resetMocks();
    vi.clearAllMocks();
    setMockData("users", { id: "user-123" });
    setMockData("subscription_events", null);
  });

  it("rejects missing stripe-signature header", async () => {
    const req = new Request(
      "https://www.nemulai.com/api/stripe/webhook",
      {
        method: "POST",
        body: "{}",
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("Missing signature");
  });

  it("rejects invalid signature", async () => {
    mockConstructEvent.mockImplementation(() => {
      throw new Error("Invalid signature");
    });

    const req = new Request(
      "https://www.nemulai.com/api/stripe/webhook",
      {
        method: "POST",
        headers: { "stripe-signature": "bad-sig" },
        body: "{}",
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("Invalid signature");
  });

  it("handles checkout.session.completed", async () => {
    mockConstructEvent.mockReturnValue({
      id: "evt_123",
      type: "checkout.session.completed",
      data: {
        object: {
          metadata: { user_id: "user-123", plan: "team" },
          subscription: "sub_123",
          amount_total: 2900,
        },
      },
    });

    mockRetrieveSubscription.mockResolvedValue({
      id: "sub_123",
      current_period_end: Math.floor(Date.now() / 1000) + 86400 * 30,
      cancel_at_period_end: false,
      metadata: { user_id: "user-123", plan: "team" },
    });

    const req = new Request(
      "https://www.nemulai.com/api/stripe/webhook",
      {
        method: "POST",
        headers: { "stripe-signature": "valid-sig" },
        body: JSON.stringify({}),
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.received).toBe(true);

    // Verify supabase update was called
    expect(mockSupabaseClient.from).toHaveBeenCalledWith("users");
  });

  it("handles customer.subscription.deleted — downgrades to free", async () => {
    mockConstructEvent.mockReturnValue({
      id: "evt_456",
      type: "customer.subscription.deleted",
      data: {
        object: {
          id: "sub_123",
          metadata: { user_id: "user-123" },
          current_period_end: Math.floor(Date.now() / 1000),
        },
      },
    });

    const req = new Request(
      "https://www.nemulai.com/api/stripe/webhook",
      {
        method: "POST",
        headers: { "stripe-signature": "valid-sig" },
        body: JSON.stringify({}),
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.received).toBe(true);
  });

  it("handles invoice.paid — extends subscription", async () => {
    mockConstructEvent.mockReturnValue({
      id: "evt_789",
      type: "invoice.paid",
      data: {
        object: {
          subscription: "sub_123",
          amount_paid: 2900,
        },
      },
    });

    mockRetrieveSubscription.mockResolvedValue({
      id: "sub_123",
      current_period_start: Math.floor(Date.now() / 1000),
      current_period_end: Math.floor(Date.now() / 1000) + 86400 * 30,
      cancel_at_period_end: false,
      metadata: { user_id: "user-123", plan: "team" },
    });

    const req = new Request(
      "https://www.nemulai.com/api/stripe/webhook",
      {
        method: "POST",
        headers: { "stripe-signature": "valid-sig" },
        body: JSON.stringify({}),
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.received).toBe(true);
  });

  it("returns 200 even on processing error (to prevent Stripe retries)", async () => {
    mockConstructEvent.mockReturnValue({
      id: "evt_err",
      type: "checkout.session.completed",
      data: {
        object: {
          metadata: { user_id: "user-123", plan: "team" },
          subscription: "sub_broken",
        },
      },
    });

    mockRetrieveSubscription.mockRejectedValue(new Error("Stripe API down"));

    const req = new Request(
      "https://www.nemulai.com/api/stripe/webhook",
      {
        method: "POST",
        headers: { "stripe-signature": "valid-sig" },
        body: JSON.stringify({}),
      }
    );
    const res = await POST(req as any);
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.received).toBe(true);
    expect(body.error).toBeDefined();
  });
});
