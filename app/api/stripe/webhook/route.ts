// POST /api/stripe/webhook — handle Stripe webhook events
export const runtime = "edge";

import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { logAudit } from "@/lib/audit";

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY ?? "", {
  apiVersion: "2024-12-18.acacia",
});

const WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET ?? "";

async function verifyStripeSignature(
  body: string,
  signature: string
): Promise<Stripe.Event> {
  // Use Stripe SDK's constructEvent for webhook verification
  return stripe.webhooks.constructEvent(body, signature, WEBHOOK_SECRET);
}

export async function POST(req: NextRequest) {
  const body = await req.text();
  const signature = req.headers.get("stripe-signature");

  if (!signature) {
    return NextResponse.json({ error: "Missing signature" }, { status: 400 });
  }

  let event: Stripe.Event;
  try {
    event = await verifyStripeSignature(body, signature);
  } catch {
    return NextResponse.json({ error: "Invalid signature" }, { status: 400 });
  }

  const supabase = createSupabaseServerClient();

  try {
    switch (event.type) {
      case "checkout.session.completed": {
        const session = event.data.object as Stripe.Checkout.Session;
        const userId = session.metadata?.user_id;
        const plan = session.metadata?.plan;
        if (!userId || !plan) break;

        // Fetch subscription details
        const subscriptionId =
          typeof session.subscription === "string"
            ? session.subscription
            : session.subscription?.id;

        if (subscriptionId) {
          const subscription =
            await stripe.subscriptions.retrieve(subscriptionId);

          await supabase
            .from("users")
            .update({
              plan,
              stripe_subscription_id: subscriptionId,
              plan_period_end: new Date(
                subscription.current_period_end * 1000
              ).toISOString(),
              plan_cancel_at_period_end: false,
            })
            .eq("id", userId);
        }

        await recordEvent(supabase, {
          userId,
          stripeEventId: event.id,
          eventType: event.type,
          plan,
          amountUsd: session.amount_total
            ? session.amount_total / 100
            : undefined,
        });

        void logAudit({
          userId,
          action: "plan.checkout_completed",
          resourceType: "subscription",
          resourceId: subscriptionId,
          metadata: { plan, amount_usd: session.amount_total ? session.amount_total / 100 : null },
        });
        break;
      }

      case "invoice.paid": {
        const invoice = event.data.object as Stripe.Invoice;
        const subscriptionId =
          typeof invoice.subscription === "string"
            ? invoice.subscription
            : invoice.subscription?.id;

        if (!subscriptionId) break;

        const subscription =
          await stripe.subscriptions.retrieve(subscriptionId);
        const userId = subscription.metadata?.user_id;
        const plan = subscription.metadata?.plan;
        if (!userId) break;

        await supabase
          .from("users")
          .update({
            plan: plan ?? "pro",
            plan_period_end: new Date(
              subscription.current_period_end * 1000
            ).toISOString(),
            plan_cancel_at_period_end: subscription.cancel_at_period_end,
          })
          .eq("id", userId);

        await recordEvent(supabase, {
          userId,
          stripeEventId: event.id,
          eventType: event.type,
          plan: plan ?? "pro",
          amountUsd: invoice.amount_paid / 100,
          periodStart: new Date(
            subscription.current_period_start * 1000
          ).toISOString(),
          periodEnd: new Date(
            subscription.current_period_end * 1000
          ).toISOString(),
        });
        break;
      }

      case "customer.subscription.updated": {
        const subscription = event.data.object as Stripe.Subscription;
        const userId = subscription.metadata?.user_id;
        if (!userId) break;

        const plan = subscription.metadata?.plan ?? "pro";

        await supabase
          .from("users")
          .update({
            plan:
              subscription.status === "active" ||
              subscription.status === "trialing"
                ? plan
                : "free",
            plan_period_end: new Date(
              subscription.current_period_end * 1000
            ).toISOString(),
            plan_cancel_at_period_end: subscription.cancel_at_period_end,
          })
          .eq("id", userId);

        await recordEvent(supabase, {
          userId,
          stripeEventId: event.id,
          eventType: event.type,
          plan,
        });
        break;
      }

      case "customer.subscription.deleted": {
        const subscription = event.data.object as Stripe.Subscription;
        const userId = subscription.metadata?.user_id;
        if (!userId) break;

        await supabase
          .from("users")
          .update({
            plan: "free",
            stripe_subscription_id: null,
            plan_cancel_at_period_end: false,
          })
          .eq("id", userId);

        await recordEvent(supabase, {
          userId,
          stripeEventId: event.id,
          eventType: event.type,
          plan: "free",
        });

        void logAudit({
          userId,
          action: "plan.subscription_deleted",
          resourceType: "subscription",
          metadata: { downgraded_to: "free" },
        });
        break;
      }
    }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    console.error("Stripe webhook processing error:", message);
    // Return 200 so Stripe doesn't retry — we logged the error
    return NextResponse.json({ received: true, error: message });
  }

  return NextResponse.json({ received: true });
}

// ── Helpers ────────────────────────────────────────────────────────────────

interface EventRecord {
  userId: string;
  stripeEventId: string;
  eventType: string;
  plan?: string;
  amountUsd?: number;
  periodStart?: string;
  periodEnd?: string;
}

async function recordEvent(
  supabase: ReturnType<typeof createSupabaseServerClient>,
  ev: EventRecord
) {
  await supabase.from("subscription_events").upsert(
    {
      user_id: ev.userId,
      stripe_event_id: ev.stripeEventId,
      event_type: ev.eventType,
      plan: ev.plan,
      amount_usd: ev.amountUsd,
      period_start: ev.periodStart,
      period_end: ev.periodEnd,
    },
    { onConflict: "stripe_event_id" }
  );
}
