// POST /api/stripe/checkout — create a Stripe Checkout session for plan upgrade
export const runtime = "edge";

import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
import { STRIPE_PRICES, type PlanTier } from "@/lib/plans";
import { rateLimit, getRateLimitHeaders } from "@/lib/rate-limiter";

function getStripe() {
  const key = process.env.STRIPE_SECRET_KEY;
  if (!key) throw new Error("STRIPE_SECRET_KEY is not set");
  return new Stripe(key, { apiVersion: "2024-12-18.acacia" });
}

export async function POST(req: NextRequest) {
  try {
    const stripe = getStripe();
    const supabaseCookie = await createSupabaseCookieClient();
    const {
      data: { user },
    } = await supabaseCookie.auth.getUser();
    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const rl = await rateLimit(`stripe-checkout:${user.id}`, 10);
    if (!rl.success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429, headers: getRateLimitHeaders(rl) }
      );
    }

    const body = await req.json();
    const { plan, interval } = body as {
      plan: PlanTier;
      interval: "monthly" | "yearly";
    };

    if (!plan || !["team", "enterprise"].includes(plan)) {
      return NextResponse.json(
        { error: "Invalid plan. Must be team or enterprise." },
        { status: 400 }
      );
    }
    if (!interval || !["monthly", "yearly"].includes(interval)) {
      return NextResponse.json(
        { error: "Invalid interval. Must be monthly or yearly." },
        { status: 400 }
      );
    }

    const priceKey = `${plan}_${interval}` as keyof typeof STRIPE_PRICES;
    const priceId = STRIPE_PRICES[priceKey];
    if (!priceId) {
      return NextResponse.json(
        { error: "Stripe price not configured for this plan/interval" },
        { status: 500 }
      );
    }

    // Get or create Stripe customer
    const supabase = createSupabaseServerClient();
    const { data: userData } = await supabase
      .from("users")
      .select("stripe_customer_id, email")
      .eq("id", user.id)
      .single();

    let customerId = userData?.stripe_customer_id;

    if (!customerId) {
      const customer = await stripe.customers.create({
        email: userData?.email ?? user.email,
        metadata: { user_id: user.id },
      });
      customerId = customer.id;

      await supabase
        .from("users")
        .update({ stripe_customer_id: customerId })
        .eq("id", user.id);
    }

    // Validate origin — only allow configured app URL, never trust request header
    const origin = process.env.NEXT_PUBLIC_APP_URL ?? "https://www.nemulai.com";

    const session = await stripe.checkout.sessions.create({
      customer: customerId,
      mode: "subscription",
      line_items: [{ price: priceId, quantity: 1 }],
      success_url: `${origin}/dashboard/settings?billing=success`,
      cancel_url: `${origin}/dashboard/settings?billing=cancel`,
      metadata: { user_id: user.id, plan },
      subscription_data: {
        metadata: { user_id: user.id, plan },
      },
    });

    return NextResponse.json({ url: session.url });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
