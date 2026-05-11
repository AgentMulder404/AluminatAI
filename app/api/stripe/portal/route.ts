// POST /api/stripe/portal — redirect user to Stripe Customer Portal
export const runtime = "edge";

import { NextRequest, NextResponse } from "next/server";
import Stripe from "stripe";
import { createSupabaseCookieClient } from "@/lib/supabase-server";
import { createSupabaseServerClient } from "@/lib/supabase-client";
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

    const rl = await rateLimit(`stripe-portal:${user.id}`, 10);
    if (!rl.success) {
      return NextResponse.json(
        { error: "Rate limit exceeded" },
        { status: 429, headers: getRateLimitHeaders(rl) }
      );
    }

    const supabase = createSupabaseServerClient();
    const { data: userData } = await supabase
      .from("users")
      .select("stripe_customer_id")
      .eq("id", user.id)
      .single();

    if (!userData?.stripe_customer_id) {
      return NextResponse.json(
        { error: "No billing account found. Subscribe to a plan first." },
        { status: 400 }
      );
    }

    // Validate origin — only allow configured app URL, never trust request header
    const origin = process.env.NEXT_PUBLIC_APP_URL ?? "https://www.aluminatai.com";

    const session = await stripe.billingPortal.sessions.create({
      customer: userData.stripe_customer_id,
      return_url: `${origin}/dashboard/settings`,
    });

    return NextResponse.json({ url: session.url });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
