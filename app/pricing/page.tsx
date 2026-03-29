"use client";

import { useState } from "react";
import { PLAN_LIMITS, PLAN_DISPLAY, type PlanTier } from "@/lib/plans";
import Link from "next/link";

const TIERS: PlanTier[] = ["free", "pro", "enterprise"];

const FEATURE_ROWS: {
  label: string;
  key: keyof typeof PLAN_LIMITS.free;
  format?: "boolean" | "count" | "days" | "rate";
}[] = [
  { label: "Teams", key: "max_teams", format: "count" },
  { label: "Team members", key: "max_team_members", format: "count" },
  { label: "Budgets", key: "max_budgets", format: "count" },
  { label: "Webhooks", key: "max_webhooks", format: "count" },
  { label: "Data exports", key: "max_export_configs", format: "count" },
  { label: "Data retention", key: "retention_days", format: "days" },
  { label: "API rate limit", key: "api_rate_limit", format: "rate" },
  { label: "Slack integration", key: "slack_integration", format: "boolean" },
  { label: "PagerDuty / OpsGenie", key: "pagerduty_opsgenie", format: "boolean" },
  { label: "SLA dashboard", key: "sla_dashboard", format: "boolean" },
  { label: "Priority support", key: "priority_support", format: "boolean" },
];

function formatValue(value: number | boolean, format?: string): string {
  if (typeof value === "boolean") return value ? "✓" : "—";
  if (value === -1) return "Unlimited";
  if (format === "days") return `${value} days`;
  if (format === "rate") return `${value}/min`;
  return String(value);
}

export default function PricingPage() {
  const [interval, setInterval] = useState<"monthly" | "yearly">("monthly");
  const [loading, setLoading] = useState<string | null>(null);

  async function handleCheckout(plan: PlanTier) {
    if (plan === "free") return;
    if (plan === "enterprise") {
      window.location.href = "/enterprise";
      return;
    }

    setLoading(plan);
    try {
      const res = await fetch("/api/stripe/checkout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan, interval }),
      });
      const data = await res.json();
      if (data.url) {
        window.location.href = data.url;
      } else {
        alert(data.error || "Failed to create checkout session");
      }
    } catch {
      alert("Something went wrong. Please try again.");
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-6xl mx-auto px-4 py-20">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">
            Simple, transparent pricing
          </h1>
          <p className="text-gray-400 text-lg max-w-2xl mx-auto">
            Start free. Upgrade when your GPU fleet needs enterprise-grade cost
            intelligence.
          </p>

          {/* Interval toggle */}
          <div className="flex items-center justify-center gap-3 mt-8">
            <button
              onClick={() => setInterval("monthly")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                interval === "monthly"
                  ? "bg-white text-gray-900"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setInterval("yearly")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                interval === "yearly"
                  ? "bg-white text-gray-900"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              Yearly{" "}
              <span className="text-green-400 text-xs ml-1">Save 20%</span>
            </button>
          </div>
        </div>

        {/* Pricing cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-16">
          {TIERS.map((tier) => {
            const display = PLAN_DISPLAY[tier];
            const price =
              interval === "monthly"
                ? display.price_monthly
                : Math.round(display.price_yearly / 12);
            const isPro = tier === "pro";

            return (
              <div
                key={tier}
                className={`rounded-2xl border p-8 flex flex-col ${
                  isPro
                    ? "border-blue-500 bg-gray-900/80 ring-1 ring-blue-500/50"
                    : "border-gray-800 bg-gray-900/40"
                }`}
              >
                {isPro && (
                  <div className="text-blue-400 text-xs font-semibold uppercase tracking-wider mb-2">
                    Most Popular
                  </div>
                )}
                <h2 className="text-xl font-bold">{display.name}</h2>
                <p className="text-gray-400 text-sm mt-1 mb-6">
                  {display.tagline}
                </p>

                <div className="mb-6">
                  {tier === "enterprise" ? (
                    <div className="text-3xl font-bold">Custom</div>
                  ) : (
                    <>
                      <span className="text-4xl font-bold">${price}</span>
                      <span className="text-gray-400 text-sm ml-1">
                        /mo
                        {interval === "yearly" && tier !== "free"
                          ? " (billed yearly)"
                          : ""}
                      </span>
                    </>
                  )}
                </div>

                <button
                  onClick={() => handleCheckout(tier)}
                  disabled={loading !== null}
                  className={`w-full py-3 rounded-lg font-medium text-sm transition ${
                    tier === "free"
                      ? "bg-gray-800 text-gray-300 hover:bg-gray-700"
                      : isPro
                      ? "bg-blue-600 text-white hover:bg-blue-500"
                      : "bg-white text-gray-900 hover:bg-gray-100"
                  } disabled:opacity-50`}
                >
                  {loading === tier
                    ? "Redirecting..."
                    : tier === "free"
                    ? "Get Started"
                    : tier === "enterprise"
                    ? "Book a Demo"
                    : "Upgrade to Pro"}
                </button>

                <ul className="mt-8 space-y-3 text-sm flex-1">
                  {FEATURE_ROWS.map((row) => {
                    const val = PLAN_LIMITS[tier][row.key];
                    const display = formatValue(val, row.format);
                    const isEnabled =
                      typeof val === "boolean" ? val : val !== 0;

                    return (
                      <li
                        key={row.key}
                        className={`flex items-center justify-between ${
                          isEnabled ? "text-gray-200" : "text-gray-600"
                        }`}
                      >
                        <span>{row.label}</span>
                        <span className="font-medium">{display}</span>
                      </li>
                    );
                  })}
                </ul>
              </div>
            );
          })}
        </div>

        {/* FAQ */}
        <div className="max-w-3xl mx-auto">
          <h2 className="text-2xl font-bold text-center mb-8">
            Frequently Asked Questions
          </h2>
          <div className="space-y-6">
            <FaqItem
              q="Can I change plans later?"
              a="Yes. Upgrade or downgrade anytime from your dashboard settings. When upgrading, you'll be charged a prorated amount. When downgrading, your current plan stays active until the end of the billing period."
            />
            <FaqItem
              q="What happens when I hit a limit?"
              a="You'll receive a clear error message telling you which limit you've reached and which plan to upgrade to. Your existing data and integrations are never affected."
            />
            <FaqItem
              q="Do you offer a free trial of Pro?"
              a="Every new account starts with a 14-day trial of Pro features. No credit card required."
            />
            <FaqItem
              q="What's included in Enterprise?"
              a="Everything in Pro plus unlimited teams/webhooks/exports, 365-day data retention, SLA dashboard, priority support, SSO/SAML, and custom integrations. Book a demo and we'll tailor it to your needs."
            />
          </div>
        </div>

        {/* Back link */}
        <div className="text-center mt-16">
          <Link
            href="/dashboard"
            className="text-gray-400 hover:text-white text-sm transition"
          >
            ← Back to Dashboard
          </Link>
        </div>
      </div>
    </div>
  );
}

function FaqItem({ q, a }: { q: string; a: string }) {
  return (
    <div className="border-b border-gray-800 pb-6">
      <h3 className="font-medium text-gray-200 mb-2">{q}</h3>
      <p className="text-gray-400 text-sm">{a}</p>
    </div>
  );
}
