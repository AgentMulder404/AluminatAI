"use client";

import Link from "next/link";

const FEATURES = [
  {
    title: "Fleet-Wide Learning Agent",
    description:
      "The agent learns patterns across your entire fleet, not just individual machines. Cross-GPU optimization recommendations improve over time.",
    icon: "🧠",
  },
  {
    title: "Unlimited Teams & Chargeback",
    description:
      "Full RBAC and per-team GPU cost chargeback across your entire organization. No caps on teams, members, or budgets.",
    icon: "👥",
  },
  {
    title: "365-Day Data Retention",
    description:
      "A full year of GPU metrics, cost history, and audit logs for compliance.",
    icon: "📊",
  },
  {
    title: "SLA Dashboard",
    description:
      "Real-time SLA monitoring with uptime guarantees and incident tracking.",
    icon: "🛡️",
  },
  {
    title: "SSO / SAML Integration",
    description:
      "Single sign-on with your identity provider. SCIM provisioning supported.",
    icon: "🔐",
  },
  {
    title: "Priority Support",
    description:
      "Dedicated Slack channel, <4hr response SLA, and onboarding assistance.",
    icon: "⚡",
  },
  {
    title: "Custom Integrations",
    description:
      "Webhooks, Prometheus export, Grafana dashboards, and custom reporting.",
    icon: "🔗",
  },
];

export default function EnterprisePage() {
  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-6xl mx-auto px-6 py-20">
        {/* Back link */}
        <Link
          href="/pricing"
          className="text-sm text-neutral-400 hover:text-neutral-200 mb-8 inline-block"
        >
          ← Back to Pricing
        </Link>

        {/* Hero */}
        <div className="text-center mb-16">
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight mb-4">
            Enterprise GPU
            <br />
            Cost Intelligence
          </h1>
          <p className="text-lg text-neutral-400 max-w-2xl mx-auto mb-4">
            Fleet-wide GPU cost optimization for teams running 100+ GPUs.
            Self-learning agents, compliance-grade retention, and dedicated support.
          </p>
          <p className="text-sm text-green-400 font-medium">
            $15/GPU/mo — billed monthly or annually with 20% discount
          </p>
        </div>

        {/* Feature grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {FEATURES.map((feat) => (
            <div
              key={feat.title}
              className="bg-neutral-900 border border-neutral-800 rounded-xl p-6"
            >
              <span className="text-2xl mb-3 block">{feat.icon}</span>
              <h3 className="text-sm font-semibold text-neutral-100 mb-1">
                {feat.title}
              </h3>
              <p className="text-sm text-neutral-400">{feat.description}</p>
            </div>
          ))}
        </div>

        {/* Calendly booking section */}
        <div className="text-center mb-16">
          <h2 className="text-2xl font-bold mb-3">Book a 30-minute demo</h2>
          <p className="text-neutral-400 mb-6 max-w-xl mx-auto">
            We&apos;ll walk through autonomous optimization for your fleet, answer
            procurement questions, and scope a custom Swarm contract.
          </p>
          <a
            href="https://calendly.com/nemulai/enterprise-demo"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block px-8 py-4 bg-green-600 hover:bg-green-500 rounded-lg font-semibold text-lg transition-colors"
          >
            Schedule on Calendly
          </a>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-neutral-500 text-sm">
            Prefer email?{" "}
            <a
              href="mailto:kevin.mello8@gmail.com"
              className="text-indigo-400 hover:text-indigo-300"
            >
              kevin.mello8@gmail.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
