"use client";

import Link from "next/link";

const FEATURES = [
  {
    title: "Unlimited Teams & Members",
    description:
      "Full RBAC across your entire org. No caps on teams, members, or budgets.",
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
          <p className="text-lg text-neutral-400 max-w-2xl mx-auto">
            Full visibility into GPU energy costs across your fleet. Built for
            teams running 100+ GPUs with compliance, security, and dedicated
            support.
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

        {/* Cal.com booking section */}
        <div className="text-center mb-16">
          <h2 className="text-2xl font-bold mb-3">Book a Demo</h2>
          <p className="text-neutral-400 mb-8">
            30-minute call to discuss your GPU fleet, pricing, and setup.
          </p>
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8 max-w-3xl mx-auto">
            <iframe
              src="https://cal.com/aluminatai/enterprise-demo"
              width="100%"
              height="600"
              frameBorder="0"
              className="rounded-lg"
              title="Schedule a demo with AluminatAI"
              sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox"
              loading="lazy"
            />
          </div>
        </div>

        {/* CTA */}
        <div className="text-center">
          <p className="text-neutral-500 text-sm">
            Prefer email?{" "}
            <a
              href="mailto:enterprise@aluminatai.com"
              className="text-indigo-400 hover:text-indigo-300"
            >
              enterprise@aluminatai.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}
