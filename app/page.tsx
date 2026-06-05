"use client";

import Link from "next/link";
import { useState, useEffect } from "react";

const FEATURES = [
  { title: "GPU Cost Attribution", desc: "Track energy spend per job, model, and team in real-time", icon: "⚡" },
  { title: "Waste Detection", desc: "Identify idle GPUs and underutilized workloads automatically", icon: "🔍" },
  { title: "Carbon Tracking", desc: "Monitor CO₂ emissions with grid-aware carbon intensity data", icon: "🌱" },
  { title: "Prospect Pipeline", desc: "Apify-powered agents discover and enrich leads automatically", icon: "🤖" },
  { title: "Budget Alerts", desc: "Set spend limits and get notified before you blow your budget", icon: "💰" },
  { title: "Green AI Index", desc: "Benchmark your efficiency against anonymous industry peers", icon: "📊" },
];

export default function Home() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  return (
    <div className="min-h-screen bg-neutral-950 text-white">
      {/* Nav */}
      <nav className="border-b border-neutral-800 px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="text-xl font-bold tracking-tight">
            <span className="text-green-400">Nemul</span>AI
          </div>
          <div className="flex items-center gap-6 text-sm text-neutral-400">
            <Link href="/pricing" className="hover:text-white transition-colors">Pricing</Link>
            <Link href="/benchmarks" className="hover:text-white transition-colors">Benchmarks</Link>
            <Link href="/login" className="px-4 py-2 bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors">
              Sign In
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="px-6 py-24 text-center">
        <div className="max-w-4xl mx-auto">
          <div className={`transition-all duration-700 ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
<h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              Know what your
              <br />
              <span className="text-green-400">GPUs actually cost</span>
            </h1>
            <p className="text-xl text-neutral-400 mb-8 max-w-2xl mx-auto">
              Real-time energy monitoring, cost attribution, and carbon tracking for every GPU workload.
              Plus AI-powered prospect discovery to find your next customers.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                href="/admin/outreach"
                className="px-8 py-3 bg-green-600 hover:bg-green-500 rounded-lg font-semibold text-lg transition-colors"
              >
                Try Prospect Pipeline
              </Link>
              <Link
                href="/dashboard"
                className="px-8 py-3 bg-neutral-800 hover:bg-neutral-700 rounded-lg font-semibold text-lg transition-colors"
              >
                View Dashboard
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">What we built</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURES.map((f) => (
              <div
                key={f.title}
                className="bg-neutral-900 border border-neutral-800 rounded-xl p-6 hover:border-green-500/30 transition-colors"
              >
                <div className="text-3xl mb-3">{f.icon}</div>
                <h3 className="text-lg font-semibold mb-2">{f.title}</h3>
                <p className="text-neutral-400 text-sm">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Apify Agent Pipeline */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <div className="inline-block px-3 py-1 bg-purple-500/10 border border-purple-500/20 rounded-full text-purple-400 text-xs font-medium mb-4">
              Powered by Apify
            </div>
            <h2 className="text-3xl font-bold mb-3">Two-Agent Prospect Pipeline</h2>
            <p className="text-neutral-400 max-w-2xl mx-auto">
              We built two autonomous agents that use Apify actors to find, enrich, verify, and load
              potential customers into our CRM — all from a single button click.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-12">
            {/* Agent 1 */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-green-500/10 border border-green-500/20 flex items-center justify-center text-green-400 font-bold">
                    1
                  </div>
                  <h3 className="text-xl font-bold">Discovery Agent</h3>
                </div>
                <p className="text-neutral-400 text-sm mb-6">
                  Finds AI/ML companies on LinkedIn and extracts decision-maker contact info.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-green-500/10 flex items-center justify-center text-green-400 text-xs mt-0.5 shrink-0">a</div>
                    <div>
                      <div className="text-sm font-medium">LinkedIn Company Search</div>
                      <div className="text-xs text-neutral-500 mt-0.5">
                        <span className="text-neutral-600">Actor:</span> harvestapi/linkedin-company-search
                      </div>
                      <div className="text-xs text-neutral-400 mt-1">
                        Searches by keywords like &quot;GPU cloud&quot;, &quot;AI infrastructure&quot;, &quot;deep learning&quot;.
                        Filters by company size and location. Returns company profiles with website, industry, and employee count.
                      </div>
                    </div>
                  </div>
                  <div className="border-l-2 border-dashed border-neutral-700 ml-3 h-4" />
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-green-500/10 flex items-center justify-center text-green-400 text-xs mt-0.5 shrink-0">b</div>
                    <div>
                      <div className="text-sm font-medium">Decision Maker Email Finder</div>
                      <div className="text-xs text-neutral-500 mt-0.5">
                        <span className="text-neutral-600">Actor:</span> snipercoder/decision-maker-email-finder
                      </div>
                      <div className="text-xs text-neutral-400 mt-1">
                        Takes each company&apos;s domain, finds CEO/founder/owner contacts with name, email, title,
                        phone, and LinkedIn URL. Up to 3 leads per company.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Agent 2 */}
            <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full -translate-y-1/2 translate-x-1/2" />
              <div className="relative">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-lg bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-400 font-bold">
                    2
                  </div>
                  <h3 className="text-xl font-bold">Verification + Load Agent</h3>
                </div>
                <p className="text-neutral-400 text-sm mb-6">
                  Validates emails and loads verified prospects into our Supabase CRM.
                </p>
                <div className="space-y-4">
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400 text-xs mt-0.5 shrink-0">a</div>
                    <div>
                      <div className="text-sm font-medium">Email Verification</div>
                      <div className="text-xs text-neutral-500 mt-0.5">
                        <span className="text-neutral-600">Actor:</span> snipercoder/email-validator
                      </div>
                      <div className="text-xs text-neutral-400 mt-1">
                        Checks DNS, MX records, and SMTP for each email. Marks valid vs. invalid so we
                        only reach out to real addresses. ~$1/1K verifications.
                      </div>
                    </div>
                  </div>
                  <div className="border-l-2 border-dashed border-neutral-700 ml-3 h-4" />
                  <div className="flex items-start gap-3">
                    <div className="w-6 h-6 rounded-full bg-blue-500/10 flex items-center justify-center text-blue-400 text-xs mt-0.5 shrink-0">b</div>
                    <div>
                      <div className="text-sm font-medium">CRM Load</div>
                      <div className="text-xs text-neutral-500 mt-0.5">
                        <span className="text-neutral-600">Storage:</span> Supabase PostgreSQL
                      </div>
                      <div className="text-xs text-neutral-400 mt-1">
                        Upserts prospects into our database with dedup on domain + email.
                        Company info, contact details, verification status — ready for outreach.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Pipeline Flow */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-8">
            <h3 className="text-lg font-semibold mb-6 text-center">Pipeline Flow</h3>
            <div className="flex items-center justify-between gap-2 max-w-4xl mx-auto">
              {[
                { label: "Keywords", sub: "GPU cloud, AI infra...", color: "green" },
                { label: "LinkedIn Search", sub: "harvestapi actor", color: "green" },
                { label: "Email Enrichment", sub: "snipercoder actor", color: "green" },
                { label: "Verification", sub: "email-validator", color: "blue" },
                { label: "CRM Load", sub: "Supabase upsert", color: "blue" },
              ].map((step, i) => (
                <div key={step.label} className="flex items-center gap-2 flex-1">
                  <div className="text-center flex-1">
                    <div className={`w-full py-3 px-2 rounded-lg border text-xs font-medium ${
                      step.color === "green"
                        ? "bg-green-500/10 border-green-500/20 text-green-400"
                        : "bg-blue-500/10 border-blue-500/20 text-blue-400"
                    }`}>
                      {step.label}
                    </div>
                    <div className="text-[10px] text-neutral-500 mt-1">{step.sub}</div>
                  </div>
                  {i < 4 && (
                    <div className="text-neutral-600 shrink-0">&#8594;</div>
                  )}
                </div>
              ))}
            </div>
            <div className="flex justify-center gap-8 mt-6 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-green-500/20 border border-green-500/30" />
                <span className="text-neutral-500">Agent 1: Discovery</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded bg-blue-500/20 border border-blue-500/30" />
                <span className="text-neutral-500">Agent 2: Verify + Load</span>
              </div>
            </div>
          </div>

          <div className="text-center mt-8">
            <Link
              href="/admin/outreach"
              className="inline-block px-8 py-3 bg-green-600 hover:bg-green-500 rounded-lg font-semibold transition-colors"
            >
              Try the Pipeline Live
            </Link>
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">Architecture</h2>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800">
              <div className="text-green-400 font-semibold mb-2">Agent (Python)</div>
              <p className="text-neutral-400">NVML probe, WAL, batched upload. Runs as systemd/K8s DaemonSet.</p>
            </div>
            <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800">
              <div className="text-green-400 font-semibold mb-2">API (Next.js)</div>
              <p className="text-neutral-400">Ingest, dashboard, cost, carbon, benchmarks, prospect agents.</p>
            </div>
            <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800">
              <div className="text-green-400 font-semibold mb-2">Apify Pipeline</div>
              <p className="text-neutral-400">LinkedIn search, email enrichment, verification, CRM load.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-neutral-800 px-6 py-8 text-center text-neutral-500 text-sm">
        NemulAI
      </footer>
    </div>
  );
}
