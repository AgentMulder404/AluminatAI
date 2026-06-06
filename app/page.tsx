"use client";

import Link from "next/link";
import { useState, useEffect } from "react";

const FEATURES = [
  {
    title: "Per-Job Cost Attribution",
    desc: "See exactly what each training run costs. Break down spend by job, model, team, or cluster — in real time, not after the invoice.",
    icon: "💰",
  },
  {
    title: "Waste Detection",
    desc: "Idle GPUs are burning cash. NemulAI flags underutilized machines the moment they start wasting money.",
    icon: "🔍",
  },
  {
    title: "Self-Learning Optimizer",
    desc: "The agent learns your workload patterns over time. It starts with basic recommendations and evolves into fleet-wide optimization.",
    icon: "🧠",
  },
  {
    title: "Team Chargeback",
    desc: "Split GPU costs by team with one environment variable. Finance gets the attribution data they've been asking for.",
    icon: "👥",
  },
  {
    title: "Budget Alerts",
    desc: "Set daily, weekly, or monthly spend limits. Get notified in Slack, email, or PagerDuty before you blow your budget.",
    icon: "🔔",
  },
  {
    title: "Green AI Index",
    desc: "Benchmark your GPU efficiency against anonymous industry peers. Useful for ESG reporting and EU AI Act compliance.",
    icon: "🌱",
  },
];

const STEPS = [
  { cmd: "pip install nemulai", label: "Install the agent" },
  { cmd: "export ALUMINATAI_API_KEY=alum_...", label: "Set your API key" },
  { cmd: "nemulai", label: "Start monitoring" },
  { cmd: null, label: "See costs in real time", icon: "📊" },
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
            <a href="https://github.com/AgentMulder404/NemulAI" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a>
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
              Cut your GPU bill.
              <br />
              <span className="text-green-400">Automatically.</span>
            </h1>
            <p className="text-xl text-neutral-400 mb-8 max-w-2xl mx-auto">
              NemulAI&apos;s self-learning agent monitors every GPU workload, finds waste,
              and optimizes power — saving teams 15-40% on compute costs.
            </p>
            <div className="flex items-center justify-center gap-4">
              <Link
                href="/login"
                className="px-8 py-3 bg-green-600 hover:bg-green-500 rounded-lg font-semibold text-lg transition-colors"
              >
                Get Started Free
              </Link>
              <a
                href="https://github.com/AgentMulder404/NemulAI"
                target="_blank"
                rel="noopener noreferrer"
                className="px-8 py-3 bg-neutral-800 hover:bg-neutral-700 rounded-lg font-semibold text-lg transition-colors"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Bar */}
      <section className="px-6 pb-16">
        <div className="max-w-3xl mx-auto flex items-center justify-center gap-8 text-sm text-neutral-500">
          {["Open source", "7-day free trial", "No credit card", "2-minute install"].map((t) => (
            <div key={t} className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500/50" />
              {t}
            </div>
          ))}
        </div>
      </section>

      {/* Problem / Solution */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-6xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-red-500/5 border border-red-500/10 rounded-xl p-8">
              <div className="text-red-400 text-sm font-semibold mb-3 uppercase tracking-wider">The Problem</div>
              <h3 className="text-lg font-bold mb-3">GPU bills are a black box</h3>
              <p className="text-neutral-400 text-sm leading-relaxed">
                You get a monthly invoice with zero attribution. Which jobs are worth it?
                Which GPUs are sitting idle? Nobody knows until it&apos;s too late.
              </p>
            </div>
            <div className="bg-yellow-500/5 border border-yellow-500/10 rounded-xl p-8">
              <div className="text-yellow-400 text-sm font-semibold mb-3 uppercase tracking-wider">The Old Way</div>
              <h3 className="text-lg font-bold mb-3">Static dashboards and spreadsheets</h3>
              <p className="text-neutral-400 text-sm leading-relaxed">
                Grafana queries and manual spreadsheets that tell you what happened
                last month — not what to do about it. No attribution, no optimization.
              </p>
            </div>
            <div className="bg-green-500/5 border border-green-500/10 rounded-xl p-8">
              <div className="text-green-400 text-sm font-semibold mb-3 uppercase tracking-wider">NemulAI</div>
              <h3 className="text-lg font-bold mb-3">Self-learning cost intelligence</h3>
              <p className="text-neutral-400 text-sm leading-relaxed">
                A self-learning agent that watches every GPU, attributes cost per job,
                detects waste in real time, and gets smarter the longer it runs.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-4">Everything you need to cut GPU costs</h2>
          <p className="text-neutral-400 text-center mb-12 max-w-2xl mx-auto">
            From per-job attribution to fleet-wide optimization. Start with visibility, graduate to automation.
          </p>
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

      {/* How It Works */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Up and running in 2 minutes</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {STEPS.map((step, i) => (
              <div key={i} className="text-center">
                <div className="w-10 h-10 rounded-full bg-green-500/10 border border-green-500/20 flex items-center justify-center text-green-400 font-bold mx-auto mb-4">
                  {i + 1}
                </div>
                {step.cmd ? (
                  <code className="text-xs bg-neutral-900 border border-neutral-800 rounded px-2 py-1 text-green-400 block mb-2">
                    {step.cmd}
                  </code>
                ) : (
                  <div className="text-3xl mb-2">{step.icon}</div>
                )}
                <p className="text-neutral-400 text-sm">{step.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Architecture */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-6">Architecture</h2>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800">
              <div className="text-green-400 font-semibold mb-2">Agent (Open Source)</div>
              <p className="text-neutral-400">NVML probe, WAL buffer, batched upload. Runs as systemd, Docker, or K8s DaemonSet.</p>
            </div>
            <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800">
              <div className="text-green-400 font-semibold mb-2">API</div>
              <p className="text-neutral-400">Ingest, cost attribution, waste detection, self-learning recommendations, benchmarks.</p>
            </div>
            <div className="bg-neutral-900 rounded-xl p-6 border border-neutral-800">
              <div className="text-green-400 font-semibold mb-2">Dashboard</div>
              <p className="text-neutral-400">Real-time cost visibility, team chargeback, budget alerts, carbon tracking.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Open Source CTA */}
      <section className="px-6 py-16 border-t border-neutral-800">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold mb-4">The agent is free and open source. Always.</h2>
          <p className="text-neutral-400 mb-8">
            Audit the code, self-host against your own endpoint, or contribute integrations.
            The agent that collects your data will never be paywalled.
          </p>
          <a
            href="https://github.com/AgentMulder404/NemulAI"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-block px-8 py-3 bg-neutral-800 hover:bg-neutral-700 rounded-lg font-semibold transition-colors"
          >
            Star on GitHub
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-neutral-800 px-6 py-12">
        <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8 text-sm">
          <div>
            <div className="font-semibold text-neutral-300 mb-3">Product</div>
            <div className="space-y-2 text-neutral-500">
              <div><Link href="/dashboard" className="hover:text-white transition-colors">Dashboard</Link></div>
              <div><Link href="/benchmarks" className="hover:text-white transition-colors">Benchmarks</Link></div>
              <div><Link href="/carbon" className="hover:text-white transition-colors">Carbon</Link></div>
              <div><Link href="/pricing" className="hover:text-white transition-colors">Pricing</Link></div>
            </div>
          </div>
          <div>
            <div className="font-semibold text-neutral-300 mb-3">Company</div>
            <div className="space-y-2 text-neutral-500">
              <div><Link href="/enterprise" className="hover:text-white transition-colors">Enterprise</Link></div>
              <div><Link href="/legal/security-questionnaire" className="hover:text-white transition-colors">Security</Link></div>
              <div><Link href="/legal/msa" className="hover:text-white transition-colors">Legal</Link></div>
            </div>
          </div>
          <div>
            <div className="font-semibold text-neutral-300 mb-3">Open Source</div>
            <div className="space-y-2 text-neutral-500">
              <div><a href="https://github.com/AgentMulder404/NemulAI" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">GitHub</a></div>
              <div><a href="https://github.com/AgentMulder404/NemulAI#quick-start" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">Install Guide</a></div>
              <div><a href="https://github.com/AgentMulder404/NemulAI#api" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">API Docs</a></div>
            </div>
          </div>
          <div>
            <div className="font-semibold text-neutral-300 mb-3">Connect</div>
            <div className="space-y-2 text-neutral-500">
              <div><a href="https://x.com/NemulAI_Dev" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">X / Twitter</a></div>
              <div><a href="https://linkedin.com/company/nemulai" target="_blank" rel="noopener noreferrer" className="hover:text-white transition-colors">LinkedIn</a></div>
            </div>
          </div>
        </div>
        <div className="max-w-6xl mx-auto mt-8 pt-8 border-t border-neutral-800 text-sm text-neutral-600 text-center">
          2026 NemulAI
        </div>
      </footer>
    </div>
  );
}
