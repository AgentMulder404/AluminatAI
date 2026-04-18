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
            <span className="text-green-400">Aluminat</span>AI
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
            <div className="inline-block px-4 py-1.5 bg-green-500/10 border border-green-500/20 rounded-full text-green-400 text-sm mb-6">
              Hackathon Demo — Apify Agent Pipeline
            </div>
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
        AluminatAI — Built for the Apify Hackathon 2026
      </footer>
    </div>
  );
}
