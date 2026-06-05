"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface PercentileData {
  top_pct: number;
  raw_percentile: number;
  user_value: number;
  peer_count: number;
  gpu_arch: string;
  model_tag: string;
  kwh_per_1m_tokens: number | null;
  framework_tag: string | null;
}

type ApiResponse =
  | ({ opt_in_required: true } & Partial<PercentileData>)
  | ({ no_data: true } & Partial<PercentileData>)
  | ({ insufficient_data: true; peer_count: number } & Partial<PercentileData>)
  | PercentileData;

function kwhPerGpuHr(jPerGpuHr: number): string {
  return (jPerGpuHr / 3_600_000).toFixed(3);
}

function handleShare(data: PercentileData) {
  const metric =
    data.kwh_per_1m_tokens != null
      ? `${data.kwh_per_1m_tokens.toFixed(4)} kWh/1M tokens`
      : `${kwhPerGpuHr(data.user_value)} kWh/GPU-hr`;
  const text = `My ${data.model_tag} on ${data.gpu_arch} uses ${metric} — top ${data.top_pct}% on the NemulAI Green AI Index 🌱 nemulai.com/benchmarks #GreenAI`;
  window.open(
    `https://x.com/intent/tweet?text=${encodeURIComponent(text)}`,
    "_blank",
    "noopener"
  );
}

export default function BenchmarkPercentileCard() {
  const [data, setData] = useState<PercentileData | null>(null);
  const [optInRequired, setOptInRequired] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function fetchPercentile() {
      try {
        const res = await fetch("/api/benchmarks/percentile");
        if (!res.ok) return;
        const json: ApiResponse = await res.json();

        if (cancelled) return;

        if ("opt_in_required" in json && json.opt_in_required) {
          setOptInRequired(true);
          return;
        }

        if ("no_data" in json && json.no_data) {
          // Seed first submission, then refetch once
          await fetch("/api/benchmarks/submit", { method: "POST" });
          if (cancelled) return;
          const res2 = await fetch("/api/benchmarks/percentile");
          if (!res2.ok || cancelled) return;
          const json2: ApiResponse = await res2.json();
          if ("top_pct" in json2) setData(json2 as PercentileData);
          return;
        }

        if ("insufficient_data" in json) return; // Not enough peers yet

        if ("top_pct" in json) {
          setData(json as PercentileData);
        }
      } catch {
        // Card won't render if benchmark data unavailable
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchPercentile();
    return () => { cancelled = true; };
  }, []);

  if (loading) return null;

  if (optInRequired) {
    return (
      <div className="border border-neutral-800 bg-neutral-950 rounded-lg p-4">
        <p className="text-xs text-neutral-500 mb-1">Energy Efficiency</p>
        <Link
          href="/dashboard/settings"
          className="text-sm text-blue-400 hover:text-blue-300 underline"
        >
          Enable benchmarking →
        </Link>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="border border-neutral-800 bg-neutral-950 rounded-lg p-4">
      <div className="flex items-center justify-between mb-1">
        <p className="text-xs text-neutral-500">Energy Efficiency</p>
        <button
          onClick={() => handleShare(data)}
          className="text-xs px-2 py-0.5 rounded border border-neutral-700 text-neutral-400 hover:border-green-700 hover:text-green-400 transition-colors"
        >
          Share →
        </button>
      </div>
      <p className="text-2xl font-bold text-green-400">
        Top {data.top_pct}%
      </p>
      <p className="text-xs text-neutral-400 mt-0.5">
        vs {data.peer_count} peers
        {data.framework_tag && data.framework_tag !== "unknown"
          ? ` · ${data.framework_tag}`
          : ""}
      </p>
      <p className="text-xs text-neutral-500 mt-2 truncate">
        {data.gpu_arch} · {data.model_tag}
      </p>
      <p className="text-xs text-neutral-500">
        {kwhPerGpuHr(data.user_value)} kWh/GPU-hr
      </p>
      {data.kwh_per_1m_tokens != null && (
        <p className="text-xs text-cyan-500 mt-0.5">
          {data.kwh_per_1m_tokens.toFixed(6)} kWh/1M tokens
        </p>
      )}
      <Link
        href="/benchmarks"
        className="text-xs text-neutral-600 hover:text-neutral-400 mt-2 block underline"
      >
        View Green AI Index →
      </Link>
    </div>
  );
}
