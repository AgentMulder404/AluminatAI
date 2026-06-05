"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface LeaderboardRow {
  gpu_arch: string;
  model_tag: string;
  grid_zone: string;
  sample_count: number;
  p50_co2e_g_per_gpu_hour: number;
  best_co2e_g_per_gpu_hour: number;
  avg_carbon_g_per_kwh: number;
}

type SortKey = keyof Pick<
  LeaderboardRow,
  "p50_co2e_g_per_gpu_hour" | "best_co2e_g_per_gpu_hour" | "avg_carbon_g_per_kwh" | "sample_count"
>;

function fmt(g: number): string {
  if (g < 1000) return `${g.toFixed(1)}g`;
  return `${(g / 1000).toFixed(2)}kg`;
}

export default function CarbonLeaderboardPage() {
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortKey, setSortKey] = useState<SortKey>("p50_co2e_g_per_gpu_hour");
  const [asc, setAsc] = useState(true);

  useEffect(() => {
    fetch("/api/carbon/leaderboard")
      .then((r) => (r.ok ? r.json() : []))
      .then((d) => setRows(Array.isArray(d) ? d : []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  function handleSort(key: SortKey) {
    if (key === sortKey) setAsc((a) => !a);
    else { setSortKey(key); setAsc(true); }
  }

  const sorted = [...rows].sort((a, b) => {
    const diff = a[sortKey] - b[sortKey];
    return asc ? diff : -diff;
  });

  function SortHeader({ k, label }: { k: SortKey; label: string }) {
    const active = k === sortKey;
    return (
      <th
        className="px-4 py-2 text-right cursor-pointer select-none hover:text-neutral-200 transition-colors"
        onClick={() => handleSort(k)}
      >
        {label}
        <span className="ml-1 opacity-50">{active ? (asc ? "↑" : "↓") : "↕"}</span>
      </th>
    );
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Header */}
      <div className="border-b border-neutral-800 px-6 py-8">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl">🌱</span>
            <h1 className="text-2xl font-bold">Carbon Efficiency Leaderboard</h1>
          </div>
          <p className="text-neutral-400 text-sm max-w-2xl">
            Real gCO₂e per GPU-hour across models, hardware, and grid zones — computed from
            measured energy × live carbon intensity (Electricity Maps). Anonymous, privacy-preserving.
          </p>
          <div className="flex gap-4 mt-4 text-xs text-neutral-500">
            <span>GHG Protocol Scope 2</span>
            <span>·</span>
            <span>EU AI Act ready</span>
            <span>·</span>
            <span>SB 253 aligned</span>
            <span>·</span>
            <Link href="/dashboard/settings" className="text-green-400 hover:text-green-300">
              Share your data →
            </Link>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {loading ? (
          <div className="space-y-2">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-10 bg-neutral-900 rounded animate-pulse" />
            ))}
          </div>
        ) : sorted.length === 0 ? (
          <div className="text-center py-24 text-neutral-500">
            <p className="text-lg mb-2">No data yet</p>
            <p className="text-sm">
              Be the first —{" "}
              <Link href="/dashboard/settings" className="text-green-400 hover:underline">
                opt in to share your carbon data
              </Link>
            </p>
          </div>
        ) : (
          <div className="border border-neutral-800 rounded-lg overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-neutral-800 text-neutral-400 text-left">
                  <th className="px-4 py-3 font-medium">Model</th>
                  <th className="px-4 py-3 font-medium">GPU</th>
                  <th className="px-4 py-3 font-medium">Zone</th>
                  <SortHeader k="p50_co2e_g_per_gpu_hour" label="Median gCO₂e/GPU-hr" />
                  <SortHeader k="best_co2e_g_per_gpu_hour" label="Best" />
                  <SortHeader k="avg_carbon_g_per_kwh" label="Avg g/kWh" />
                  <SortHeader k="sample_count" label="Submissions" />
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, i) => (
                  <tr
                    key={i}
                    className="border-b border-neutral-900 hover:bg-neutral-900/50 transition-colors"
                  >
                    <td className="px-4 py-2 text-neutral-300">{row.model_tag}</td>
                    <td className="px-4 py-2 text-neutral-400">{row.gpu_arch}</td>
                    <td className="px-4 py-2 font-mono text-neutral-500">{row.grid_zone}</td>
                    <td className="px-4 py-2 text-right text-neutral-200 font-medium">
                      {fmt(row.p50_co2e_g_per_gpu_hour)}
                    </td>
                    <td className="px-4 py-2 text-right text-green-400">
                      {fmt(row.best_co2e_g_per_gpu_hour)}
                    </td>
                    <td className="px-4 py-2 text-right text-neutral-400">
                      {row.avg_carbon_g_per_kwh.toFixed(0)}
                    </td>
                    <td className="px-4 py-2 text-right text-neutral-500">
                      {row.sample_count}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <p className="mt-6 text-xs text-neutral-600 text-center">
          Groups with fewer than 5 submissions are hidden to protect privacy.
          <br />
          Energy measured by NemulAI agent · Intensity from Electricity Maps (lifecycle factors)
        </p>
      </div>
    </div>
  );
}
