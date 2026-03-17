"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

interface LeaderboardRow {
  gpu_arch: string;
  model_tag: string;
  sample_count: number;
  p25_j_per_gpu_hour: number;
  p50_j_per_gpu_hour: number;
  p75_j_per_gpu_hour: number;
  best_j_per_gpu_hour: number;
  best_precision_tag: string;
  best_kwh_per_1m_tokens: number | null;
  p50_kwh_per_1m_tokens: number | null;
  top_framework_tag: string | null;
}

type View = "gpu_hr" | "token";
type SortKey = "p50" | "best" | "p75" | "sample_count" | "model_tag" | "gpu_arch";

function jToKwh(j: number): string {
  return (j / 3_600_000).toFixed(3);
}

function SortArrow({ active, asc }: { active: boolean; asc: boolean }) {
  if (!active) return <span className="text-neutral-700"> ↕</span>;
  return <span className="text-neutral-400"> {asc ? "↑" : "↓"}</span>;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  function copy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }
  return (
    <button
      onClick={copy}
      className="text-xs px-2 py-1 rounded bg-neutral-800 hover:bg-neutral-700 text-neutral-400 hover:text-neutral-200 transition-colors"
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

function rowColor(index: number, total: number): string {
  const third = total / 3;
  if (index < third) return "text-green-400";
  if (index < third * 2) return "text-amber-400";
  return "text-neutral-400";
}

export default function BenchmarksPage() {
  const [rows, setRows] = useState<LeaderboardRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<View>("gpu_hr");
  const [sortKey, setSortKey] = useState<SortKey>("p50");
  const [sortAsc, setSortAsc] = useState(true);
  const tooltipRef = useRef<HTMLSpanElement>(null);
  const [tooltipOpen, setTooltipOpen] = useState(false);

  useEffect(() => {
    fetch("/api/benchmarks/leaderboard")
      .then((r) => (r.ok ? r.json() : []))
      .then((d) => setRows(d as LeaderboardRow[]))
      .catch(() => setRows([]))
      .finally(() => setLoading(false));
  }, []);

  function handleSort(key: SortKey) {
    if (key === sortKey) setSortAsc((a) => !a);
    else { setSortKey(key); setSortAsc(true); }
  }

  // Filter + sort
  const visible = view === "token"
    ? rows.filter((r) => r.best_kwh_per_1m_tokens != null)
    : rows;

  const sorted = [...visible].sort((a, b) => {
    let av: number | string, bv: number | string;
    switch (sortKey) {
      case "p50":
        av = view === "token" ? (a.p50_kwh_per_1m_tokens ?? Infinity) : a.p50_j_per_gpu_hour;
        bv = view === "token" ? (b.p50_kwh_per_1m_tokens ?? Infinity) : b.p50_j_per_gpu_hour;
        break;
      case "best":
        av = view === "token" ? (a.best_kwh_per_1m_tokens ?? Infinity) : a.best_j_per_gpu_hour;
        bv = view === "token" ? (b.best_kwh_per_1m_tokens ?? Infinity) : b.best_j_per_gpu_hour;
        break;
      case "p75": av = a.p75_j_per_gpu_hour; bv = b.p75_j_per_gpu_hour; break;
      case "sample_count": av = a.sample_count; bv = b.sample_count; break;
      case "model_tag": av = a.model_tag; bv = b.model_tag; break;
      case "gpu_arch": av = a.gpu_arch; bv = b.gpu_arch; break;
    }
    if (typeof av === "number" && typeof bv === "number") return sortAsc ? av - bv : bv - av;
    return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  });

  const CLI_CMD = "aluminatiai benchmark --model-tag llama-3-8b --throughput 1500 --upload";

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Hero */}
      <div className="max-w-5xl mx-auto px-6 pt-16 pb-10">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-3xl">🌱</span>
          <h1 className="text-4xl font-bold tracking-tight">Green AI Index</h1>
        </div>
        <p className="text-lg text-neutral-300 max-w-2xl">
          The world&apos;s first open leaderboard for AI energy efficiency.
        </p>
        <p className="mt-1 text-sm text-neutral-500 max-w-2xl">
          kWh per 1M tokens · Community-sourced · Anonymous · ≥10 submissions per group
        </p>
        {/* Compliance chips */}
        <div className="flex flex-wrap gap-2 mt-4">
          {["EU AI Act", "GHG Protocol", "SB 253"].map((label) => (
            <span
              key={label}
              className="text-xs px-2.5 py-1 rounded-full border border-neutral-700 text-neutral-400"
            >
              {label}
            </span>
          ))}
        </div>
      </div>

      {/* Submit CTA strip */}
      <div className="max-w-5xl mx-auto px-6 pb-10">
        <div className="border border-neutral-800 rounded-lg p-5 bg-neutral-900/50">
          <p className="text-xs text-neutral-500 mb-2 uppercase tracking-wider">
            Submit to the leaderboard
          </p>
          <div className="flex items-center gap-3 flex-wrap">
            <code className="flex-1 text-xs font-mono bg-neutral-950 border border-neutral-800 rounded px-3 py-2 text-green-400 min-w-0 overflow-x-auto whitespace-nowrap">
              {CLI_CMD}
            </code>
            <CopyButton text={CLI_CMD} />
          </div>
          <p className="mt-2 text-xs text-neutral-600">
            <span
              ref={tooltipRef}
              className="relative cursor-help underline decoration-dotted text-neutral-500"
              onClick={() => setTooltipOpen((o) => !o)}
            >
              What is throughput?
              {tooltipOpen && (
                <span className="absolute left-0 top-5 z-10 w-64 bg-neutral-800 border border-neutral-700 rounded p-3 text-xs text-neutral-300 shadow-lg">
                  Tokens per second measured while your model is running inference.
                  Pass the value from your framework&apos;s benchmark output (e.g. vLLM
                  throughput_tokens_s). Used to compute kWh per 1M tokens.
                </span>
              )}
            </span>
          </p>
        </div>
      </div>

      {/* View toggle */}
      <div className="max-w-5xl mx-auto px-6 pb-4 flex items-center gap-2">
        <button
          onClick={() => setView("gpu_hr")}
          className={`text-sm px-4 py-1.5 rounded-full border transition-colors ${
            view === "gpu_hr"
              ? "border-green-600 text-green-400 bg-green-950/40"
              : "border-neutral-700 text-neutral-500 hover:border-neutral-500"
          }`}
        >
          kWh / GPU-hr
        </button>
        <button
          onClick={() => setView("token")}
          className={`text-sm px-4 py-1.5 rounded-full border transition-colors ${
            view === "token"
              ? "border-green-600 text-green-400 bg-green-950/40"
              : "border-neutral-700 text-neutral-500 hover:border-neutral-500"
          }`}
        >
          kWh / 1M Tokens
        </button>
      </div>

      {/* Leaderboard table */}
      <div className="max-w-5xl mx-auto px-6 pb-16">
        {loading ? (
          <p className="text-sm text-neutral-500">Loading…</p>
        ) : sorted.length === 0 ? (
          <div className="border border-neutral-800 rounded-lg p-8 text-center">
            {view === "token" ? (
              <>
                <p className="text-neutral-400 text-sm">
                  No kWh/1M token data yet.
                </p>
                <p className="text-neutral-600 text-xs mt-2">
                  Submit with <code className="text-green-500">--throughput</code> to populate this view.
                </p>
              </>
            ) : (
              <>
                <p className="text-neutral-400 text-sm">
                  No leaderboard data yet. At least 10 submissions per model×GPU
                  pair are required before results appear.
                </p>
                <p className="text-neutral-600 text-xs mt-2">
                  Enable benchmarking in your{" "}
                  <Link href="/dashboard/settings" className="underline text-neutral-500 hover:text-neutral-300">
                    dashboard settings
                  </Link>{" "}
                  to contribute.
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-neutral-800">
            <table className="w-full text-xs text-neutral-300">
              <thead className="bg-neutral-900">
                <tr>
                  <th className="py-2 px-3 text-left text-neutral-500 font-medium w-8">#</th>
                  {[
                    { key: "model_tag" as SortKey, label: "Model", numeric: false },
                    { key: "gpu_arch" as SortKey, label: "GPU", numeric: false },
                    { key: "p50" as SortKey, label: view === "token" ? "Median kWh/1M tok" : "Median kWh/GPU-hr", numeric: true },
                    { key: "best" as SortKey, label: "Best", numeric: true },
                    ...(view === "gpu_hr"
                      ? [{ key: "p75" as SortKey, label: "p75", numeric: true }]
                      : []),
                    { key: "sample_count" as SortKey, label: "Submissions", numeric: true },
                  ].map((c) => (
                    <th
                      key={c.key}
                      onClick={() => handleSort(c.key)}
                      className={`py-2 px-3 font-medium text-neutral-400 cursor-pointer select-none hover:text-neutral-200 ${
                        c.numeric ? "text-right" : "text-left"
                      }`}
                    >
                      {c.label}
                      <SortArrow active={sortKey === c.key} asc={sortAsc} />
                    </th>
                  ))}
                  <th className="py-2 px-3 text-left text-neutral-400 font-medium">Framework</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((row, i) => {
                  const medal = i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : null;
                  const medianColor = rowColor(i, sorted.length);
                  const medianVal =
                    view === "token"
                      ? (row.p50_kwh_per_1m_tokens != null
                          ? row.p50_kwh_per_1m_tokens.toFixed(6)
                          : "—")
                      : jToKwh(row.p50_j_per_gpu_hour);
                  const bestVal =
                    view === "token"
                      ? (row.best_kwh_per_1m_tokens != null
                          ? row.best_kwh_per_1m_tokens.toFixed(6)
                          : "—")
                      : jToKwh(row.best_j_per_gpu_hour);

                  return (
                    <tr
                      key={`${row.gpu_arch}-${row.model_tag}`}
                      className={`border-t border-neutral-800 ${
                        i % 2 === 0 ? "bg-neutral-950" : "bg-neutral-900/40"
                      }`}
                    >
                      <td className="py-2 px-3 text-neutral-600">
                        {medal ?? i + 1}
                      </td>
                      <td className="py-2 px-3 font-medium">{row.model_tag}</td>
                      <td className="py-2 px-3 text-neutral-400">{row.gpu_arch}</td>
                      <td className={`py-2 px-3 text-right font-mono ${medianColor}`}>
                        {medianVal}
                      </td>
                      <td className="py-2 px-3 text-right font-mono text-green-400">
                        {bestVal}
                      </td>
                      {view === "gpu_hr" && (
                        <td className="py-2 px-3 text-right font-mono text-neutral-500">
                          {jToKwh(row.p75_j_per_gpu_hour)}
                        </td>
                      )}
                      <td className="py-2 px-3 text-right text-neutral-500">
                        {row.sample_count}
                      </td>
                      <td className="py-2 px-3 text-neutral-500">
                        {row.top_framework_tag ?? row.best_precision_tag}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Methodology footer */}
        <div className="mt-6 text-xs text-neutral-600 space-y-1">
          <p>
            All submissions are anonymized using HMAC-SHA256. Individual users are
            never identified. Groups with fewer than 10 submissions are excluded.
          </p>
          <p>
            <Link href="/dashboard/settings" className="underline hover:text-neutral-400">
              Enable benchmarking in your dashboard
            </Link>{" "}
            to contribute your data.
          </p>
        </div>
      </div>
    </div>
  );
}
