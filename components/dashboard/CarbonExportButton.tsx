"use client";

import { useState } from "react";

type Format = "csv" | "json" | "pdf";
type Range = "7d" | "30d" | "90d";

const RANGE_LABELS: Record<Range, string> = {
  "7d": "Last 7 days",
  "30d": "Last 30 days",
  "90d": "Last 90 days",
};

function getRangeDates(range: Range): { start: string; end: string } {
  const end = new Date();
  const start = new Date();
  const days = range === "7d" ? 7 : range === "30d" ? 30 : 90;
  start.setDate(start.getDate() - days);
  return {
    start: start.toISOString(),
    end: end.toISOString(),
  };
}

export default function CarbonExportButton({ clusterTag }: { clusterTag?: string }) {
  const [format, setFormat] = useState<Format>("csv");
  const [range, setRange] = useState<Range>("30d");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleExport() {
    setLoading(true);
    setError(null);

    const { start, end } = getRangeDates(range);

    try {
      const res = await fetch("/api/reports/carbon", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          format,
          start_date: start,
          end_date: end,
          cluster_tag: clusterTag,
        }),
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error ?? `HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const ext = format;
      const slug = `${start.slice(0, 10)}-${end.slice(0, 10)}`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `nemulai-carbon-${slug}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="border border-neutral-800 rounded-lg p-5 max-w-lg">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-neutral-200">Carbon Report Export</h3>
        <span className="text-xs text-neutral-500">GHG Protocol Scope 2 · EU AI Act · SB 253 · SEC</span>
      </div>

      <div className="flex gap-3 mb-4 flex-wrap">
        {/* Date range */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-500">Date range</label>
          <select
            value={range}
            onChange={(e) => setRange(e.target.value as Range)}
            className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs text-neutral-200 focus:outline-none focus:ring-1 focus:ring-neutral-500"
          >
            {(Object.keys(RANGE_LABELS) as Range[]).map((r) => (
              <option key={r} value={r}>{RANGE_LABELS[r]}</option>
            ))}
          </select>
        </div>

        {/* Format */}
        <div className="flex flex-col gap-1">
          <label className="text-xs text-neutral-500">Format</label>
          <select
            value={format}
            onChange={(e) => setFormat(e.target.value as Format)}
            className="bg-neutral-900 border border-neutral-700 rounded px-2 py-1 text-xs text-neutral-200 focus:outline-none focus:ring-1 focus:ring-neutral-500"
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
            <option value="pdf">PDF</option>
          </select>
        </div>

        <div className="flex flex-col gap-1 justify-end">
          <button
            onClick={handleExport}
            disabled={loading}
            className="bg-green-600 hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-medium px-4 py-1 rounded transition-colors"
          >
            {loading ? "Generating…" : "Export Report"}
          </button>
        </div>
      </div>

      {error && (
        <p className="text-xs text-red-400 mt-1">{error}</p>
      )}

      <p className="text-xs text-neutral-600">
        Includes per-job kWh, gCO₂eq, and cost with GHG Protocol methodology metadata.
        {format === "pdf" && " PDF includes cover page, executive summary, and methodology section."}
      </p>
    </div>
  );
}
