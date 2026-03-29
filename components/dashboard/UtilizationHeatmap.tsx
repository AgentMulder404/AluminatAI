"use client";

import { useState, useEffect } from "react";

interface MetricPoint {
  time: string;
  utilization_gpu_pct: number;
  power_draw_w: number;
}

interface UtilizationHeatmapProps {
  clusterParam?: string;
}

function utilColor(pct: number): string {
  if (pct >= 70) return "bg-green-600";
  if (pct >= 30) return "bg-amber-600";
  if (pct > 0) return "bg-red-600";
  return "bg-neutral-800";
}

export default function UtilizationHeatmap({ clusterParam }: UtilizationHeatmapProps) {
  const [grid, setGrid] = useState<Map<string, number[]>>(new Map());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);

    const params = new URLSearchParams();
    if (clusterParam) params.set("cluster_tag", clusterParam);

    fetch(`/api/dashboard/utilization-chart?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((data: MetricPoint[]) => {
        // Bucket by hour (0-23) — aggregate utilization per hour
        // We create a single row for now (all GPUs averaged)
        const hourBuckets = new Array(24).fill(0);
        const hourCounts = new Array(24).fill(0);

        for (const point of data) {
          const hour = new Date(point.time).getUTCHours();
          hourBuckets[hour] += point.utilization_gpu_pct;
          hourCounts[hour]++;
        }

        const avgByHour = hourBuckets.map((sum, i) =>
          hourCounts[i] > 0 ? Math.round(sum / hourCounts[i]) : 0
        );

        const m = new Map<string, number[]>();
        m.set("All GPUs", avgByHour);
        setGrid(m);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [clusterParam]);

  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5 animate-pulse">
        <div className="h-4 bg-neutral-800 rounded w-40 mb-3" />
        <div className="h-16 bg-neutral-800 rounded" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
        <p className="text-red-400 text-sm">Failed to load utilization data.</p>
      </div>
    );
  }

  if (grid.size === 0) return null;

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <h3 className="text-sm font-medium text-neutral-200 mb-3">
        GPU Utilization — Last 24h by Hour (UTC)
      </h3>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <th className="text-xs text-neutral-500 text-left pr-3 pb-1">GPU</th>
              {Array.from({ length: 24 }, (_, i) => (
                <th key={i} className="text-[10px] text-neutral-500 px-0.5 pb-1 min-w-[20px]">
                  {i}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[...grid.entries()].map(([label, hours]) => (
              <tr key={label}>
                <td className="text-xs text-neutral-400 pr-3 py-0.5 whitespace-nowrap">
                  {label}
                </td>
                {hours.map((pct, i) => (
                  <td key={i} className="px-0.5 py-0.5">
                    <div
                      className={`w-5 h-5 rounded-sm ${utilColor(pct)} flex items-center justify-center`}
                      title={`${label} @ ${i}:00 UTC — ${pct}%`}
                    >
                      <span className="text-[8px] text-white/70 font-mono">
                        {pct > 0 ? pct : ""}
                      </span>
                    </div>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center gap-3 mt-3 text-[10px] text-neutral-500">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-green-600" /> 70%+
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-amber-600" /> 30-70%
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-red-600" /> &lt;30%
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-sm bg-neutral-800" /> Idle
        </span>
      </div>
    </div>
  );
}
