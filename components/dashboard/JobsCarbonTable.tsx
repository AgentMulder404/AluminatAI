"use client";

import { useEffect, useState } from "react";

interface Job {
  job_id: string;
  team_id: string | null;
  model_tag: string | null;
  gpu_name: string | null;
  gpu_count: number;
  total_energy_j: number;
  total_kwh: number;
  cost_usd: number;
  total_co2e_g: number | null;
  grid_zone: string | null;
  start_time: string;
  end_time: string;
}

function formatCo2e(g: number | null): { text: string; color: string } {
  if (g === null) return { text: "—", color: "text-neutral-500" };
  if (g < 1000) return { text: `${g.toFixed(1)}g`, color: g < 100 ? "text-green-400" : "text-amber-400" };
  return { text: `${(g / 1000).toFixed(2)}kg`, color: g < 1000 ? "text-amber-400" : "text-red-400" };
}

function formatKwh(kwh: number): string {
  if (kwh < 0.001) return `${(kwh * 1e6).toFixed(0)}mJ`;
  if (kwh < 1) return `${(kwh * 1000).toFixed(1)}Wh`;
  return `${kwh.toFixed(3)} kWh`;
}

function gpuHours(job: Job): string {
  const start = new Date(job.start_time).getTime();
  const end = new Date(job.end_time).getTime();
  const hrs = (end - start) / 3_600_000;
  return `${hrs.toFixed(2)}h`;
}

export default function JobsCarbonTable({
  clusterParam = "",
  onJobClick,
}: {
  clusterParam?: string;
  onJobClick?: (jobId: string) => void;
}) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setError(false);
    const url = `/api/dashboard/jobs${clusterParam ? `?${clusterParam}` : ""}`;
    fetch(url)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((d) => setJobs(Array.isArray(d) ? d.slice(0, 50) : []))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [clusterParam]);

  if (loading) {
    return (
      <div className="border border-neutral-800 rounded-lg p-5">
        <div className="h-4 w-32 bg-neutral-800 rounded animate-pulse mb-3" />
        <div className="h-24 bg-neutral-900 rounded animate-pulse" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="border border-neutral-800 rounded-lg p-5">
        <p className="text-red-400 text-sm">Failed to load jobs.</p>
      </div>
    );
  }

  if (jobs.length === 0) return null;

  const hasCarbon = jobs.some((j) => j.total_co2e_g !== null);

  return (
    <div className="border border-neutral-800 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-5 py-3 border-b border-neutral-800">
        <h2 className="text-sm font-semibold text-neutral-200">Jobs</h2>
        {!hasCarbon && (
          <span className="text-xs text-neutral-500">
            Set <code className="text-neutral-400">ALUMINATAI_GRID_ZONE</code> to track CO₂e
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-neutral-800 text-neutral-400">
              <th className="text-left px-4 py-2 font-medium">Job</th>
              <th className="text-left px-4 py-2 font-medium">Model</th>
              <th className="text-left px-4 py-2 font-medium">GPU</th>
              <th className="text-right px-4 py-2 font-medium">GPU-hrs</th>
              <th className="text-right px-4 py-2 font-medium">Energy</th>
              <th className="text-right px-4 py-2 font-medium">Cost</th>
              <th className="text-right px-4 py-2 font-medium">CO₂e</th>
              <th className="text-left px-4 py-2 font-medium">Zone</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => {
              const co2 = formatCo2e(job.total_co2e_g);
              return (
                <tr
                  key={job.job_id}
                  className={`border-b border-neutral-900 hover:bg-neutral-900/50 transition-colors ${onJobClick ? "cursor-pointer" : ""}`}
                  onClick={() => onJobClick?.(job.job_id)}
                >
                  <td className="px-4 py-2 font-mono text-neutral-300 max-w-[140px] truncate">
                    {job.job_id}
                  </td>
                  <td className="px-4 py-2 text-neutral-400">
                    {job.model_tag ?? "—"}
                  </td>
                  <td className="px-4 py-2 text-neutral-400">
                    {job.gpu_name ?? "—"}
                    {job.gpu_count > 1 && (
                      <span className="ml-1 text-neutral-600">×{job.gpu_count}</span>
                    )}
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {gpuHours(job)}
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    {formatKwh(job.total_kwh)}
                  </td>
                  <td className="px-4 py-2 text-right text-neutral-300">
                    ${job.cost_usd.toFixed(4)}
                  </td>
                  <td className={`px-4 py-2 text-right ${co2.color}`}>
                    {co2.text}
                  </td>
                  <td className="px-4 py-2 text-neutral-500 font-mono">
                    {job.grid_zone ?? "—"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
