"use client";

import { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface JobDetail {
  job_id: string;
  team_id: string | null;
  model_tag: string | null;
  scheduler_source: string | null;
  cluster_tag: string | null;
  gpu_count: number;
  gpu_name: string | null;
  start_time: string;
  end_time: string;
  summary: {
    total_kwh: number;
    cost_usd: number;
    total_co2e_g: number;
    max_power_w: number;
    peak_utilization_pct: number;
    sample_count: number;
  };
  timeseries: Array<{
    time: string;
    power_draw_w: number;
    utilization_gpu_pct: number;
    temperature_c: number | null;
    memory_used_mb: number | null;
  }>;
}

interface JobDetailDrawerProps {
  jobId: string | null;
  onClose: () => void;
}

export default function JobDetailDrawer({ jobId, onClose }: JobDetailDrawerProps) {
  const [data, setData] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    if (!jobId) {
      setData(null);
      return;
    }

    setLoading(true);
    setError(false);

    fetch(`/api/dashboard/jobs/${encodeURIComponent(jobId)}`)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((d) => setData(d))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [jobId]);

  if (!jobId) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
      />

      {/* Drawer */}
      <div className="fixed right-0 top-0 h-full w-full max-w-2xl bg-neutral-950 border-l border-neutral-800 z-50 overflow-y-auto">
        <div className="p-6">
          {/* Header */}
          <div className="flex items-start justify-between mb-6">
            <div>
              <h2 className="text-lg font-semibold text-neutral-100">Job Detail</h2>
              <p className="text-sm text-neutral-400 font-mono mt-0.5">
                {jobId.length > 32 ? `${jobId.slice(0, 32)}...` : jobId}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-neutral-500 hover:text-neutral-300 text-xl"
            >
              ✕
            </button>
          </div>

          {loading && (
            <div className="space-y-4 animate-pulse">
              <div className="h-20 bg-neutral-900 rounded-lg" />
              <div className="h-48 bg-neutral-900 rounded-lg" />
            </div>
          )}

          {error && (
            <p className="text-red-400 text-sm">Failed to load job details.</p>
          )}

          {data && (
            <>
              {/* Metadata */}
              <div className="grid grid-cols-2 gap-3 mb-6">
                {[
                  { label: "Team", value: data.team_id ?? "—" },
                  { label: "Model", value: data.model_tag ?? "—" },
                  { label: "GPU", value: `${data.gpu_count}x ${data.gpu_name ?? "—"}` },
                  { label: "Scheduler", value: data.scheduler_source ?? "—" },
                  { label: "Cluster", value: data.cluster_tag ?? "—" },
                  { label: "Duration", value: formatDuration(data.start_time, data.end_time) },
                ].map((item) => (
                  <div key={item.label} className="bg-neutral-900 rounded-lg px-3 py-2">
                    <p className="text-xs text-neutral-500">{item.label}</p>
                    <p className="text-sm text-neutral-200 truncate">{item.value}</p>
                  </div>
                ))}
              </div>

              {/* Summary Stats */}
              <div className="grid grid-cols-4 gap-3 mb-6">
                {[
                  { label: "Cost", value: `$${data.summary.cost_usd.toFixed(2)}` },
                  { label: "Energy", value: `${data.summary.total_kwh.toFixed(3)} kWh` },
                  { label: "Peak Power", value: `${data.summary.max_power_w}W` },
                  { label: "CO2e", value: `${data.summary.total_co2e_g.toFixed(1)}g` },
                ].map((stat) => (
                  <div key={stat.label} className="text-center">
                    <p className="text-xs text-neutral-500">{stat.label}</p>
                    <p className="text-sm font-semibold text-neutral-100">{stat.value}</p>
                  </div>
                ))}
              </div>

              {/* Power Timeline */}
              <div className="mb-6">
                <h3 className="text-sm font-medium text-neutral-200 mb-3">
                  Power &amp; Utilization Timeline
                </h3>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={data.timeseries}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                    <XAxis
                      dataKey="time"
                      tick={{ fill: "#737373", fontSize: 10 }}
                      axisLine={{ stroke: "#404040" }}
                      tickLine={false}
                      tickFormatter={(v: string) => new Date(v).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    />
                    <YAxis
                      yAxisId="power"
                      tick={{ fill: "#737373", fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                      tickFormatter={(v: number) => `${v}W`}
                    />
                    <YAxis
                      yAxisId="util"
                      orientation="right"
                      tick={{ fill: "#737373", fontSize: 10 }}
                      axisLine={false}
                      tickLine={false}
                      domain={[0, 100]}
                      tickFormatter={(v: number) => `${v}%`}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "#171717",
                        border: "1px solid #404040",
                        borderRadius: "8px",
                        fontSize: "11px",
                      }}
                      labelFormatter={(v: string) => new Date(v).toLocaleString()}
                    />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Line
                      yAxisId="power"
                      type="monotone"
                      dataKey="power_draw_w"
                      name="Power (W)"
                      stroke="#6366f1"
                      strokeWidth={1.5}
                      dot={false}
                    />
                    <Line
                      yAxisId="util"
                      type="monotone"
                      dataKey="utilization_gpu_pct"
                      name="GPU Util (%)"
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Temperature & Memory (if available) */}
              {data.timeseries.some((t) => t.temperature_c != null) && (
                <div>
                  <h3 className="text-sm font-medium text-neutral-200 mb-3">
                    Temperature &amp; Memory
                  </h3>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={data.timeseries}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                      <XAxis
                        dataKey="time"
                        tick={{ fill: "#737373", fontSize: 10 }}
                        axisLine={{ stroke: "#404040" }}
                        tickLine={false}
                        tickFormatter={(v: string) => new Date(v).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                      />
                      <YAxis
                        yAxisId="temp"
                        tick={{ fill: "#737373", fontSize: 10 }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(v: number) => `${v}C`}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#171717",
                          border: "1px solid #404040",
                          borderRadius: "8px",
                          fontSize: "11px",
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: "11px" }} />
                      <Line
                        yAxisId="temp"
                        type="monotone"
                        dataKey="temperature_c"
                        name="Temp (C)"
                        stroke="#f59e0b"
                        strokeWidth={1.5}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </>
  );
}

function formatDuration(start: string, end: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime();
  const hours = Math.floor(ms / 3_600_000);
  const mins = Math.floor((ms % 3_600_000) / 60_000);
  if (hours > 0) return `${hours}h ${mins}m`;
  return `${mins}m`;
}
