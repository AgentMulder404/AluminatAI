"use client";

import { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

type Granularity = "day" | "week" | "month";

interface TrendPoint {
  period: string;
  cost_usd: number;
  kwh: number;
  co2e_g: number | null;
  cloud_equivalent_usd?: number;
}

interface TrendData {
  points: TrendPoint[];
  projection: { end_of_month_usd: number; trend_pct: number } | null;
}

interface CostTrendChartProps {
  clusterParam?: string;
  from?: string;
  to?: string;
}

export default function CostTrendChart({ clusterParam, from, to }: CostTrendChartProps) {
  const [granularity, setGranularity] = useState<Granularity>("day");
  const [showCloud, setShowCloud] = useState(true);
  const [data, setData] = useState<TrendData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);

    const params = new URLSearchParams({ granularity });
    if (clusterParam) params.set("cluster_tag", clusterParam);
    if (from) params.set("from", from);
    if (to) params.set("to", to);
    if (showCloud) params.set("include_cloud", "true");

    fetch(`/api/cost/trend?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((d) => setData(d))
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [granularity, clusterParam, from, to, showCloud]);

  if (loading) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5 h-72 animate-pulse">
        <div className="h-4 bg-neutral-800 rounded w-32 mb-4" />
        <div className="h-48 bg-neutral-800 rounded" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
        <p className="text-red-400 text-sm">Failed to load cost trend.</p>
      </div>
    );
  }

  const points = data?.points ?? [];
  const hasCloudData = points.some((p) => p.cloud_equivalent_usd != null);

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-medium text-neutral-200">Cost Trend</h3>
          {data?.projection && (
            <p className="text-xs text-neutral-500 mt-0.5">
              Projected end-of-month: ${data.projection.end_of_month_usd.toFixed(2)}
              <span
                className={`ml-1 ${
                  data.projection.trend_pct > 0 ? "text-red-400" : "text-green-400"
                }`}
              >
                ({data.projection.trend_pct > 0 ? "+" : ""}
                {data.projection.trend_pct}%)
              </span>
            </p>
          )}
        </div>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showCloud}
              onChange={(e) => setShowCloud(e.target.checked)}
              className="rounded border-neutral-600 bg-neutral-800 text-green-500 focus:ring-green-500 h-3 w-3"
            />
            <span className="text-xs text-neutral-400">Cloud comparison</span>
          </label>
          <div className="flex gap-1">
            {(["day", "week", "month"] as Granularity[]).map((g) => (
              <button
                key={g}
                onClick={() => setGranularity(g)}
                className={`px-2 py-0.5 text-xs rounded font-medium ${
                  granularity === g
                    ? "bg-indigo-600 text-white"
                    : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
                }`}
              >
                {g === "day" ? "Daily" : g === "week" ? "Weekly" : "Monthly"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {points.length === 0 ? (
        <p className="text-neutral-500 text-sm text-center py-12">No cost data yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={points}>
            <defs>
              <linearGradient id="costGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="cloudGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#737373" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#737373" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
            <XAxis
              dataKey="period"
              tick={{ fill: "#737373", fontSize: 11 }}
              axisLine={{ stroke: "#404040" }}
              tickLine={false}
              tickFormatter={(v: string) => {
                if (granularity === "day") return v.slice(5); // MM-DD
                if (granularity === "week") return `W${v.slice(5)}`;
                return v; // YYYY-MM
              }}
            />
            <YAxis
              tick={{ fill: "#737373", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v: number) => `$${v}`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#171717",
                border: "1px solid #404040",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "#a3a3a3" }}
              formatter={(v: number, name: string) => [
                `$${v.toFixed(2)}`,
                name === "cloud_equivalent_usd" ? "Cloud Equivalent" : "Your Cost",
              ]}
            />
            {showCloud && hasCloudData && (
              <Area
                type="monotone"
                dataKey="cloud_equivalent_usd"
                stroke="#737373"
                strokeWidth={1.5}
                strokeDasharray="5 3"
                fill="url(#cloudGradient)"
              />
            )}
            <Area
              type="monotone"
              dataKey="cost_usd"
              stroke="#6366f1"
              strokeWidth={2}
              fill="url(#costGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
