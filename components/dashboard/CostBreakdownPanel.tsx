"use client";

import { useState, useEffect } from "react";

type Dimension = "team" | "model" | "gpu" | "cluster";

interface BreakdownItem {
  dimension_value: string;
  total_kwh: number;
  cost_usd: number;
  pct_of_total: number;
  job_count: number;
}

interface CostBreakdownPanelProps {
  clusterParam?: string;
  from?: string;
  to?: string;
}

const DIMENSION_LABELS: Record<Dimension, string> = {
  team: "By Team",
  model: "By Model",
  gpu: "By GPU",
  cluster: "By Cluster",
};

export default function CostBreakdownPanel({
  clusterParam,
  from,
  to,
}: CostBreakdownPanelProps) {
  const [dimension, setDimension] = useState<Dimension>("gpu");
  const [items, setItems] = useState<BreakdownItem[]>([]);
  const [totalCost, setTotalCost] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);

    const params = new URLSearchParams({ dimension });
    if (clusterParam) params.set("cluster_tag", clusterParam);
    if (from) params.set("from", from);
    if (to) params.set("to", to);

    fetch(`/api/cost/breakdown?${params}`)
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.json();
      })
      .then((data) => {
        setItems(data.breakdown ?? []);
        setTotalCost(data.total_cost_usd ?? 0);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, [dimension, clusterParam, from, to]);

  return (
    <div className="rounded-xl bg-neutral-900 border border-neutral-800 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium text-neutral-200">
          Cost Breakdown
          {totalCost > 0 && (
            <span className="text-neutral-500 font-normal ml-2">
              ${totalCost.toFixed(2)} total
            </span>
          )}
        </h3>
        <div className="flex gap-1">
          {(Object.keys(DIMENSION_LABELS) as Dimension[]).map((d) => (
            <button
              key={d}
              onClick={() => setDimension(d)}
              className={`px-2 py-0.5 text-xs rounded font-medium ${
                dimension === d
                  ? "bg-indigo-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
              }`}
            >
              {DIMENSION_LABELS[d]}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="space-y-2 animate-pulse">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-8 bg-neutral-800 rounded" />
          ))}
        </div>
      ) : error ? (
        <p className="text-red-400 text-sm">Failed to load breakdown.</p>
      ) : items.length === 0 ? (
        <p className="text-neutral-500 text-sm text-center py-6">No data for this dimension.</p>
      ) : (
        <div className="space-y-2">
          {items.slice(0, 10).map((item) => (
            <div key={item.dimension_value}>
              <div className="flex items-center justify-between text-xs mb-0.5">
                <span className="text-neutral-300 truncate max-w-[50%]">
                  {item.dimension_value}
                </span>
                <span className="text-neutral-400">
                  ${item.cost_usd.toFixed(2)}{" "}
                  <span className="text-neutral-600">({item.pct_of_total}%)</span>
                </span>
              </div>
              <div className="w-full bg-neutral-800 rounded-full h-2">
                <div
                  className="bg-indigo-500 h-2 rounded-full transition-all"
                  style={{ width: `${Math.min(item.pct_of_total, 100)}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
